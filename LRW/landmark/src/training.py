# Copyright 2023 Jungwoo Park
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
from functools import partial

import flax
import flax.linen as nn
import flax.struct
import jax
import jax.numpy as jnp
import optax
from chex import Array, ArrayTree, PRNGKey
from flax.training import train_state
from flax.training.common_utils import shard_prng_key
from jax.tree_util import tree_map_with_path

from modeling import Transformer, trunc_normal
from utils import CutMix, CutMixWB, load_pretrained_params

Dense = partial(nn.Dense, kernel_init=trunc_normal())

class TrainModule(nn.Module):
    model: Transformer
    cutmix: CutMix
    label_smoothing: float = 0.0
    audio_alignment: int = 4
    vq_groups: int = 2
    audio_vocab_size: int = 320
    cmts_lambda: float = 1.0
    use_word_boundary: bool = False

    def setup(self):
        self.audio_classifier = Dense(self.audio_alignment * self.vq_groups * self.audio_vocab_size)

    def __call__(self, inputs: Array, labels: Array, audio_tokens, word_mask: Array = None, det: bool = True) -> ArrayTree:
        B, seq_len, _ = inputs.shape

        inputs = jnp.where(inputs == -100.0, 0, inputs)
        labels = nn.one_hot(labels, self.model.labels)
        audio_tokens = audio_tokens[:, : seq_len * self.audio_alignment, :] # [B, seq_len * self.audio_alignment, self.vq_groups]

        if not det:
            labels = optax.smooth_labels(labels, self.label_smoothing)
            if self.use_word_boundary:
                inputs, labels, audio_tokens, word_mask = self.cutmix(inputs, labels, audio_tokens, word_mask)
            elif not self.use_word_boundary:
                inputs, labels, audio_tokens = self.cutmix(inputs, labels, audio_tokens)

        # get classification loss
        if self.use_word_boundary:
            logits_pooling, seq_outputs = self.model(inputs, word_mask=word_mask, det=det)
        elif not self.use_word_boundary:
            logits_pooling, seq_outputs = self.model(inputs, det=det)
        
        loss_pooling = optax.softmax_cross_entropy(logits_pooling, labels).mean()

        # get CMTS auxiliary loss
        logits_audio = self.audio_classifier(seq_outputs) # [B, seq_len, dim]
        logits_audio = logits_audio.reshape((B, seq_len, self.audio_alignment * self.vq_groups, self.audio_vocab_size))
        loss_audio = optax.softmax_cross_entropy_with_integer_labels(
            logits_audio.reshape((B * seq_len * self.audio_alignment * self.vq_groups, self.audio_vocab_size)),
            audio_tokens.flatten(),
        ).mean()

        loss = loss_pooling + self.cmts_lambda * loss_audio

        labels = jnp.argmax(labels, axis=-1)
        acc1 = (jnp.argmax(logits_pooling, axis=-1) == labels).mean()
        acc5 = (jax.lax.top_k(logits_pooling, k=5)[1] == labels[:, None]).any(1).mean()
        return {"loss": loss, "acc1": acc1, "acc5": acc5, "loss_pooling": loss_pooling, "loss_audio": loss_audio}


class TrainState(train_state.TrainState):
    mixup_rng: PRNGKey
    dropout_rng: PRNGKey

    def replicate(self) -> TrainState:
        return flax.jax_utils.replicate(self).replace(
            mixup_rng=shard_prng_key(self.mixup_rng),
            dropout_rng=shard_prng_key(self.dropout_rng),
        )


@partial(jax.pmap, axis_name="batch", donate_argnums=(0,))
def training_step(state: TrainState, batch: ArrayTree) -> tuple[TrainState, ArrayTree]:
    mixup_rng, new_mixup_rng = jax.random.split(state.mixup_rng)
    dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)
    rngs = {"mixup": mixup_rng, "dropout": dropout_rng}

    def loss_fn(params: ArrayTree) -> tuple[Array, ArrayTree]:
        metrics = state.apply_fn({"params": params}, *batch, det=False, rngs=rngs)
        return metrics["loss"], metrics

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    metrics, grads = jax.lax.pmean((metrics, grads), axis_name="batch")

    state = state.apply_gradients(
        grads=grads,
        mixup_rng=new_mixup_rng,
        dropout_rng=new_dropout_rng,
    )
    return state, {**metrics, "learning_rate": state.opt_state.hyperparams["lr"]}


@partial(jax.pmap, axis_name="batch")
def validation_step(
    state: TrainState, batch: ArrayTree
) -> tuple[TrainState, ArrayTree]:
    metrics = state.apply_fn({"params": state.params}, *batch, det=True)
    return jax.lax.pmean(metrics, axis_name="batch")


def create_train_state(args: argparse.Namespace, steps_per_epoch: int) -> TrainState:
    model = Transformer(
        layers=args.layers,
        dim=args.dim,
        heads=args.heads,
        labels=args.labels,
        emb_dropout=args.emb_dropout,
        msa_dropout=args.msa_dropout,
        mlp_dropout=args.mlp_dropout,
        droppath=args.droppath,
        use_word_boundary=args.use_word_boundary,
    )
    module = TrainModule(
        model=model, 
        cutmix=CutMix(args.cutmix_alpha) if not args.use_word_boundary else CutMixWB(args.cutmix_alpha),
        label_smoothing=args.label_smoothing,
        audio_alignment=args.audio_alignment,
        vq_groups=args.vq_groups,
        audio_vocab_size=args.audio_vocab_size,
        cmts_lambda=args.cmts_lambda,
        use_word_boundary=args.use_word_boundary,
    )

    # Initialize the model weights with dummy inputs. Using the init RNGS and inputs, we
    # will visualize the summary of the model including parameters.
    init_rngs = {"params": jax.random.PRNGKey(args.init_seed)}
    
    example_inputs = {
        "inputs": jnp.zeros((1, args.input_length, args.input_features)),
        "labels": jnp.zeros((1,), dtype=jnp.int32),
        "audio_tokens": jnp.zeros((1, 120, 2), dtype=jnp.int32),
    }
    if args.use_word_boundary:
        example_inputs["word_mask"] = jnp.zeros((1, args.input_length), dtype=jnp.int32)
    
    params = module.init(init_rngs, **example_inputs)["params"]
    print(module.tabulate(init_rngs, **example_inputs))

    if args.pretrained_ckpt is not None:
        params = load_pretrained_params(args, params)

    # Create learning rate scheduler and optimizer with gradient clipping. The learning
    # rate will be recorded into `hyperparams` by `optax.inject_hyperparameters`.
    @optax.inject_hyperparams
    def create_optimizer(lr: optax.Schedule) -> optax.GradientTransformation:
        optimizer = optax.adamw(
            learning_rate=lr,
            b1=args.adam_b1,
            b2=args.adam_b2,
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
            mask=partial(tree_map_with_path, lambda x, _: x[-1].key == "kernel"),
        )
        if args.max_norm > 0:
            optimizer = optax.chain(optax.clip_by_global_norm(args.max_norm), optimizer)
        return optimizer

    learning_rate = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=args.learning_rate,
        warmup_steps=args.warmup_epochs * steps_per_epoch,
        decay_steps=args.training_epochs * steps_per_epoch,
        end_value=1e-5,
    )
    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=create_optimizer(learning_rate),
        mixup_rng=jax.random.PRNGKey(args.mixup_seed),
        dropout_rng=jax.random.PRNGKey(args.dropout_seed),
    )
