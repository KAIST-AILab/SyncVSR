from __future__ import annotations

import argparse

import flax
import flax.linen as nn
import flax.linen.initializers as init
import jax
import jax.numpy as jnp
import optax
from chex import Array, ArrayDType, ArrayTree, PRNGKey, Shape


class CutMix(nn.Module):
    cutmix_alpha: float = 1.0

    def __call__(self, inputs: Array, labels: Array, audio_tensors: Array) -> tuple[Array, Array, Array]:
        if self.cutmix_alpha == 0:
            return inputs, labels, audio_tensors

        ratio = jax.random.beta(self.make_rng("mixup"), *(self.cutmix_alpha,) * 2)
        start = (1 - ratio) * jax.random.uniform(self.make_rng("mixup"))

        mask = jnp.linspace(0, 1, inputs.shape[1])
        mask = ~((start < mask) & (mask <= start + ratio))
        
        audio_mask = jnp.repeat(mask, audio_tensors.shape[1] // inputs.shape[1], axis=0) # repeat elements for audio alignment

        inputs = jnp.where(mask[None, :, None], inputs, jnp.flip(inputs, axis=0))
        labels = mask.mean() * labels + (1 - mask.mean()) * jnp.flip(labels, axis=0)

        audio_tensors = audio_mask[None, :, None] * audio_tensors + (1 - audio_mask[None, :, None]) * jnp.flip(audio_tensors, axis=0)
        return inputs, labels, audio_tensors


class CutMixWB(nn.Module):
    cutmix_alpha: float = 1.0

    def __call__(self, inputs: Array, labels: Array, audio_tensors: Array, word_mask: Array) -> tuple[Array, Array, Array, Array]:
        if self.cutmix_alpha == 0:
            return inputs, labels, audio_tensors

        ratio = jax.random.beta(self.make_rng("mixup"), *(self.cutmix_alpha,) * 2)
        start = (1 - ratio) * jax.random.uniform(self.make_rng("mixup"))

        mask = jnp.linspace(0, 1, inputs.shape[1])
        mask = ~((start < mask) & (mask <= start + ratio))
        
        audio_mask = jnp.repeat(mask, audio_tensors.shape[1] // inputs.shape[1], axis=0) # repeat elements for audio alignment

        inputs = jnp.where(mask[None, :, None], inputs, jnp.flip(inputs, axis=0))
        labels = mask.mean() * labels + (1 - mask.mean()) * jnp.flip(labels, axis=0)
        word_mask = mask.mean() * word_mask + (1 - mask.mean()) * jnp.flip(word_mask, axis=0)

        audio_tensors = audio_mask[None, :, None] * audio_tensors + (1 - audio_mask[None, :, None]) * jnp.flip(audio_tensors, axis=0)
        return inputs, labels, audio_tensors, word_mask


def load_pretrained_params(args: argparse.Namespace, params: ArrayTree) -> ArrayTree:
    with open(args.pretrained_ckpt, "rb") as fp:
        pretrained = flax.serialization.msgpack_restore(fp.read())
        pretrained = {"model": pretrained["student"]["encoder"]}
        pretrained = flax.traverse_util.flatten_dict(pretrained, sep=".")

    params = flax.traverse_util.flatten_dict(params, sep=".")
    intersection = len(set(params.keys()).intersection(pretrained.keys()))
    print(f"[*] Load {intersection}/{len(params)} params from pretrained model")

    params = dict(params, **pretrained)
    params = flax.traverse_util.unflatten_dict(params, sep=".")
    return params


def trunc_normal(stddev: float = 0.02) -> init.Initializer:
    def init(key: PRNGKey, shape: Shape, dtype: ArrayDType = jnp.float32) -> Array:
        return jax.random.truncated_normal(key, -2, 2, shape, dtype) * stddev

    return init
