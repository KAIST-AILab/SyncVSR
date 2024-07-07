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

from dataclasses import dataclass, fields
from functools import partial
from typing import Any

import flax.linen as nn
import jax.numpy as jnp
from chex import Array

from utils import trunc_normal

DenseGeneral = partial(nn.DenseGeneral, kernel_init=trunc_normal(0.02))
Dense = partial(nn.Dense, kernel_init=trunc_normal(0.02))
Conv = partial(nn.Conv, kernel_init=trunc_normal(0.02))


@dataclass
class TransformerBase:
    layers: int
    dim: int
    heads: int
    labels: int = 500

    emb_dropout: float = 0.1
    msa_dropout: float = 0.1
    mlp_dropout: float = 0.1
    droppath: float = 0.1

    use_word_boundary: bool = False

    @property
    def dim_backbone(self) -> int:
        return self.dim + 1 if self.use_word_boundary else self.dim

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    @property
    def hidden_dim(self) -> int:
        return 4 * self.dim

    @property
    def kwargs(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(TransformerBase)}


class Attention(TransformerBase, nn.Module):
    def setup(self):
        self.wq = DenseGeneral((self.heads, self.head_dim))
        self.wk = DenseGeneral((self.heads, self.head_dim))
        self.wv = DenseGeneral((self.heads, self.head_dim))
        self.wo = DenseGeneral(self.dim_backbone, axis=(-2, -1))
        self.drop = nn.Dropout(self.msa_dropout)

    def apply_rotary_embedding(self, x: Array, pos: Array) -> Array:
        freqs = 10000.0 ** -jnp.linspace(0, 1, x.shape[-1] // 2, endpoint=False)
        theta = pos[:, :, None, None] * freqs[None, None, None, :]

        cos, sin, (rx, ry) = jnp.cos(theta), jnp.sin(theta), jnp.split(x, 2, axis=-1)
        return jnp.concatenate((rx * cos - ry * sin, rx * sin + ry * cos), axis=-1)

    def __call__(self, x: Array, pos: Array, mask: Array, det: bool = True) -> Array:
        q = self.apply_rotary_embedding(self.wq(x), pos)
        k = self.apply_rotary_embedding(self.wk(x), pos)
        v = self.wv(x)

        x = jnp.einsum("bqhd,bkhd->bhqk", q / q.shape[-1] ** 0.5, k)
        x = jnp.einsum("bhqk,bkhd->bqhd", self.drop(nn.softmax(x + mask), det), v)
        return self.wo(x)


class FeedForward(TransformerBase, nn.Module):
    def setup(self):
        self.w1 = Dense(self.hidden_dim)
        self.w2 = Dense(self.dim_backbone)
        self.drop = nn.Dropout(self.mlp_dropout)

    def __call__(self, x: Array, det: bool = True) -> Array:
        return self.w2(self.drop(nn.gelu(self.w1(x)), det))


class TransformerLayer(TransformerBase, nn.Module):
    def setup(self):
        self.attn = Attention(**self.kwargs)
        self.ff = FeedForward(**self.kwargs)

        self.norm_attn = nn.LayerNorm()
        self.norm_ff = nn.LayerNorm()
        self.drop = nn.Dropout(self.droppath, broadcast_dims=(1, 2))

    def __call__(self, x: Array, pos: Array, mask: Array, det: bool = True) -> Array:
        x = x + self.drop(self.attn(self.norm_attn(x), pos, mask, det), det)
        x = x + self.drop(self.ff(self.norm_ff(x), det), det)
        return x


class Transformer(TransformerBase, nn.Module):
    def setup(self):
        self.wte = Conv(self.dim, kernel_size=(1,))
        self.cls_token = self.param("cls_token", trunc_normal(), (1, 1, self.dim_backbone))
        if self.use_word_boundary:
            self.cls_token = self.cls_token.at[:, :, -1].set(0.0)
        self.drop = nn.Dropout(self.emb_dropout)

        self.layer = [TransformerLayer(**self.kwargs) for _ in range(self.layers)]
        self.norm = nn.LayerNorm()
        self.head = Dense(self.labels)

    def __call__(self, x: Array, word_mask: Array = None, det: bool = True) -> tuple[Array, Array]:
        x = self.drop(self.wte(x), det)
        if self.use_word_boundary:
            x = jnp.concatenate((x, word_mask[:, :, None]), axis=-1)
        
        cls_token = jnp.broadcast_to(self.cls_token, (x.shape[0], 1, self.dim_backbone))
        x = jnp.concatenate((cls_token, x), axis=1)

        pos, mask = jnp.arange(x.shape[1])[None, :], 0
        for layer in self.layer:
            x = layer(x, pos, mask, det)
        return self.head(self.norm(x[:, 0, :])), x[:, 1:, :]