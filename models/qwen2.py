# Copyright © 2023-2024 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen2 import ModelArgs, TransformerBlock

class IdentityBlock(nn.Module):
    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[KVCache] = None) -> mx.array:
        return x
@dataclass
class ModelArgs(ModelArgs):
    start_layer: int = 0
    end_layer: int = 35

class Qwen2Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        self.start_layer = args.start_layer
        self.end_layer = args.end_layer
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = []
        for i in range(self.num_hidden_layers):
            if self.start_layer <= i < self.end_layer:
                self.layers.append(TransformerBlock(args=args))
            else:
                self.layers.append(IdentityBlock())

        if self.end_layer == self.num_hidden_layers:
            self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        if self.start_layer == 0:
            h = self.embed_tokens(inputs)
            h = h * (self.args.hidden_size ** 0.5)
        else:
            h = inputs

        if mask is None:
            mask = create_attention_mask(h, cache)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        if self.end_layer == self.num_hidden_layers:
            h = self.norm(h)

        return h

class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.start_layer = args.start_layer
        self.end_layer = args.end_layer
        self.model = Qwen2Model(args)
        if self.end_layer == self.args.num_hidden_layers:
            self.lm_head = nn.Linear(
                args.hidden_size, args.vocab_size, bias=False)
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        out = self.model(inputs, mask, cache)
        if self.end_layer == self.args.num_hidden_layers:
            if self.args.tie_word_embeddings:
                out = self.model.embed_tokens.as_linear(out)
            else:
                out = self.lm_head(out)
            return out
        else:
            return out

    def sanitize(self, weights):
        total_layers = len(self.layers)
        shard_state_dict = {}
        for key, value in weights.items():
            if "self_attn.rotary_emb.inv_freq" in key:
                continue
            if key.startswith('model.layers.'):
                layer_num = int(key.split('.')[2])
                if self.start_layer <= layer_num < self.end_layer:
                    shard_state_dict[key] = value
            elif (self.start_layer == 0 or self.end_layer == total_layers) and key.startswith('model.embed_tokens'):
                shard_state_dict[key] = value
            elif self.end_layer == total_layers and (key.startswith('model.norm') or key.startswith('lm_head')):
                shard_state_dict[key] = value

        return shard_state_dict

    @property
    def layers(self):
        return self.model.layers

    @property
    def n_kv_heads(self):
        return self.args.num_key_value_heads

    @property
    def head_dim(self):
        return self.args.hidden_size // self.args.num_attention_heads