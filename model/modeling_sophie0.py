import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import transformers

from typing import Optional, Dict, Tuple, List, Union, Unpack, Sequence, Any
from flash_attn import (
    flash_attn_kvpacked_func,
    flash_attn_varlen_func
)
from flash_attn.layers.rotary import RotaryEmbedding, apply_rotary_emb
from flash_attn.ops.triton.layer_norm import RMSNorm
from flash_attn.modules.mlp import GatedMlp
from flash_attn.losses.cross_entropy import CrossEntropyLoss
from einops import rearrange
from itertools import chain
from flash_attn.bert_padding import unpad_input

from .configuration_sophie0 import Sophie0Config
from transformers.modeling_utils import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

#########################################################
#            --- basic functions ---
#########################################################
class Cache(transformers.cache_utils.Cache):
    """
    A cache used for storing hidden states produced by flash linear attention models.

    **Input:**
        - attn_state: Cache for standard attention, tuple(size(bsz, k_len/v_len, dmodel) * 2)
    """

    is_compileable = True

    def __init__(self, cache_position: int = 0):
        super().__init__()

        self.states: List[Dict[str, Any]] = []
        self._cache_position = [cache_position] # Used in `generate` to keep tally of how many tokens the cache has seen

    def __getitem__(self, layer_idx: int) -> Dict[str, Any]:
        if layer_idx < len(self):
            return self.states[layer_idx]
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        for state in self.states: yield state

    def __len__(self):
        return len(self.states)

    def update(
        self,
        attn_state: Tuple[torch.Tensor, torch.Tensor] = None,
        layer_idx: int = 0,
        offset: Optional[int] = 1,
        cache_kwargs: Optional[Dict[str, Any]] = {},
    ) -> Dict[str, Any]:
        """
        Updates the cache with the new `recurrent_state`/`attn_state`/`conv_state` for the layer `layer_idx`.

        Args:
            attn_state (`Tuple[torch.Tensor, torch.Tensor]`, `optional`):
                The new attention key/value states to cache.
            layer_idx (`int`, defaults to 0):
                The index of the layer to cache the states for.
            offset (`int`, `optional`, defaults to 1):
                The number of new tokens being processed.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass.

        Return:
            Dictionary of the updated state.
        """

        # Update the number of seen tokens
        if len(self._cache_position) <= layer_idx:
            self._cache_position.append(0)

        self._cache_position[layer_idx] += offset

        if attn_state is not None:
            input_size = attn_state[0].shape[-2]
            window_size = cache_kwargs.get('window_size', None)
            if not isinstance(attn_state, Tuple) or len(attn_state) != 2:
                raise ValueError("`attn_state` must be a tuple of two tensors for key/value states")
        if len(self.states) <= layer_idx:
            if attn_state is not None:
                if window_size is not None and input_size > window_size:
                    attn_state = (attn_state[0][..., -window_size:, :].contiguous(),
                                  attn_state[1][..., -window_size:, :].contiguous())
            state = dict(
                attn_state=attn_state,
            )
            self.states.append(state)
        else:
            state = self.states[layer_idx]
            if attn_state is not None:
                if state['attn_state'] is None:
                    if window_size is not None and input_size > window_size:
                        attn_state = (attn_state[0][..., -window_size:, :].contiguous(),
                                      attn_state[1][..., -window_size:, :].contiguous())
                else:
                    key_state, value_state = state['attn_state']
                    if window_size is not None and key_state.shape[-2] == window_size:
                        # DO NOT allocate new memory if the cache is full
                        # roll the key/value states to the left by `input_size`
                        key_state = key_state.roll(-input_size, -2)
                        value_state = value_state.roll(-input_size, -2)
                        # replace the last `input_size` tokens with the new key/value states
                        key_state[..., -input_size:, :] = attn_state[0]
                        value_state[..., -input_size:, :] = attn_state[1]
                        attn_state = (key_state, value_state)
                    else:
                        attn_state = (torch.cat([key_state, attn_state[0]], -2),
                                      torch.cat([value_state, attn_state[1]], -2),)
                state['attn_state'] = attn_state

        return state

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.states) <= layer_idx:
            return 0
        return self._cache_position[layer_idx]

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. Cache does not have a maximum length."""
        return None

    def to_legacy_cache(self) -> Tuple:
        return tuple(self.states)
    
    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.states)):
            for k in self.states[layer_idx].keys():
                if isinstance(self.states[layer_idx][k], torch.Tensor):
                    device = self.states[layer_idx][k].device
                    self.states[layer_idx][k] = self.states[layer_idx][k].index_select(0, beam_idx.to(device))
                elif isinstance(self.states[layer_idx][k], Tuple):
                    _temp = []
                    for i in range(len(self.states[layer_idx][k])):
                        device = self.states[layer_idx][k][i].device
                        _temp.append(self.states[layer_idx][k][i].index_select(0, beam_idx.to(device)))
                    self.states[layer_idx][k] = tuple(_temp)

    @classmethod
    @torch.compiler.disable
    def from_legacy_cache(
        cls,
        past_key_values: Optional[Tuple] = None,
        cache_position: int = 0
    ):
        """Converts a cache in the legacy cache format into an equivalent `Cache`."""

        cache = cls(cache_position)
        if isinstance(past_key_values, list):
            for layer_idx in range(len(past_key_values)):
                cache.states.append(past_key_values[layer_idx])
        return cache

class VarlenCache(transformers.cache_utils.Cache):
    """
    A varlen cache used for storing hidden states produced by varlen batch inference.

    **Input:**
        - attn_state: Cache for standard attention, tuple(size(total_nnz, dmodel) * 2)
    """

    is_compileable = True

    def __init__(self, cache_position: int = 0, batch_size: int = 1, device: str | torch.device = None):
        super().__init__()

        self.states: List[Dict[str, Any]] = []
        self._cache_position = [torch.full((batch_size,), cache_position, dtype=torch.int64, device=device)] # Used in `generate` to keep tally of how many tokens the cache has seen
        self.batch_size =  batch_size
        self.device = device
    
    def __getitem__(self, layer_idx: int) -> Dict[str, Any]:
        if layer_idx < len(self):
            return self.states[layer_idx]
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

    def __iter__(self):
        for state in self.states: yield state

    def __len__(self):
        return len(self.states)
    
    def update(
        self,
        attn_state: Tuple[torch.Tensor, torch.Tensor] = None,
        cu_seqlens: torch.LongTensor = None,
        layer_idx: int = 0,
        cache_kwargs: Optional[Dict[str, Any]] = {},
    ) -> Dict[str, Any]:
        """
        Updates the cache with the new `attn_state` for the layer `layer_idx`.

        Args:
            attn_state (`Tuple[torch.Tensor, torch.Tensor]`, `optional`):
                The new attention key/value states to cache, sizes (total_nnz, hidden_size)
            cu_seqlens (`torch.LongTensor`):
                the accumulated sequence length for current states, sizes (bsz + 1,)
            layer_idx (`int`, defaults to 0):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass.

        Return:
            Dictionary of the updated state.
        """

        if attn_state is not None:
            if not isinstance(attn_state, Tuple) or len(attn_state) != 2:
                raise ValueError("`attn_state` must be a tuple of two tensors for key/value states")
        
        dtype = attn_state[0].dtype
        device = attn_state[0].device
        hidden_size = attn_state[0].size(-1)

        # Case 1: prefill at the 1st step
        if len(self._cache_position) <= layer_idx:
            self._cache_position.append(
                torch.zeros((cu_seqlens.size(0) - 1,), dtype=torch.int64, device=cu_seqlens.device)
            )

        kv_seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        kv_seqlens_cpu = kv_seqlens.cpu().tolist()
        self._cache_position[layer_idx] += kv_seqlens

        if len(self.states) <= layer_idx:
            key_state, value_state = list(map(lambda x: torch.split(x, kv_seqlens_cpu), attn_state))
            state = dict(
                attn_state=(key_state, value_state),
                cu_seqlens=cu_seqlens,
                max_seqlen=kv_seqlens.max().item()
            )
            self.states.append(state)

        # Case 2: append current step's kv cache
        else:
            state = self.states[layer_idx]
            if state["attn_state"] is not None:
                key_state, value_state = list(map(lambda x: torch.split(x, kv_seqlens_cpu), attn_state))
                key_cache, value_cache = state['attn_state']
                old_cu_seqlens = state['cu_seqlens']

                key_cache = tuple(map(lambda x, y: torch.cat([x, y], dim=0), key_cache, key_state))
                value_cache = tuple(map(lambda x, y: torch.cat([x, y], dim=0), value_cache, value_state))
                
                new_cu_seqlens = old_cu_seqlens + cu_seqlens
                state.update(
                    attn_state=(key_cache, value_cache),
                    cu_seqlens=new_cu_seqlens,
                    max_seqlen=(new_cu_seqlens[1:] - new_cu_seqlens[:-1]).max().item()
                )
        return state
    
    def get_kv_cache(self, state: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        return tuple(map(lambda x: torch.cat(x, 0), state['attn_state']))
    
    def get_seq_length(self, layer_idx: Optional[int] = 0) -> torch.Tensor:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.states) <= layer_idx:
            return torch.zeros(self.batch_size, dtype=torch.int64, device=self.device)
        return self._cache_position[layer_idx]
    
    def get_cu_seq_length(self, layer_idx: Optional[int] = 0) -> torch.Tensor:
        """Returns the accumulated sequence length of the cached states. A layer index can be optionally passed."""
        if len(self.states) <= layer_idx:
            return torch.zeros(self.batch_size + 1, dtype=torch.int64, device=self.device)
        return self.states[layer_idx]['cu_seqlens']

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states. Cache does not have a maximum length."""
        return None

    def to_legacy_cache(self) -> Tuple:
        return tuple(self.states)
    
    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        raise NotImplementedError("Varlen Batch Inference does not support beam search at now.")
    
    @classmethod
    @torch.compiler.disable
    def from_legacy_cache(
        cls,
        past_key_values: Optional[Tuple] = None,
        cache_position: int = 0,
        batch_size: int = 1,
        device: str | torch.device = None
    ):
        """Converts a cache in the legacy cache format into an equivalent `Cache`."""

        cache = cls(cache_position, batch_size=batch_size, device=device)
        if isinstance(past_key_values, list):
            for layer_idx in range(len(past_key_values)):
                cache.states.append(past_key_values[layer_idx])
        return cache

@torch.no_grad()
def linear_init(
    linear: nn.Linear,
    distribution: Optional[str]='normal',
    zero_bias: Optional[bool]=False,
    gain: Optional[float]=1.0
) ->None:
    if distribution == 'normal':
        nn.init.xavier_normal_(linear.weight, gain=gain)
    elif distribution == 'uniform':
        nn.init.xavier_uniform_(linear.weight, gain=gain)
    if linear.bias is not None:
        if zero_bias:
            nn.init.zeros_(linear.bias)
        else:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(linear.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(linear.bias, -bound, bound)


@torch.no_grad()
def embedding_init(embedding: nn.Embedding) ->None:
    fan_out = embedding.weight.size(1)
    std = 1.0 * math.sqrt(1.0 / float(fan_out))
    nn.init.normal_(embedding.weight, 0., std)
    if embedding.padding_idx is not None:
        embedding.weight[embedding.padding_idx].fill_(0)

def sparse_to_dense(src: torch.Tensor, length: torch.Tensor) ->torch.Tensor:
    maxLength = length.max().item()

    length = length.cpu().numpy()
    broadcastIdx = np.arange(length[0], dtype=np.int64)
    for i in range(1, length.shape[0]): broadcastIdx = np.concatenate([broadcastIdx, np.arange(length[i], dtype=np.int64) + maxLength * i], axis=0)
    broadcastIdx = torch.tensor(broadcastIdx, dtype=torch.int64, device=src.device)

    tgt = torch.zeros((length.shape[0] * maxLength, src.size(-1)), dtype=src.dtype, device=src.device)
    tgt[broadcastIdx] = src
    tgt = tgt.reshape(length.shape[0], maxLength, -1).contiguous()
    return tgt

#########################################################
#                   --- model ---
#########################################################
class FullAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rotary_base: int,
        dropout: float,
        layer_idx: int,
        **kwargs
    ):
        super(FullAttention, self).__init__()

        self.hidden_size = hidden_size
        self.num_q_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = hidden_size // num_heads
        self.dropout = dropout
        self.layer_idx = layer_idx

        self.qkv = nn.Linear(hidden_size, hidden_size + 2 * num_kv_heads * self.head_size, bias=False)
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.rotary = RotaryEmbedding(dim=self.head_size, base=rotary_base)

        self._init_weights()
    
    def _init_weights(self):
        for k, v in self.named_modules():
            if isinstance(v, nn.Linear): linear_init(v, zero_bias=True)
    
    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor=None, max_seqlen: int=None, causal: bool=True, past_key_values: Cache | VarlenCache=None):
        """
        Training with varlen:

        x -> size(B*L, D)
        cu_seqlens -> size(B+1)

        Generating with padding:
        
        x -> size(B, L, D)
        cu_seqlens -> None
        """

        if cu_seqlens is None:
            qkv: torch.Tensor = self.qkv(x)
            qkv = rearrange(qkv, "B L (H D) -> B L H D", H=(self.num_q_heads + 2 * self.num_kv_heads), D=self.head_size)
            q, kv = torch.split(qkv, [self.num_q_heads, 2 * self.num_kv_heads], dim=-2)
            kv = rearrange(kv, "B L (C H) D -> B L C H D", C=2, H=self.num_kv_heads)

            if past_key_values is not None:
                seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
                _max_seqlen = q.size(1) + seqlen_offset
                q, kv = self.rotary(q, kv, seqlen_offset=seqlen_offset, max_seqlen=_max_seqlen, num_heads_q=self.num_q_heads)

                k, v = kv.unbind(dim=2)
                k, v = past_key_values.update(
                    attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                    layer_idx=self.layer_idx,
                    offset=q.size(1),
                    cache_kwargs=dict()
                )["attn_state"]
                k, v = rearrange(k, "... (H D) -> ... H D", H=self.num_kv_heads, D=self.head_size), rearrange(v, "... (H D) -> ... H D", H=self.num_kv_heads, D=self.head_size)
            
                kv = torch.cat([k.unsqueeze(2), v.unsqueeze(2)], dim=2)
            else:
                q, kv = self.rotary(q, kv)

            out = flash_attn_kvpacked_func(q, kv, dropout_p=self.dropout if self.training else 0, causal=causal)
            out = self.out(rearrange(out, "B L H D -> B L (H D)"))
        else:
            qkv: torch.Tensor = self.qkv(x)
            qkv = rearrange(qkv, "L (H D) -> L H D", H=(self.num_q_heads + 2 * self.num_kv_heads), D=self.head_size)
            q, k, v = torch.split(qkv, [self.num_q_heads, self.num_kv_heads, self.num_kv_heads], dim=-2)

            if past_key_values is not None:
                assert isinstance(past_key_values, VarlenCache)

                seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
                _seqlen = cu_seqlens[1:] - cu_seqlens[:-1]
                _max_seqlen = (seqlen_offset + _seqlen).max().item()

                self.rotary._update_cos_sin_cache(seqlen=_max_seqlen, device=q.device, dtype=q.dtype)
                q, k = apply_rotary_emb(q, self.rotary._cos_cached, self.rotary._sin_cached, seqlen_offsets=seqlen_offset, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen),\
                    apply_rotary_emb(k, self.rotary._cos_cached, self.rotary._sin_cached, seqlen_offsets=seqlen_offset, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)
                
                new_cache = past_key_values.update(
                    attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                    cu_seqlens=cu_seqlens,
                    layer_idx=self.layer_idx,
                    cache_kwargs=dict()
                )
                k, v = past_key_values.get_kv_cache(new_cache)
                k, v = rearrange(k, "... (H D) -> ... H D", H=self.num_kv_heads, D=self.head_size), rearrange(v, "... (H D) -> ... H D", H=self.num_kv_heads, D=self.head_size)

                kv_cu_seqlens, kv_max_seqlen = new_cache['cu_seqlens'], new_cache['max_seqlen']

                out = flash_attn_varlen_func(q, k, v, cu_seqlens, kv_cu_seqlens, max_seqlen, kv_max_seqlen, dropout_p=self.dropout if self.training else 0, causal=causal)
            else:
                self.rotary._update_cos_sin_cache(seqlen=max_seqlen, device=q.device, dtype=q.dtype)
                q, k = apply_rotary_emb(q, self.rotary._cos_cached, self.rotary._sin_cached, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen),\
                    apply_rotary_emb(k, self.rotary._cos_cached, self.rotary._sin_cached, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen)

                out = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, dropout_p=self.dropout if self.training else 0, causal=causal)
            out = self.out(rearrange(out, "L H D -> L (H D)"))

        return out, None, past_key_values

class TransformerBlock(nn.Module):
    def __init__(self, config: Sophie0Config, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        self.attn_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.eps)
        self.attn = FullAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            rotary_base=config.rope_base,
            dropout=config.dropout,
            layer_idx=self.layer_idx
        )
        self.ffn_norm = RMSNorm(hidden_size=config.hidden_size, eps=config.eps)
        self.ffn = GatedMlp(
            in_features=config.hidden_size,
            hidden_features=config.intermediate_size,
            activation=F.silu,
            bias1=False,
            bias2=False,
            multiple_of=1
        )

        self._init_weights()
    
    def _init_weights(self):
        for k, v in self.ffn.named_modules():
            if isinstance(v, nn.Linear): linear_init(v, zero_bias=True)
    
    def forward(self, x: torch.Tensor, cu_seqlens: torch.Tensor=None, max_seqlen: int=None, causal: bool=True, past_key_values: Cache=None):
        out, _, past_key_values = self.attn(self.attn_norm(x), cu_seqlens, max_seqlen, causal, past_key_values)
        x = x + out
        x = x + self.ffn(self.ffn_norm(x))

        return (x, _, past_key_values)


class Sophie0PretraindModel(PreTrainedModel):
    config_class = Sophie0Config
    supports_gradient_checkpointing = True
    _supports_cache_class = True
    _no_split_modules = ["TransformerBlock"]

    def __init__(self, *inputs, **kwargs):
        super().__init__(*inputs, **kwargs)
    
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Embedding):
            embedding_init(module)
        elif isinstance(module, nn.Linear):
            linear_init(module, zero_bias=True)

class Sophie0Model(Sophie0PretraindModel):
    def __init__(self, config: Sophie0Config, **kwargs):
        super().__init__(config, **kwargs)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([TransformerBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.eps)

        self.post_init()
    
    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        max_seqlen: Optional[int] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Union[Cache, VarlenCache, List[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        **kwargs: Unpack[Dict]
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_hidden_states = output_hidden_states if output_hidden_states is not None else getattr(self.config, "output_hidden_states", False)
        use_cache = use_cache if use_cache is not None else (self.config.use_cache if not self.training else False)
        return_dict = return_dict if return_dict is not None else getattr(self.config, "use_return_dict", False)

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
        hidden_states = inputs_embeds

        if cu_seqlens is not None:
            if use_cache and not isinstance(past_key_values, VarlenCache): past_key_values = VarlenCache.from_legacy_cache(past_key_values, batch_size=cu_seqlens.size(0)-1, device=cu_seqlens.device)
        else:
            if use_cache and not isinstance(past_key_values, Cache): past_key_values = Cache.from_legacy_cache(past_key_values)

        if kwargs.get("use_gradient_checkpoint", False) is True and self.supports_gradient_checkpointing and self.training: self.gradient_checkpointing_enable()
        else: self.gradient_checkpointing = False

        all_hidden_states = () if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states: all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                hidden_states, _, past_key_values = checkpoint.checkpoint(
                    layer.__call__,
                    hidden_states,
                    cu_seqlens,
                    max_seqlen,
                    True,
                    past_key_values
                )
            else:
                hidden_states, _, past_key_values = layer(hidden_states, cu_seqlens, max_seqlen, True, past_key_values)
        
        hidden_states = self.norm(hidden_states)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, past_key_values] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=None
        )

class Sophie0ForCausalLM(Sophie0PretraindModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Sophie0Config):
        super().__init__(config)

        self.model = Sophie0Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None

        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def generate(self, *args, **kwargs):
        try:
            return super().generate(*args, **kwargs)
        except AttributeError as exception:
            if 'past_key_values' in str(exception):
                raise AttributeError(
                    f"You tried to call `generate` with a decoding strategy that manipulates `past_key_values`, "
                    f"which is not supported for {self.__class__.__name__}. "
                    f"Try another generation strategy instead. "
                    f"For the available generation strategies, check this doc: "
                    f"https://huggingface.co/docs/transformers/en/generation_strategies#decoding-strategies"
                )
            else:
                raise exception
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[int] = None,
        use_cache: Optional[bool] = True,
        logits_to_keep = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        max_seqlen: Optional[int] = None,
        use_varlen_inference: Optional[bool]=False,
        **kwargs
    ):
        if inputs_embeds is not None and len(past_key_values) == 0:
            model_inputs = {'inputs_embeds': inputs_embeds}
        else:
            if past_key_values is not None and len(past_key_values) > 0:
                input_ids = input_ids[:, -1:]
                if isinstance(past_key_values, VarlenCache):
                    input_ids = input_ids.squeeze(-1)
                    cu_seqlens = torch.arange(past_key_values.batch_size + 1, dtype=torch.int32, device=input_ids.device)
                    max_seqlen = 1
            else:
                if use_varlen_inference:
                    input_ids, _, cu_seqlens, max_seqlen, _ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
                    input_ids = input_ids.squeeze(-1)
            
            model_inputs = {'input_ids': input_ids.contiguous()}
        
        if logits_to_keep is not None:
            model_inputs['logits_to_keep'] = logits_to_keep
        
        model_inputs.update({
            'past_key_values': past_key_values,
            'use_cache': use_cache,
            'cu_seqlens': cu_seqlens,
            'max_seqlen': max_seqlen
        })
        return model_inputs
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        cu_seqlens: Optional[torch.LongTensor] = None,
        max_seqlen: Optional[int] = None,
        use_varlen_inference: Optional[bool]=False,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Union[Cache, VarlenCache, List[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        labels_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        **kwargs: Unpack[Dict]
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else getattr(self.config, "output_attentions", False)
        output_hidden_states = output_hidden_states if output_hidden_states is not None else getattr(self.config, "output_hidden_states", False)
        return_dict = return_dict if return_dict is not None else getattr(self.config, "use_return_dict", False)

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs
        )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)
        past_key_values = outputs.past_key_values

        loss = None
        if labels is not None:
            self.criterion = CrossEntropyLoss(ignore_index=self.config.pad_token_id, reduction="mean" if labels_mask is None else "none")
            if logits.dim() == 2: # varlen
                assert labels.dim() == 1
                loss = self.criterion(logits, labels)
                if labels_mask is not None:
                    loss = loss * labels_mask
                    loss = loss.sum() / labels_mask.sum()
                else:
                    loss = loss.mean()
            else:
                assert labels.dim() == 2
                if self.config.right_shift:
                    labels = labels[:, 1:]
                    logits = logits[:, :-1].contiguous()
                    
                loss = self.criterion(logits.flatten(0, 1), labels.flatten(0, 1))
                if labels_mask is not None:
                    loss = loss * labels_mask.flatten(0, 1)
                    loss = loss.sum() / labels_mask.sum()
                else:
                    loss = loss.mean()
        
        else:
            if isinstance(past_key_values, VarlenCache):
                kv_cu_seqlens = past_key_values.get_cu_seq_length()
                if logits.size(0) > past_key_values.batch_size: logits = logits.index_select(0, kv_cu_seqlens[1:] - 1)
                logits = logits.unsqueeze(1)
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )