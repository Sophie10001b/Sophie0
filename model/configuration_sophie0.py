from typing import Dict, List, Tuple, Union, Optional, Any
from transformers.configuration_utils import PretrainedConfig

class Sophie0Config(PretrainedConfig):

    model_type = 'transformer'
    keys_to_ignore_at_inference = ['past_key_values']

    def __init__(
        self,
        hidden_size: int = 1024,
        num_hidden_layers: int = 28,
        num_heads: int = 16,
        num_kv_heads: int = 8,
        window_size: Optional[int] = None,
        rope_base: Optional[int] = int(1e6),
        intermediate_size: Optional[int] = 4096,
        hidden_act: str = "swish",
        eps: float = 1e-5,
        use_cache: bool = True,
        pad_token_id: int = 3,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        prompt_token_id: int = 8,
        user_token_id: int = 9,
        bot_token_id: int = 10,
        tie_word_embeddings: bool = True,
        vocab_size: int = 65536,
        dropout: float = 0.0,
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.window_size = window_size
        self.rope_base = rope_base

        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

        self.eps = eps
        self.use_cache = use_cache

        self.vocab_size = vocab_size
        self.dropout = dropout

        self.prompt_token_id = prompt_token_id
        self.user_token_id = user_token_id
        self.bot_token_id = bot_token_id

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )