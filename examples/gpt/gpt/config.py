from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GPTConfig:
    model_type: str
    vocab_size: int
    block_size: int
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    attn_pdrop: float = 0.1
    n_layer: Optional[int] = None
    n_head: Optional[int] = None
    n_embd: Optional[int] = None

    def __post_init__(self):
        type_given = self.model_type is not None
        params_given = all((self.n_layer is not None, self.n_head is not None, self.n_embd is not None))
        assert type_given ^ params_given
        if type_given:
            # translate from model_type to detailed configuration
            values = {
                # names follow the huggingface naming conventions
                # GPT-1
                "openai-gpt": dict(n_layer=12, n_head=12, n_embd=768),  # 117M params
                # GPT-2 configs
                "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
                "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
                "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
                "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1475M params
                "gpt2-xxl": dict(n_layer=96, n_head=25, n_embd=1600),  # 2951M params
                "gpt2-xxxl": dict(n_layer=100, n_head=30, n_embd=1920),  # 4426M params
                "gpt2-4xl": dict(n_layer=190, n_head=30, n_embd=1920),  # 8409M params
            }[self.model_type]
            self.n_layer = values["n_layer"]
            self.n_head = values["n_head"]
            self.n_embd = values["n_embd"]


@dataclass
class TrainerConfig:
    block_size: int
    num_workers: int
    batch_size: int
    learning_rate: float
    betas: Tuple[int]
    weight_decay: float
    grad_norm_clip: float
    seed: int = 1
    max_iters: int = -1
