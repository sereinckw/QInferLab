from dataclasses import dataclass

import torch
from torch import Tensor
from torch import nn

from qinferlab.layers import RMSNorm
from qinferlab.layers import TransformerDecoderBlock


@dataclass
class ModelConfig:
    """Configuration of the decoder-only Transformer."""

    vocab_size: int = 4096
    hidden_size: int = 256
    intermediate_size: int = 1024
    num_layers: int = 4
    num_heads: int = 8
    max_sequence_length: int = 512
    rms_norm_eps: float = 1e-6
    rope_base: float = 10000.0
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        if self.intermediate_size <= 0:
            raise ValueError(
                "intermediate_size must be positive"
            )

        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")

        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                "hidden_size must be divisible by num_heads"
            )

        if self.max_sequence_length <= 0:
            raise ValueError(
                "max_sequence_length must be positive"
            )

        if self.rms_norm_eps <= 0:
            raise ValueError(
                "rms_norm_eps must be positive"
            )

        if self.rope_base <= 0:
            raise ValueError("rope_base must be positive")


class DecoderOnlyTransformer(nn.Module):
    """A minimal decoder-only Transformer language model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.config = config

        self.token_embedding = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )

        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    num_heads=config.num_heads,
                    eps=config.rms_norm_eps,
                    rope_base=config.rope_base,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = RMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

        if config.tie_word_embeddings:
            self.lm_head.weight = (
                self.token_embedding.weight
            )

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor | None = None,
    ) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape "
                "(batch_size, sequence_length)"
            )

        if input_ids.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError(
                "input_ids must contain integer token IDs"
            )

        batch_size, sequence_length = input_ids.shape

        if sequence_length == 0:
            raise ValueError(
                "sequence_length must be greater than zero"
            )

        if sequence_length > self.config.max_sequence_length:
            raise ValueError(
                "sequence_length exceeds max_sequence_length"
            )

        if position_ids is None:
            position_ids = torch.arange(
                sequence_length,
                device=input_ids.device,
            )
            position_ids = position_ids.unsqueeze(0).expand(
                batch_size,
                -1,
            )
        elif position_ids.shape != input_ids.shape:
            raise ValueError(
                "position_ids must have the same shape "
                "as input_ids"
            )

        hidden_states = self.token_embedding(input_ids)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
            )

        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits

    def parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
        )