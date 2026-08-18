from torch import Tensor
from torch import nn

class SwiGLUFeedForward(nn.Module):
    """A SwiGLU feed-forward block used in decoder-only language model."""

    def __init__(
        self,
        hidden_size:int,
        intermediate_size:int,
        bias:bool=False,
    )->None:
        super().__init__()

        if hidden_size<=0:
            raise ValueError("hidden_size must be positive")

        if intermediate_size<=0:
            raise ValueError("intermediate_size must be positive")

        self.hidden_size=hidden_size
        self.intermediate_size=intermediate_size

        self.gate_proj=nn.Linear(
            hidden_size,
            intermediate_size,
            bias=bias,
        )
        self.up_proj=nn.Linear(
            hidden_size,
            intermediate_size,
            bias=bias,
        )
        self.down_proj=nn.Linear(
            intermediate_size,
            hidden_size,
            bias=bias,
        )

        self.activation=nn.SiLU()

    def forward(self,hidden_states:Tensor)->Tensor:
        gate=self.activation(self.gate_proj(hidden_states))
        up=self.up_proj(hidden_states)
        activated_states=gate*up
        output=self.down_proj(activated_states)

        return output

    def parameter_count(self)->int:
        return sum(parameter.numel() for parameter in self.parameters())