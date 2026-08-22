from torch import Tensor
from torch import nn
import torch

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

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalizaztion"""

    def __init__(
        self,
        hidden_size:int,
        eps:float=1e-6,
    )->None:
        super().__init__()
        if hidden_size<=0:
            raise ValueError("hidden_size must be positive")

        if eps<=0:
            raise ValueError("eps must be positive")

        self.hidden_size=hidden_size
        self.eps=eps

        self.weight=nn.Parameter(torch.ones(hidden_size))

    def forward(self,hidden_states:Tensor)->Tensor:
        input_dtype=hidden_states.dtype
        hidden_states_fp32=hidden_states.float()

        mean_square=hidden_states_fp32.pow(2).mean(dim=-1,keepdim=True)

        normalized_states=hidden_states_fp32*torch.rsqrt(mean_square+self.eps)
        normalized_states=normalized_states.to(input_dtype)
        
        return normalized_states*self.weight

class PreNormFeedForwardBlock(nn.Module):
    """A pre-norm feed-forward block with a residual connection"""

    def __init__(
        self,
        hidden_size:int,
        intermediate_size:int,
        eps:float=1e-6,
        bias:bool=False,
    )->None:
        super().__init__()
        self.input_layer_norm=RMSNorm(hidden_size=hidden_size,eps=eps)
        self.feed_forward=SwiGLUFeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=bias,
        )

    def forward(self,hidden_states:Tensor)->Tensor: 
        residual=hidden_states
        normalized_states=self.input_layer_norm(hidden_states)
        feed_forward_output=self.feed_forward(normalized_states)
        output=feed_forward_output+residual
        return output

    def parameter_count(self)->int:
        return sum(parameter.numel() for parameter in self.parameters())




        
    