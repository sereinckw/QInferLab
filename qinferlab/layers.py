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

class RotaryPositionEmbedding(nn.Module):
    """Rotary position embedding for attention queries and keys."""

    def __init__(
        self,
        head_dim:int,
        base:float=1000.0,
    )->None:
        super().__init__()

        if  head_dim<=0:
            raise ValueError("head_dim must be positive")

        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even")

        if base <= 0:
            raise ValueError("base must be positive")

        self.head_dim=head_dim
        self.base=base

        dimension_indices=torch.arange(
            0,
            head_dim,
            2,
            dtype=torch.float32,
        )

        inverse_frequencies=1.0/(base**(dimension_indices/head_dim))

        self.register_buffer(
            "inverse_frequencies",
            inverse_frequencies,
            persistent=False,
        )

    def _apply_rotation(
        self,
        tensor:Tensor,
        cosine: Tensor,
        sine:Tensor,
    )->Tensor:
        even_values=tensor[...,0::2]
        odd_values=tensor[...,1::2]

        rotated_even=(
            even_values*cosine-odd_values*sine
        )

        rotated_odd=(
            even_values*sine+odd_values*cosine
        )

        rotated_pairs=torch.stack(
            (rotated_even,rotated_odd),
            dim=-1,
        )

        return rotated_pairs.flatten(-2)

    def forward(
        self,
        query:Tensor,
        key:Tensor,
        position_ids:Tensor | None=None,
    )->tuple[Tensor,Tensor]:
        if query.shape != key.shape:
            raise ValueError(
                "query and key must have identical shapes"
            )

        if query.ndim != 4:
            raise ValueError(
                "query and key must have shape "
                "(batch_size, num_heads, sequence_length, head_dim)"
            )

        if query.shape[-1] != self.head_dim:
            raise ValueError(
                "The last dimension must equal head_dim"
            )

        batch_size=query.shape[0]
        sequence_length=query.shape[2]

        if position_ids is None:
            position_ids=torch.arange(
                sequence_length,
                device=query.device,
            )
            position_ids=position_ids.unsqueeze(0).expand(
                batch_size,
                -1,
            )
        elif position_ids.ndim==1:
            if position_ids.shape[0]!=sequence_length:
                raise ValueError("One-dimensional position_ids must have length sequence_length"
                )

            position_ids=position_ids.unsqueeze(0).expand(
                batch_size,
                -1,
            )

        elif position_ids.shape!=(batch_size,sequence_length):
            raise ValueError(
                "position_ids must have shape "
                "(batch_size, sequence_length)"
            )

        position_ids=position_ids.to(query.device)

        angles=(
            position_ids.float().unsqueeze(-1)
            * self.inverse_frequencies.float().view(1,1,-1)
        )

        cosine=angles.cos().unsqueeze(1).to(query.dtype)
        sine=angles.sin().unsqueeze(1).to(query.dtype)

        rotated_query=self._apply_rotation(
            query,
            cosine,
            sine,
        )   
        rotated_key=self._apply_rotation(
            key,
            cosine,
            sine,
        )

        return rotated_query,rotated_key

class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask."""

    def __init__(
        self,
        hidden_size:int,
        num_heads:int,
        bias:bool=False,
        use_rope:bool=False,
        rope_base:float=1000.0,
    )->None:
        super().__init__()

        if hidden_size<=0:
            raise ValueError("hidden_size must be positive")

        if num_heads<=0:
            raise ValueError("num_heads must be positive")

        if hidden_size%num_heads!=0:
            raise ValueError(
                "hidden_size must be divisible by num_heads"
            )

        self.hidden_size=hidden_size
        self.num_heads=num_heads
        self.head_dim=hidden_size//num_heads
        self.scale = self.head_dim**-0.5
        
        self.rotary_embedding=(
            RotaryPositionEmbedding(
                head_dim=self.head_dim,
                base=rope_base,
            )
            if use_rope
            else None
        )

        self.query_proj=nn.Linear(
            hidden_size,
            hidden_size,
            bias=bias,
        )

        self.key_proj=nn.Linear(
            hidden_size,
            hidden_size,
            bias=bias,
        )

        self.value_proj=nn.Linear(
            hidden_size,
            hidden_size,
            bias=bias,
        )

        self.output_proj=nn.Linear(
            hidden_size,
            hidden_size,
            bias=bias,
        )

    def _split_heads(self,tensor:Tensor)->Tensor:
        batch_size,sequence_length,_=tensor.shape

        tensor=tensor.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        )

        return tensor.transpose(1,2)

    def _merge_heads(self,tensor:Tensor)->Tensor:
        batch_size,_,sequence_length,_=tensor.shape

        tensor=tensor.transpose(1,2).contiguous()

        return tensor.view(
            batch_size,
            sequence_length,
            self.hidden_size,
        )

    def forward(self,hidden_states:Tensor,position_ids:Tensor | None=None)->Tensor:
        if hidden_states.ndim!=3:
            raise ValueError(
                "hidden_states must have shape"
                "(batch_size, sequence_length, hidden_size)"
            )
        if hidden_states.shape[-1]!=self.hidden_size:
            raise ValueError(
                "The last input dimensioin must equal hidden_size"
            )

        query = self._split_heads(
            self.query_proj(hidden_states)
        )
        key=self._split_heads(
            self.key_proj(hidden_states)
        )
        value=self._split_heads(
            self.value_proj(hidden_states)
        )

        if self.rotary_embedding is not None:
            query,key=self.rotary_embedding(
                query=query,
                key=key,
                position_ids=position_ids,
            )

        attention_scores=torch.matmul(
            query,
            key.transpose(-2,-1),
        )
        attention_scores=attention_scores*self.scale

        sequence_length=hidden_states.shape[1]

        causal_mask=torch.ones(
            sequence_length,
            sequence_length,
            device=hidden_states.device,
            dtype=torch.bool,
        ).triu(diagonal=1)

        attention_scores=attention_scores.masked_fill(
            causal_mask,
            float("-inf"),
        )

        attention_weights=torch.softmax(
            attention_scores.float(),
            dim=-1,
        ).to(query.dtype)

        context=torch.matmul(
            attention_weights,
            value,
        )

        context=self._merge_heads(context)
        output=self.output_proj(context)

        return output

    def parameter_count(self)->int:
        return sum(
            parameter.numel() for parameter in self.parameters()
        )


class TransformerDecoderBlock(nn.Module):
    """A pre-norm Transformer decoder block."""

    def __init__(
        self,
        hidden_size:int,
        intermediate_size:int,
        num_heads:int,
        eps:float=1e-6,
        bias:bool=False,
        rope_base:float=1000.0,
    )->None:
        super().__init__()

        self.input_layer_norm=RMSNorm(
            hidden_size=hidden_size,
            eps=eps,
        )

        self.self_attention=CausalSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            bias=bias,
            use_rope=True,
            rope_base=rope_base,
        )

        self.post_attention_layer_norm=RMSNorm(
            hidden_size=hidden_size,
            eps=eps,
        )

        self.feed_forward=SwiGLUFeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            bias=bias,
        )

    def forward(
        self,
        hidden_states:Tensor,
        position_ids:Tensor | None=None,
    )->Tensor:
        attention_residual=hidden_states
        normalized_states=self.input_layer_norm(
            hidden_states
        )
        attention_output=self.self_attention(
            hidden_states=normalized_states,
            position_ids=position_ids,
        )

        hidden_states=(
            attention_residual+attention_output
        )

        feed_forward_residual=hidden_states

        normalized_states=(
            self.post_attention_layer_norm(hidden_states)
        )
        feed_forward_output=self.feed_forward(normalized_states)

        hidden_states=(
            feed_forward_residual+feed_forward_output
        )
        return hidden_states

    def parameter_count(self)->int:
        return sum(
            parameter.numel() for parameter in self.parameters()
        )





        
    