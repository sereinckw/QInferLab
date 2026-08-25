import pytest
import torch
import torch.nn.functional as functional
from qinferlab.layers import CausalSelfAttention


def test_attention_output_shape()->None:
    attention=CausalSelfAttention(
        hidden_size=128,
        num_heads=4,
    )

    hidden_states=torch.randn(2,16,128)
    output=attention(hidden_states)

    assert output.shape==hidden_states.shape

def test_head_dimensions()->None:
    attention=CausalSelfAttention(
        hidden_size=128,
        num_heads=4,
    )
    assert attention.num_heads==4
    assert attention.head_dim==32

def test_parameter_count()->None:
    hidden_size=128
    attention=CausalSelfAttention(
        hidden_size=hidden_size,
        num_heads=4,
    )
    expected_parameter_count=4*hidden_size**2

    assert attention.parameter_count()==expected_parameter_count

def test_attention_matches_pytorch_reference()->None:
    torch.manual_seed(0)
    attention=CausalSelfAttention(
        hidden_size=64,
        num_heads=4,
    )
    attention.eval()
    hidden_states=torch.randn(2,8,64)
    model_output=attention(hidden_states)

    query=attention._split_heads(
        attention.query_proj(hidden_states)
    )
    key=attention._split_heads(
        attention.key_proj(hidden_states)
    )
    value=attention._split_heads(
        attention.value_proj(hidden_states)
    )

    reference_context=functional.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=0.0,
        is_causal=True,
    )

    reference_context=attention._merge_heads(
        reference_context
    )
    reference_output=attention.output_proj(
        reference_context
    )

    torch.testing.assert_close(
        model_output,
        reference_output,
        atol=1e-5,
        rtol=1e-5,
    )

def test_future_tokens_do_not_affect_past_outputs()->None:
    torch.manual_seed(0)

    attention=CausalSelfAttention(
        hidden_size=64,
        num_heads=4,
    )
    attention.eval()

    original_input=torch.randn(1,8,64)
    modified_input=original_input.clone()

    modified_input[:,4:,:]=(
        modified_input[:,4:,:]+10.0
    )

    original_output=attention(original_input)
    modified_output=attention(modified_input)

    torch.testing.assert_close(
        original_output[:,:4,:],
        modified_output[:,:4,:],
        atol=1e-5,
        rtol=1e-5,
    )

    assert not torch.allclose(
        original_output[:,4:,:],
        modified_output[:,4:,:],
    )

@pytest.mark.parametrize(
    "hidden_size,num_heads",
    [
        (0,4),
        (128,0),
        (128,-1),
        (130,4),
    ],
)

def test_invalid_attention_configuration(
    hidden_size:int,
    num_heads:int,
)->None:
    with pytest.raises(ValueError):
        CausalSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
        )

def test_invalid_input_shape()->None:
    attention=CausalSelfAttention(
        hidden_size=128,
        num_heads=4,
    )

    invalid_input=torch.randn(2,128)

    with pytest.raises(ValueError):
        attention(invalid_input)

