import pytest
import torch
from qinferlab.layers import SwiGLUFeedForward

def test_output_shape()->None:
    model=SwiGLUFeedForward(
        hidden_size=128,
        intermediate_size=512,
    )

    input_tensor=torch.randn(2,16,128)
    output=model(input_tensor)

    assert output.shape==input_tensor.shape

def test_parameter_count()->None:
    hidden_size=128
    intermediate_size=512

    model=SwiGLUFeedForward(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )

    expected_parameter_count=3*hidden_size*intermediate_size

    assert model.parameter_count()==expected_parameter_count

def test_forward_matches_explicit_formula()->None:
    torch.manual_seed(0)

    model=SwiGLUFeedForward(
        hidden_size=64,
        intermediate_size=256,

    )

    input_tensor=torch.randn(2,8,64)
    model_output=model(input_tensor)

    gate=torch.nn.functional.silu(model.gate_proj(input_tensor))
    up=model.up_proj(input_tensor)
    expected_output=model.down_proj(gate*up)

    torch.testing.assert_close(model_output, expected_output)

@pytest.mark.parametrize(
    "hidden_size, intermediate_size",
    [
        (0, 512),
        (128, 0),
        (-1, 512),
        (128, -1),
    ],
)

def test_invaild_dimensions(
    hidden_size:int,
    intermediate_size:int,
)->None:
    with pytest.raises(ValueError):
        SwiGLUFeedForward(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )