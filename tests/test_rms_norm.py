import pytest
import torch

from qinferlab.layers import PreNormFeedForwardBlock
from qinferlab.layers import RMSNorm

def test_rms_norm_output_shape()->None:
    norm=RMSNorm(hidden_size=128)
    input_tensor=torch.randn(2,16,128)
    output=norm(input_tensor)
    assert output.shape==input_tensor.shape

def test_rms_norm_matches_explicit_formula()->None:
    torch.manual_seed(0)
    eps=1e-6
    norm=RMSNorm(
        hidden_size=64,
        eps=eps,
    )
    input_tensor=torch.randn(2,8,64)
    model_output=norm(input_tensor)
    mean_square=input_tensor.pow(2).mean(dim=-1,keepdim=True)
    expected_output=input_tensor*torch.rsqrt(mean_square+eps)*norm.weight

    torch.testing.assert_close(model_output,expected_output)

def test_rms_norm_produces_unit_mean_square()->None:
    torch.manual_seed(0)
    norm=RMSNorm(
        hidden_size=128,
        eps=1e-6,
    )
    input_tensor=torch.randn(2,16,128)
    output=norm(input_tensor)
    output_mean_square=output.pow(2).mean(dim=-1)
    torch.testing.assert_close(output_mean_square,torch.ones_like(output_mean_square),rtol=1e-4,atol=1e-4)

def test_block_output_shape()->None:
    block=PreNormFeedForwardBlock(
        hidden_size=128,
        intermediate_size=512,
    )
    input_tensor=torch.randn(2,16,128)
    output=block(input_tensor)
    assert output.shape==input_tensor.shape

def test_block_parameter_count()->None:
    hidden_size=128
    intermediate_size=512
    block=PreNormFeedForwardBlock(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    expected_parameter_count=hidden_size+3*hidden_size*intermediate_size
    assert block.parameter_count()==expected_parameter_count

def test_residual_path_when_ffn_is_zero()->None:
    torch.manual_seed(0)
    block=PreNormFeedForwardBlock(
        hidden_size=64,
        intermediate_size=256,
    )

    for parameter in block.feed_forward.parameters():
        torch.nn.init.zeros_(parameter)

    input_tensor=torch.randn(2,8,64)
    output=block(input_tensor)
    torch.testing.assert_close(output,input_tensor)

@pytest.mark.parametrize(
    "hidden_size,eps",
    [
        (0,1e-6),
        (-1,1e-6),
        (128,0.0),
        (128,-1e-6),
    ],
)

def test_rms_norm_invaild_arguments(
    hidden_size:int,
    eps:float,
)->None:
    with pytest.raises(ValueError):
        RMSNorm(hidden_size=hidden_size,eps=eps)
