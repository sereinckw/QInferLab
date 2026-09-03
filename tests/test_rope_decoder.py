import pytest
import torch

from qinferlab.layers import RotaryPositionEmbedding
from qinferlab.layers import TransformerDecoderBlock


def test_rope_preserves_shape_and_norm() -> None:
    torch.manual_seed(0)

    rope = RotaryPositionEmbedding(head_dim=32)

    query = torch.randn(2, 4, 8, 32)
    key = torch.randn(2, 4, 8, 32)

    rotated_query, rotated_key = rope(query, key)

    assert rotated_query.shape == query.shape
    assert rotated_key.shape == key.shape

    torch.testing.assert_close(
        torch.linalg.vector_norm(
            rotated_query,
            dim=-1,
        ),
        torch.linalg.vector_norm(
            query,
            dim=-1,
        ),
        atol=1e-5,
        rtol=1e-5,
    )

    torch.testing.assert_close(
        torch.linalg.vector_norm(
            rotated_key,
            dim=-1,
        ),
        torch.linalg.vector_norm(
            key,
            dim=-1,
        ),
        atol=1e-5,
        rtol=1e-5,
    )


def test_rope_does_not_rotate_position_zero() -> None:
    torch.manual_seed(0)

    rope = RotaryPositionEmbedding(head_dim=32)

    query = torch.randn(1, 4, 8, 32)
    key = torch.randn(1, 4, 8, 32)

    rotated_query, rotated_key = rope(query, key)

    torch.testing.assert_close(
        rotated_query[:, :, 0, :],
        query[:, :, 0, :],
    )
    torch.testing.assert_close(
        rotated_key[:, :, 0, :],
        key[:, :, 0, :],
    )


def test_different_positions_produce_different_rotations() -> None:
    rope = RotaryPositionEmbedding(head_dim=32)

    repeated_vector = torch.ones(1, 4, 8, 32)

    rotated_query, _ = rope(
        repeated_vector,
        repeated_vector,
    )

    assert not torch.allclose(
        rotated_query[:, :, 0, :],
        rotated_query[:, :, 1, :],
    )


def test_explicit_position_ids() -> None:
    torch.manual_seed(0)

    rope = RotaryPositionEmbedding(head_dim=32)

    query = torch.randn(1, 4, 3, 32)
    key = torch.randn(1, 4, 3, 32)

    position_ids = torch.tensor([[5, 6, 7]])

    explicit_query, explicit_key = rope(
        query,
        key,
        position_ids=position_ids,
    )

    assert explicit_query.shape == query.shape
    assert explicit_key.shape == key.shape

    assert not torch.allclose(
        explicit_query,
        query,
    )


def test_decoder_block_output_shape() -> None:
    block = TransformerDecoderBlock(
        hidden_size=128,
        intermediate_size=512,
        num_heads=4,
    )

    hidden_states = torch.randn(2, 16, 128)
    output = block(hidden_states)

    assert output.shape == hidden_states.shape


def test_decoder_block_parameter_count() -> None:
    hidden_size = 128
    intermediate_size = 512

    block = TransformerDecoderBlock(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_heads=4,
    )

    expected_parameter_count = (
        4 * hidden_size**2
        + 3 * hidden_size * intermediate_size
        + 2 * hidden_size
    )

    assert (
        block.parameter_count()
        == expected_parameter_count
    )


def test_decoder_residual_path_when_sublayers_are_zero() -> None:
    torch.manual_seed(0)

    block = TransformerDecoderBlock(
        hidden_size=64,
        intermediate_size=256,
        num_heads=4,
    )

    for parameter in block.self_attention.parameters():
        torch.nn.init.zeros_(parameter)

    for parameter in block.feed_forward.parameters():
        torch.nn.init.zeros_(parameter)

    hidden_states = torch.randn(2, 8, 64)
    output = block(hidden_states)

    torch.testing.assert_close(
        output,
        hidden_states,
    )


def test_decoder_remains_causal() -> None:
    torch.manual_seed(0)

    block = TransformerDecoderBlock(
        hidden_size=64,
        intermediate_size=256,
        num_heads=4,
    )
    block.eval()

    original_input = torch.randn(1, 8, 64)
    modified_input = original_input.clone()

    modified_input[:, 4:, :] += 10.0

    original_output = block(original_input)
    modified_output = block(modified_input)

    torch.testing.assert_close(
        original_output[:, :4, :],
        modified_output[:, :4, :],
        atol=1e-5,
        rtol=1e-5,
    )

    assert not torch.allclose(
        original_output[:, 4:, :],
        modified_output[:, 4:, :],
    )


@pytest.mark.parametrize(
    "head_dim,base",
    [
        (0, 10000.0),
        (-2, 10000.0),
        (31, 10000.0),
        (32, 0.0),
        (32, -1.0),
    ],
)
def test_invalid_rope_configuration(
    head_dim: int,
    base: float,
) -> None:
    with pytest.raises(ValueError):
        RotaryPositionEmbedding(
            head_dim=head_dim,
            base=base,
        )