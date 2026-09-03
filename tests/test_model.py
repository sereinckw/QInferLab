import pytest
import torch

from qinferlab.model import DecoderOnlyTransformer
from qinferlab.model import ModelConfig


def create_test_config(
    tie_word_embeddings: bool = True,
) -> ModelConfig:
    return ModelConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_heads=4,
        max_sequence_length=32,
        tie_word_embeddings=tie_word_embeddings,
    )


def test_model_output_shape() -> None:
    config = create_test_config()
    model = DecoderOnlyTransformer(config)

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (2, 16),
    )

    logits = model(input_ids)

    assert logits.shape == (
        2,
        16,
        config.vocab_size,
    )


def test_model_has_expected_number_of_layers() -> None:
    config = create_test_config()
    model = DecoderOnlyTransformer(config)

    assert len(model.layers) == config.num_layers


def test_word_embeddings_are_tied() -> None:
    config = create_test_config(
        tie_word_embeddings=True
    )
    model = DecoderOnlyTransformer(config)

    assert (
        model.token_embedding.weight
        is model.lm_head.weight
    )

    assert (
        model.token_embedding.weight.data_ptr()
        == model.lm_head.weight.data_ptr()
    )


def test_word_embeddings_can_be_untied() -> None:
    config = create_test_config(
        tie_word_embeddings=False
    )
    model = DecoderOnlyTransformer(config)

    assert (
        model.token_embedding.weight
        is not model.lm_head.weight
    )


def test_parameter_count_with_tied_embeddings() -> None:
    config = create_test_config(
        tie_word_embeddings=True
    )
    model = DecoderOnlyTransformer(config)

    embedding_parameters = (
        config.vocab_size * config.hidden_size
    )

    block_parameters = (
        4 * config.hidden_size**2
        + 3
        * config.hidden_size
        * config.intermediate_size
        + 2 * config.hidden_size
    )

    expected_parameter_count = (
        embedding_parameters
        + config.num_layers * block_parameters
        + config.hidden_size
    )

    assert (
        model.parameter_count()
        == expected_parameter_count
    )


def test_model_remains_causal() -> None:
    torch.manual_seed(0)

    config = create_test_config()
    model = DecoderOnlyTransformer(config)
    model.eval()

    original_ids = torch.randint(
        0,
        config.vocab_size,
        (1, 8),
    )
    modified_ids = original_ids.clone()

    modified_ids[:, 4:] = (
        modified_ids[:, 4:] + 1
    ) % config.vocab_size

    original_logits = model(original_ids)
    modified_logits = model(modified_ids)

    torch.testing.assert_close(
        original_logits[:, :4, :],
        modified_logits[:, :4, :],
        atol=1e-5,
        rtol=1e-5,
    )

    assert not torch.allclose(
        original_logits[:, 4:, :],
        modified_logits[:, 4:, :],
    )


def test_logits_are_finite() -> None:
    config = create_test_config()
    model = DecoderOnlyTransformer(config)

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (2, 16),
    )

    logits = model(input_ids)

    assert torch.isfinite(logits).all()


def test_explicit_position_ids() -> None:
    config = create_test_config()
    model = DecoderOnlyTransformer(config)

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (1, 4),
    )
    position_ids = torch.tensor([[5, 6, 7, 8]])

    logits = model(
        input_ids=input_ids,
        position_ids=position_ids,
    )

    assert logits.shape == (
        1,
        4,
        config.vocab_size,
    )


@pytest.mark.parametrize(
    "config_changes",
    [
        {"vocab_size": 0},
        {"hidden_size": 0},
        {"intermediate_size": 0},
        {"num_layers": 0},
        {"num_heads": 0},
        {"hidden_size": 30, "num_heads": 8},
        {"max_sequence_length": 0},
        {"rms_norm_eps": 0.0},
        {"rope_base": 0.0},
    ],
)
def test_invalid_configurations(
    config_changes: dict,
) -> None:
    default_values = {
        "vocab_size": 128,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_layers": 2,
        "num_heads": 4,
        "max_sequence_length": 32,
    }
    default_values.update(config_changes)

    with pytest.raises(ValueError):
        ModelConfig(**default_values)


def test_invalid_input_shape() -> None:
    model = DecoderOnlyTransformer(
        create_test_config()
    )

    invalid_input = torch.randint(0, 128, (16,))

    with pytest.raises(ValueError):
        model(invalid_input)


def test_invalid_input_dtype() -> None:
    model = DecoderOnlyTransformer(
        create_test_config()
    )

    invalid_input = torch.randn(2, 16)

    with pytest.raises(ValueError):
        model(invalid_input)


def test_sequence_too_long() -> None:
    config = create_test_config()
    model = DecoderOnlyTransformer(config)

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (1, config.max_sequence_length + 1),
    )

    with pytest.raises(ValueError):
        model(input_ids)