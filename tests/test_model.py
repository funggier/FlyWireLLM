from __future__ import annotations

import pytest
import torch

from flywire_llm.config import BlankLLMConfig, BlankLLMConfigError
from flywire_llm.model import BlankCausalLM
from flywire_llm.training import causal_batch_from_text, optimizer_step


def test_smoke_config_is_valid_gqa_profile():
    config = BlankLLMConfig.smoke()
    assert config.n_heads == 4
    assert config.n_kv_heads == 2
    assert config.n_heads > config.n_kv_heads


def test_bad_head_geometry_is_rejected():
    with pytest.raises(BlankLLMConfigError, match="divisible"):
        BlankLLMConfig(d_model=63).validate()


def test_forward_loss_shape_and_finiteness():
    torch.manual_seed(7)
    config = BlankLLMConfig.smoke()
    model = BlankCausalLM(config)
    inputs, targets = causal_batch_from_text(
        "ภาษาไทย and English",
        max_seq_len=64,
    )
    output = model(inputs, targets=targets)
    assert output.logits.shape == (
        1,
        inputs.shape[1],
        config.vocab_size,
    )
    assert output.loss is not None
    assert torch.isfinite(output.loss)


def test_causal_attention_prevents_future_token_leakage():
    torch.manual_seed(17)
    model = BlankCausalLM(BlankLLMConfig.smoke()).eval()
    a = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)
    b = a.clone()
    b[0, 4] = 99

    with torch.no_grad():
        logits_a = model(a).logits
        logits_b = model(b).logits

    torch.testing.assert_close(
        logits_a[:, :4, :],
        logits_b[:, :4, :],
        rtol=0.0,
        atol=0.0,
    )
    assert not torch.equal(logits_a[:, 4, :], logits_b[:, 4, :])


def test_optimizer_step_changes_random_initialized_weights():
    torch.manual_seed(11)
    model = BlankCausalLM(BlankLLMConfig.smoke())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs, targets = causal_batch_from_text(
        "FlyWireLLM โมเดลเปล่า ready for training",
        max_seq_len=96,
    )
    before = model.token_embedding.weight.detach().clone()
    loss = optimizer_step(model, optimizer, inputs, targets)
    after = model.token_embedding.weight.detach()

    assert loss > 0
    assert not torch.equal(before, after)


def test_tied_embeddings_share_storage():
    model = BlankCausalLM(BlankLLMConfig.smoke())
    assert (
        model.token_embedding.weight.data_ptr()
        == model.lm_head.weight.data_ptr()
    )


def test_random_seed_reproduces_initial_weights():
    torch.manual_seed(1234)
    a = BlankCausalLM(BlankLLMConfig.smoke())
    torch.manual_seed(1234)
    b = BlankCausalLM(BlankLLMConfig.smoke())
    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(pa, pb, rtol=0.0, atol=0.0)
