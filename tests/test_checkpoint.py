from __future__ import annotations

from pathlib import Path

import pytest
import torch

from flywire_llm.checkpoint import load_checkpoint, save_checkpoint
from flywire_llm.config import BlankLLMConfig
from flywire_llm.model import BlankCausalLM, BlankLLMError
from flywire_llm.training import causal_batch_from_text


def test_checkpoint_round_trip_preserves_logits_and_provenance(tmp_path: Path):
    torch.manual_seed(23)
    model = BlankCausalLM(BlankLLMConfig.smoke()).eval()
    sample, _ = causal_batch_from_text("ทดสอบ checkpoint", max_seq_len=64)
    with torch.no_grad():
        before = model(sample).logits

    path = tmp_path / "blank.pt"
    save_checkpoint(path, model, training_step=0)
    loaded, metadata = load_checkpoint(path)
    loaded.eval()
    with torch.no_grad():
        after = loaded(sample).logits

    torch.testing.assert_close(before, after, rtol=0.0, atol=0.0)
    assert metadata["pretrained"] is False
    assert metadata["external_pretrained_weights"] is False
    assert metadata["initialization_origin"] == "random"
    assert metadata["random_initialization_origin"] is True
    assert metadata["knowledge_in_weights_claimed"] is False
    assert metadata["untrained"] is True
    assert metadata["training_step"] == 0


def test_smoke_trained_checkpoint_keeps_random_origin_but_is_not_untrained(tmp_path: Path):
    model = BlankCausalLM(BlankLLMConfig.smoke())
    path = tmp_path / "step1.pt"
    save_checkpoint(path, model, training_step=1)
    _, metadata = load_checkpoint(path)
    assert metadata["pretrained"] is False
    assert metadata["random_initialization_origin"] is True
    assert metadata["untrained"] is False
    assert metadata["training_step"] == 1


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("pretrained", True, "pretrained=false"),
        ("external_pretrained_weights", True, "pretrained weights"),
        ("initialization_origin", "downloaded", "origin must be random"),
        ("knowledge_in_weights_claimed", True, "cannot claim"),
    ],
)
def test_loader_rejects_nonblank_provenance(
    tmp_path: Path,
    field: str,
    value,
    message: str,
):
    model = BlankCausalLM(BlankLLMConfig.smoke())
    path = tmp_path / "bad.pt"
    save_checkpoint(path, model)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    payload["metadata"][field] = value
    torch.save(payload, path)

    with pytest.raises(BlankLLMError, match=message):
        load_checkpoint(path)
