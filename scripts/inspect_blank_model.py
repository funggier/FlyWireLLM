from __future__ import annotations

import json
from pathlib import Path

import torch

from flywire_llm.config import BlankLLMConfig
from flywire_llm.model import BlankCausalLM
from flywire_llm.tokenizer import UTF8ByteTokenizer


def main() -> None:
    config_path = Path("configs/smoke.json")
    config = BlankLLMConfig.from_dict(
        json.loads(config_path.read_text(encoding="utf-8"))
    )
    torch.manual_seed(1234)
    model = BlankCausalLM(config)
    tokenizer = UTF8ByteTokenizer()

    thai = "สวัสดี FlyWireLLM"
    english = "Hello FlyWireLLM"
    assert tokenizer.decode(tokenizer.encode(thai)) == thai
    assert tokenizer.decode(tokenizer.encode(english)) == english

    print("model_type=flywire_blank_causal_lm")
    print(f"parameters={model.parameter_count}")
    print(f"vocab_size={config.vocab_size}")
    print(f"layers={config.n_layers}")
    print(f"heads={config.n_heads}")
    print(f"kv_heads={config.n_kv_heads}")
    print("pretrained=false")
    print("initialization_origin=random")
    print("can_converse=false")
    print("thai_roundtrip=true")
    print("english_roundtrip=true")
    print("training_ready=true")


if __name__ == "__main__":
    main()
