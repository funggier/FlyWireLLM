from __future__ import annotations

import argparse
import ctypes
import json
import platform
import time
from pathlib import Path

import torch

from flywire_llm.model import BlankCausalLM
from flywire_llm.pretraining_execution import (
    load_pretraining_execution_contract,
    verify_base50m_tokenizer,
)
from flywire_llm.pretraining_training import (
    build_adamw,
    build_primary_schedule,
    gradient_clip_norm,
    learning_rate_for_update,
    load_training_configuration,
    pretraining_update,
)


ROOT = Path(__file__).resolve().parents[1]


def _windows_memory_bytes() -> dict[str, int] | None:
    if platform.system() != "Windows":
        return None

    class ProcessMemoryCountersEx(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCountersEx()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCountersEx),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    handle = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(
        handle,
        ctypes.byref(counters),
        counters.cb,
    )
    if not ok:
        return None
    return {
        "working_set_bytes": int(counters.WorkingSetSize),
        "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
        "private_bytes": int(counters.PrivateUsage),
        "peak_pagefile_bytes": int(counters.PeakPagefileUsage),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one ephemeral Base-50M optimizer qualification update. "
            "This is not a production pretraining checkpoint."
        )
    )
    parser.add_argument("--external-root", required=True)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--microbatches", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--precision",
        choices=("fp32", "bf16"),
        default="fp32",
    )
    parser.add_argument("--threads", type=int)
    parser.add_argument("--output-json")
    return parser.parse_args()


def _sample_ids(tokenizer, sequence_length: int) -> list[int]:
    if sequence_length < 8 or sequence_length > 1024:
        raise ValueError("sequence length must be in [8, 1024]")
    text = (
        "การทดสอบ pretraining runtime ต้องรักษา provenance และ resume "
        "อย่างถูกต้อง. Base-50M learns causal next-token structure from "
        "authorized train-only data. code_id=RUNTIME_004 value=12345 "
    )
    chunk = tokenizer.encode(text, add_eos=True)
    if len(chunk) < 2:
        raise RuntimeError("qualification text produced too few tokens")
    ids: list[int] = []
    while len(ids) < sequence_length + 1:
        ids.extend(chunk)
    return ids[: sequence_length + 1]


def main() -> None:
    args = parse_args()
    if args.microbatches < 1:
        raise ValueError("microbatches must be positive")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.threads is not None:
        if args.threads < 1:
            raise ValueError("threads must be positive")
        torch.set_num_threads(args.threads)

    contract = load_pretraining_execution_contract(
        ROOT / "configs" / "pretraining-execution-l004.json",
        repo_root=ROOT,
        external_root=args.external_root,
    )
    tokenizer = verify_base50m_tokenizer(
        contract.tokenizer_path,
        expected_sha256=contract.tokenizer_sha256,
        expected_vocab_size=contract.tokenizer_vocab_size,
    )
    training = load_training_configuration(
        ROOT / "configs" / "training-base-50m-v1.json"
    )
    schedule = build_primary_schedule(training)

    torch.manual_seed(contract.primary_seed)
    init_start = time.perf_counter()
    model = BlankCausalLM(contract.model_config)
    init_seconds = time.perf_counter() - init_start
    if model.parameter_count != 50_213_376:
        raise RuntimeError("Base-50M runtime parameter count changed")

    optimizer = build_adamw(model, training)
    ids = _sample_ids(tokenizer, args.sequence_length)
    input_ids = torch.tensor(
        ids[:-1],
        dtype=torch.long,
    ).unsqueeze(0).repeat(args.batch_size, 1)
    targets = torch.tensor(
        ids[1:],
        dtype=torch.long,
    ).unsqueeze(0).repeat(args.batch_size, 1)
    batches = [
        (input_ids.clone(), targets.clone())
        for _ in range(args.microbatches)
    ]

    learning_rate = learning_rate_for_update(schedule, 1)
    print(
        "qualification_update_start "
        f"sequence_length={args.sequence_length} "
        f"batch_size={args.batch_size} "
        f"microbatches={args.microbatches}",
        flush=True,
    )
    update_start = time.perf_counter()
    autocast_dtype = (
        torch.bfloat16 if args.precision == "bf16" else None
    )
    result = pretraining_update(
        model,
        optimizer,
        batches,
        learning_rate=learning_rate,
        clip_norm=gradient_clip_norm(training),
        autocast_dtype=autocast_dtype,
    )
    update_seconds = time.perf_counter() - update_start
    tokens = (
        args.sequence_length
        * args.batch_size
        * args.microbatches
    )
    tokens_per_second = tokens / update_seconds

    memory = _windows_memory_bytes()
    payload = {
        "schema_version": 1,
        "stage": "L004",
        "qualification_only": True,
        "production_optimizer_steps_recorded": 0,
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "device": "cpu",
        "precision": args.precision,
        "threads": torch.get_num_threads(),
        "mkldnn_available": bool(torch.backends.mkldnn.is_available()),
        "process_memory": memory,
        "model_parameters": model.parameter_count,
        "sequence_length": args.sequence_length,
        "batch_size": args.batch_size,
        "microbatches": args.microbatches,
        "tokens_processed": tokens,
        "model_init_seconds": init_seconds,
        "update_seconds": update_seconds,
        "tokens_per_second": tokens_per_second,
        "mean_loss": result.mean_loss,
        "gradient_norm": result.gradient_norm,
        "learning_rate": result.learning_rate,
        "checkpoint_lane": contract.checkpoint_lane,
        "tokenizer_sha256": contract.tokenizer_sha256,
        "global_manifest_report_sha256": (
            contract.global_manifest_report_sha256
        ),
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    print("base50m_ephemeral_optimizer_qualification=PASS")
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            rendered + "\n",
            encoding="utf-8",
            newline="\n",
        )


if __name__ == "__main__":
    main()
