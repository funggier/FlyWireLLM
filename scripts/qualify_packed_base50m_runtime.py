from __future__ import annotations

import argparse
import ctypes
import json
import platform
import time
from pathlib import Path

import torch

from flywire_llm.model import BlankCausalLM
from flywire_llm.pretraining_data import (
    PackedDataCursor,
    PackedPretrainingData,
    cursor_state_dict,
    load_pretraining_data_contract,
)
from flywire_llm.pretraining_execution import (
    load_pretraining_execution_contract,
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
            "Run an ephemeral Base-50M optimizer update on the frozen "
            "L004 packed training stream. No production step is recorded."
        )
    )
    parser.add_argument("--external-root", required=True)
    parser.add_argument(
        "--supervised-budget",
        type=int,
        default=65_536,
    )
    parser.add_argument("--output-json")
    parser.add_argument(
        "--skip-external-hash-verification",
        action="store_true",
        help=(
            "Skip expensive packed-file hashes only when they were already "
            "verified in the same qualification flow."
        ),
    )
    return parser.parse_args()


def _load_local_runtime() -> dict:
    path = ROOT / "configs" / "pretraining-local-runtime-l004.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "stage": "L004",
        "runtime_id": "base50m-local-cpu-v1",
        "device": "cpu",
        "precision": "fp32",
        "threads": 12,
        "micro_batch_size": 1,
        "max_physical_sequence_length": 1024,
        "global_supervised_tokens_per_update": 65_536,
        "optimizer_steps": 7_630,
        "full_updates": 7_629,
        "final_update_supervised_tokens": 25_856,
        "bf16_status": "rejected_for_current_cpu_runtime",
        "production_optimizer_steps_recorded": 0,
        "public_release_eligibility": "not_qualified",
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(
                f"local runtime contract {key} changed"
            )
    return payload


def main() -> None:
    args = parse_args()
    if args.supervised_budget <= 0 or args.supervised_budget > 65_536:
        raise ValueError(
            "supervised budget must be in [1, 65536]"
        )

    execution = load_pretraining_execution_contract(
        ROOT / "configs" / "pretraining-execution-l004.json",
        repo_root=ROOT,
        external_root=args.external_root,
    )
    data_contract = load_pretraining_data_contract(
        ROOT / "configs" / "pretraining-data-runtime-l004.json",
        repo_root=ROOT,
        external_root=args.external_root,
        verify_external_hashes=(
            not args.skip_external_hash_verification
        ),
    )
    local = _load_local_runtime()
    if execution.data_runtime_sha256 != (
        "2d99c61c84fbfffb0ba5f23434bf6e8b92a92b3f588d7716238c0ebc76ee424b"
    ):
        raise RuntimeError(
            "execution/data runtime pin changed"
        )
    if local["max_physical_sequence_length"] != (
        data_contract.max_physical_sequence_length
    ):
        raise RuntimeError(
            "local sequence length does not match packed block contract"
        )

    torch.set_num_threads(int(local["threads"]))
    training = load_training_configuration(
        ROOT / "configs" / "training-base-50m-v1.json"
    )
    schedule = build_primary_schedule(training)
    torch.manual_seed(execution.primary_seed)

    init_start = time.perf_counter()
    model = BlankCausalLM(execution.model_config)
    init_seconds = time.perf_counter() - init_start
    if model.parameter_count != 50_213_376:
        raise RuntimeError("Base-50M parameter count changed")
    optimizer = build_adamw(model, training)

    cursor = PackedDataCursor()
    data_start = time.perf_counter()
    with PackedPretrainingData(data_contract) as data:
        packed = list(
            data.iter_update_microbatches(
                cursor,
                batch_size=int(local["micro_batch_size"]),
                max_input_length=int(
                    local["max_physical_sequence_length"]
                ),
                supervised_budget=args.supervised_budget,
            )
        )
    data_seconds = time.perf_counter() - data_start
    if not packed:
        raise RuntimeError("packed runtime produced no microbatches")
    final_cursor = packed[-1].cursor
    supervised = sum(
        batch.supervised_positions for batch in packed
    )
    physical_inputs = sum(
        int(batch.input_ids.numel()) for batch in packed
    )
    if supervised != args.supervised_budget:
        raise RuntimeError(
            "packed runtime did not hit exact supervised budget"
        )
    if final_cursor.supervised_tokens_consumed != supervised:
        raise RuntimeError(
            "packed cursor supervised progress mismatch"
        )

    batches = [
        (batch.input_ids, batch.targets)
        for batch in packed
    ]
    lr = learning_rate_for_update(schedule, 1)
    print(
        "packed_qualification_update_start "
        f"supervised_budget={supervised} "
        f"microbatches={len(batches)} "
        f"physical_inputs={physical_inputs}",
        flush=True,
    )
    update_start = time.perf_counter()
    result = pretraining_update(
        model,
        optimizer,
        batches,
        learning_rate=lr,
        clip_norm=gradient_clip_norm(training),
    )
    update_seconds = time.perf_counter() - update_start
    if result.supervised_tokens != supervised:
        raise RuntimeError(
            "optimizer supervised token accounting mismatch"
        )

    payload = {
        "schema_version": 1,
        "stage": "L004",
        "qualification_only": True,
        "uses_frozen_packed_training_data": True,
        "external_hash_verification_performed": (
            not args.skip_external_hash_verification
        ),
        "production_optimizer_steps_recorded": 0,
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "device": "cpu",
        "precision": "fp32",
        "threads": torch.get_num_threads(),
        "model_parameters": model.parameter_count,
        "max_physical_sequence_length": int(
            local["max_physical_sequence_length"]
        ),
        "batch_size": int(local["micro_batch_size"]),
        "microbatches": len(batches),
        "physical_input_positions": physical_inputs,
        "supervised_tokens": supervised,
        "masked_input_positions": physical_inputs - supervised,
        "data_cursor_after": cursor_state_dict(final_cursor),
        "model_init_seconds": init_seconds,
        "data_materialization_seconds": data_seconds,
        "update_seconds": update_seconds,
        "supervised_tokens_per_second": (
            supervised / update_seconds
        ),
        "mean_loss": result.mean_loss,
        "gradient_norm": result.gradient_norm,
        "learning_rate": result.learning_rate,
        "process_memory": _windows_memory_bytes(),
        "checkpoint_lane": execution.checkpoint_lane,
        "data_runtime_sha256": execution.data_runtime_sha256,
        "token_stream_sha256": data_contract.token_stream_sha256,
        "block_order_sha256": data_contract.block_order_sha256,
        "tokenizer_sha256": execution.tokenizer_sha256,
        "global_manifest_report_sha256": (
            execution.global_manifest_report_sha256
        ),
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    print("packed_base50m_ephemeral_update=PASS")
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            rendered + "\n",
            encoding="utf-8",
            newline="\n",
        )


if __name__ == "__main__":
    main()
