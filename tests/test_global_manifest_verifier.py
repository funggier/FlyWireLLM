from __future__ import annotations

import hashlib

import pytest

import scripts.verify_global_manifest_l003 as verifier


def test_verify_file_accepts_exact_size_and_hash(tmp_path):
    path = tmp_path / "artifact.bin"
    payload = b"verified-artifact"
    path.write_bytes(payload)
    verifier.verify_file(
        path,
        expected_bytes=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        label="artifact",
    )


def test_verify_file_rejects_hash_mismatch(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"verified-artifact")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        verifier.verify_file(
            path,
            expected_bytes=path.stat().st_size,
            expected_sha256="0" * 64,
            label="artifact",
        )


def test_verify_file_rejects_size_mismatch_before_hash(tmp_path):
    path = tmp_path / "artifact.bin"
    path.write_bytes(b"verified-artifact")
    with pytest.raises(RuntimeError, match="byte size mismatch"):
        verifier.verify_file(
            path,
            expected_bytes=1,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            label="artifact",
        )
