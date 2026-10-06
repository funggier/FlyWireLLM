from __future__ import annotations

import hashlib


class SamplingContractError(ValueError):
    pass


def stable_ppm_selected(
    record_id: str,
    *,
    seed: str,
    parts_per_million: int,
) -> bool:
    if not isinstance(record_id, str) or not record_id:
        raise SamplingContractError("record_id must be non-empty")
    if not isinstance(seed, str) or not seed:
        raise SamplingContractError("seed must be non-empty")
    if not 1 <= parts_per_million <= 1_000_000:
        raise SamplingContractError(
            "parts_per_million must be in [1, 1000000]"
        )
    digest = hashlib.sha256(
        (seed + "\0" + record_id).encode("utf-8")
    ).digest()
    bucket = int.from_bytes(digest[:8], "big") % 1_000_000
    return bucket < parts_per_million


def stable_source_group_id(
    *,
    source_url: str | None,
    record_id: str,
) -> str:
    if not isinstance(record_id, str) or not record_id:
        raise SamplingContractError("record_id must be non-empty")
    if source_url is not None and str(source_url).strip():
        value = "url:" + str(source_url).strip()
    else:
        value = "id:" + record_id
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
