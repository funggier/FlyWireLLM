from __future__ import annotations

import hashlib
import json
import sqlite3
from array import array
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from .data_contract import dedup_fingerprint
from .manifest_inputs import (
    GlobalManifestInput,
    GlobalManifestInputPlan,
    verify_global_manifest_input_file,
)
from .mixing import MixtureContract, evaluate_mixture_readiness
from .near_duplicate import SQLiteSimhashIndex
from .quality import simhash64
from .technical import classify_technical_text


class GlobalManifestError(ValueError):
    pass


_PARTITION_PRIORITY = {
    "holdout": 0,
    "validation": 1,
    "train": 2,
}
_LANE_PRIORITY = {
    "release_safe": 0,
    "research_only": 1,
}
_GENERAL_CATEGORY = {
    "th": "general_thai",
    "en": "general_english",
}
_ALLOWED_CATEGORIES = {
    "general_thai",
    "general_english",
    "technical_scientific_code",
    "flywire_domain",
}


@dataclass(frozen=True)
class GlobalCandidate:
    record_id: str
    text: str
    language: str
    source_id: str
    artifact_id: str
    source_group_id: str
    partition: str
    rights_lane: str
    content_sha256: str
    dedup_fingerprint: str
    simhash64_hex: str

    @property
    def canonical_id(self) -> str:
        return (
            f"{self.source_id}::{self.artifact_id}::{self.record_id}"
        )


@dataclass(frozen=True)
class GlobalDecision:
    canonical_id: str
    record_id: str
    source_id: str
    artifact_id: str
    partition: str
    rights_lane: str
    primary_category: str | None
    status: str
    reason: str | None
    duplicate_of: str | None
    hamming_distance: int | None
    token_count: int


def priority_key(candidate: GlobalCandidate) -> tuple:
    """Canonical cross-source order.

    Partition and rights lane dominate to prevent a train copy from becoming
    canonical over validation/holdout, and to prefer release-safe material over
    research-only material. Within one immutable source artifact, file order is
    the final deterministic tiebreaker and is pinned by the artifact SHA-256.
    """
    try:
        partition_priority = _PARTITION_PRIORITY[candidate.partition]
    except KeyError as exc:
        raise GlobalManifestError(
            f"invalid partition {candidate.partition!r}"
        ) from exc
    try:
        lane_priority = _LANE_PRIORITY[candidate.rights_lane]
    except KeyError as exc:
        raise GlobalManifestError(
            f"invalid rights lane {candidate.rights_lane!r}"
        ) from exc
    return (
        partition_priority,
        lane_priority,
        candidate.source_id,
        candidate.artifact_id,
    )


def category_for_text(
    *,
    text: str,
    language: str,
    declared_domain: str | None = None,
) -> str:
    if declared_domain == "flywire_domain":
        return "flywire_domain"
    technical = classify_technical_text(text)
    if technical.category == "technical_scientific_code":
        return "technical_scientific_code"
    try:
        return _GENERAL_CATEGORY[language]
    except KeyError as exc:
        raise GlobalManifestError(
            f"cannot route non-technical language {language!r}"
        ) from exc


def parse_global_candidate(payload: dict) -> GlobalCandidate:
    required = {
        "record_id",
        "text",
        "language",
        "source_id",
        "artifact_id",
        "source_group_id",
        "partition",
        "rights_lane",
        "content_sha256",
        "dedup_fingerprint",
        "simhash64",
    }
    missing = required - set(payload)
    if missing:
        raise GlobalManifestError(
            "candidate missing fields: " + ", ".join(sorted(missing))
        )
    values: dict[str, str] = {}
    for field in required:
        value = payload[field]
        if not isinstance(value, str) or not value:
            raise GlobalManifestError(
                f"{field} must be a non-empty string"
            )
        values[field] = value

    for field, length in (
        ("content_sha256", 64),
        ("dedup_fingerprint", 64),
        ("simhash64", 16),
    ):
        if len(values[field]) != length:
            raise GlobalManifestError(
                f"{field} must be {length} hex chars"
            )
        try:
            int(values[field], 16)
        except ValueError as exc:
            raise GlobalManifestError(
                f"{field} must be hexadecimal"
            ) from exc

    if values["partition"] not in _PARTITION_PRIORITY:
        raise GlobalManifestError("invalid partition")
    if values["rights_lane"] not in _LANE_PRIORITY:
        raise GlobalManifestError("invalid rights_lane")
    if values["language"] not in {"th", "en", "mixed"}:
        raise GlobalManifestError("unsupported language")

    text = values["text"]
    observed_content_sha = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()
    if observed_content_sha != values["content_sha256"]:
        raise GlobalManifestError(
            "content_sha256 does not match text"
        )
    observed_dedup = dedup_fingerprint(text)
    if observed_dedup != values["dedup_fingerprint"]:
        raise GlobalManifestError(
            "dedup_fingerprint does not match text"
        )
    observed_simhash = f"{simhash64(text):016x}"
    if observed_simhash != values["simhash64"]:
        raise GlobalManifestError(
            "simhash64 does not match text"
        )

    return GlobalCandidate(
        record_id=values["record_id"],
        text=text,
        language=values["language"],
        source_id=values["source_id"],
        artifact_id=values["artifact_id"],
        source_group_id=values["source_group_id"],
        partition=values["partition"],
        rights_lane=values["rights_lane"],
        content_sha256=values["content_sha256"],
        dedup_fingerprint=values["dedup_fingerprint"],
        simhash64_hex=values["simhash64"],
    )


class CompactGlobalDeduplicator:
    """Memory-compact exact/near dedup for multi-million-record global runs.

    Four 16-bit LSH tables guarantee candidate capture for Hamming distance
    <=3. Buckets store only unsigned 32-bit accepted-record indexes.
    """

    _BANDS = 4
    _BAND_MASK = 0xFFFF

    def __init__(self, *, near_duplicate_distance: int = 3) -> None:
        if not 0 <= near_duplicate_distance <= 3:
            raise GlobalManifestError(
                "near_duplicate_distance must remain in [0,3]"
            )
        self.near_duplicate_distance = near_duplicate_distance
        self._exact_seen: dict[bytes, int] = {}
        self._canonical_ids: list[str] = []
        self._canonical_id_seen: set[str] = set()
        self._simhashes = array("Q")
        self._buckets = [
            [array("I") for _ in range(1 << 16)]
            for _ in range(self._BANDS)
        ]
        self._last_priority: tuple | None = None

    def close(self) -> None:
        return None

    def __enter__(self) -> "CompactGlobalDeduplicator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @classmethod
    def _band_value(cls, fingerprint: int, band: int) -> int:
        return (fingerprint >> (band * 16)) & cls._BAND_MASK

    def _near_match(self, fingerprint: int) -> tuple[str, int] | None:
        best_id: str | None = None
        best_distance = self.near_duplicate_distance + 1
        for band in range(self._BANDS):
            value = self._band_value(fingerprint, band)
            for index in self._buckets[band][value]:
                distance = (
                    fingerprint ^ int(self._simhashes[index])
                ).bit_count()
                if distance > self.near_duplicate_distance:
                    continue
                candidate_id = self._canonical_ids[index]
                if (
                    distance < best_distance
                    or (
                        distance == best_distance
                        and (best_id is None or candidate_id < best_id)
                    )
                ):
                    best_distance = distance
                    best_id = candidate_id
        if best_id is None:
            return None
        return best_id, best_distance

    def consider(
        self,
        candidate: GlobalCandidate,
        *,
        token_count: int,
        primary_category: str,
    ) -> GlobalDecision:
        if primary_category not in _ALLOWED_CATEGORIES:
            raise GlobalManifestError("invalid primary category")
        if token_count <= 0:
            raise GlobalManifestError("token_count must be positive")
        priority = priority_key(candidate)
        if (
            self._last_priority is not None
            and priority < self._last_priority
        ):
            raise GlobalManifestError(
                "candidates are not in canonical priority order"
            )
        self._last_priority = priority

        canonical_id = candidate.canonical_id
        if canonical_id in self._canonical_id_seen:
            raise GlobalManifestError(
                f"duplicate canonical_id {canonical_id!r}"
            )
        self._canonical_id_seen.add(canonical_id)

        exact_key = bytes.fromhex(candidate.dedup_fingerprint)
        exact_index = self._exact_seen.get(exact_key)
        if exact_index is not None:
            return GlobalDecision(
                canonical_id=canonical_id,
                record_id=candidate.record_id,
                source_id=candidate.source_id,
                artifact_id=candidate.artifact_id,
                partition=candidate.partition,
                rights_lane=candidate.rights_lane,
                primary_category=None,
                status="exclude",
                reason="cross_source_exact_duplicate",
                duplicate_of=self._canonical_ids[exact_index],
                hamming_distance=0,
                token_count=0,
            )

        fingerprint = int(candidate.simhash64_hex, 16)
        near = self._near_match(fingerprint)
        if near is not None:
            duplicate_of, distance = near
            return GlobalDecision(
                canonical_id=canonical_id,
                record_id=candidate.record_id,
                source_id=candidate.source_id,
                artifact_id=candidate.artifact_id,
                partition=candidate.partition,
                rights_lane=candidate.rights_lane,
                primary_category=None,
                status="exclude",
                reason="cross_source_near_duplicate",
                duplicate_of=duplicate_of,
                hamming_distance=distance,
                token_count=0,
            )

        index = len(self._canonical_ids)
        if index > 0xFFFFFFFF:
            raise GlobalManifestError(
                "compact dedup index exceeded uint32 capacity"
            )
        self._canonical_ids.append(canonical_id)
        self._simhashes.append(fingerprint)
        self._exact_seen[exact_key] = index
        for band in range(self._BANDS):
            value = self._band_value(fingerprint, band)
            self._buckets[band][value].append(index)

        return GlobalDecision(
            canonical_id=canonical_id,
            record_id=candidate.record_id,
            source_id=candidate.source_id,
            artifact_id=candidate.artifact_id,
            partition=candidate.partition,
            rights_lane=candidate.rights_lane,
            primary_category=primary_category,
            status="accept",
            reason=None,
            duplicate_of=None,
            hamming_distance=None,
            token_count=token_count,
        )

    @property
    def accepted_record_count(self) -> int:
        return len(self._canonical_ids)


class GlobalDeduplicator:
    """Deterministic cross-source exact/near deduplication.

    Callers feed records in non-decreasing partition/lane/source/artifact
    priority. Immutable order inside one accepted artifact is pinned by that
    artifact's SHA-256.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        near_duplicate_distance: int = 3,
        commit_interval: int = 2_000,
    ) -> None:
        if not 0 <= near_duplicate_distance <= 3:
            raise GlobalManifestError(
                "near_duplicate_distance must remain in [0,3]"
            )
        if commit_interval < 1:
            raise GlobalManifestError("commit_interval must be positive")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.near_duplicate_distance = near_duplicate_distance
        self.commit_interval = commit_interval
        self._db = sqlite3.connect(directory / "global.sqlite")
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS exact_seen (
                fingerprint TEXT PRIMARY KEY,
                canonical_id TEXT NOT NULL
            )
            """
        )
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS canonical_ids (
                canonical_id TEXT PRIMARY KEY
            )
            """
        )
        self._db.commit()
        self._near = SQLiteSimhashIndex(
            directory / "simhash.sqlite",
            bands=4,
        )
        self._last_priority: tuple | None = None
        self._pending = 0

    def _flush(self) -> None:
        if self._pending:
            self._db.commit()
            self._near.commit()
            self._pending = 0

    def close(self) -> None:
        self._flush()
        self._near.close()
        self._db.close()

    def __enter__(self) -> "GlobalDeduplicator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def consider(
        self,
        candidate: GlobalCandidate,
        *,
        token_count: int,
        primary_category: str,
    ) -> GlobalDecision:
        if primary_category not in _ALLOWED_CATEGORIES:
            raise GlobalManifestError("invalid primary category")
        if token_count <= 0:
            raise GlobalManifestError(
                "token_count must be positive"
            )
        priority = priority_key(candidate)
        if (
            self._last_priority is not None
            and priority < self._last_priority
        ):
            raise GlobalManifestError(
                "candidates are not in canonical priority order"
            )
        self._last_priority = priority

        canonical_id = candidate.canonical_id
        existing_id = self._db.execute(
            "SELECT 1 FROM canonical_ids WHERE canonical_id=?",
            (canonical_id,),
        ).fetchone()
        if existing_id is not None:
            raise GlobalManifestError(
                f"duplicate canonical_id {canonical_id!r}"
            )

        row = self._db.execute(
            "SELECT canonical_id FROM exact_seen WHERE fingerprint=?",
            (candidate.dedup_fingerprint,),
        ).fetchone()
        if row is not None:
            return GlobalDecision(
                canonical_id=canonical_id,
                record_id=candidate.record_id,
                source_id=candidate.source_id,
                artifact_id=candidate.artifact_id,
                partition=candidate.partition,
                rights_lane=candidate.rights_lane,
                primary_category=None,
                status="exclude",
                reason="cross_source_exact_duplicate",
                duplicate_of=str(row[0]),
                hamming_distance=0,
                token_count=0,
            )

        fingerprint = int(candidate.simhash64_hex, 16)
        matches = self._near.find_matches(
            fingerprint,
            max_hamming_distance=self.near_duplicate_distance,
        )
        if matches:
            match = min(
                matches,
                key=lambda item: (
                    item.hamming_distance,
                    item.record_id,
                ),
            )
            return GlobalDecision(
                canonical_id=canonical_id,
                record_id=candidate.record_id,
                source_id=candidate.source_id,
                artifact_id=candidate.artifact_id,
                partition=candidate.partition,
                rights_lane=candidate.rights_lane,
                primary_category=None,
                status="exclude",
                reason="cross_source_near_duplicate",
                duplicate_of=match.record_id,
                hamming_distance=match.hamming_distance,
                token_count=0,
            )

        try:
            self._db.execute(
                "INSERT INTO exact_seen(fingerprint,canonical_id) VALUES(?,?)",
                (candidate.dedup_fingerprint, canonical_id),
            )
            self._db.execute(
                "INSERT INTO canonical_ids(canonical_id) VALUES(?)",
                (canonical_id,),
            )
            self._near.add(
                record_id=canonical_id,
                fingerprint=fingerprint,
                commit=False,
            )
        except sqlite3.IntegrityError as exc:
            self._db.rollback()
            raise GlobalManifestError(
                f"failed to register canonical record {canonical_id!r}"
            ) from exc

        self._pending += 1
        if self._pending >= self.commit_interval:
            self._flush()

        return GlobalDecision(
            canonical_id=canonical_id,
            record_id=candidate.record_id,
            source_id=candidate.source_id,
            artifact_id=candidate.artifact_id,
            partition=candidate.partition,
            rights_lane=candidate.rights_lane,
            primary_category=primary_category,
            status="accept",
            reason=None,
            duplicate_of=None,
            hamming_distance=None,
            token_count=token_count,
        )

    @property
    def accepted_record_count(self) -> int:
        row = self._db.execute(
            "SELECT COUNT(*) FROM canonical_ids"
        ).fetchone()
        assert row is not None
        return int(row[0])


def _manifest_input_sort_key(item: GlobalManifestInput) -> tuple:
    try:
        lane_priority = _LANE_PRIORITY[item.rights_lane]
    except KeyError as exc:
        raise GlobalManifestError(
            f"invalid manifest input rights lane {item.rights_lane!r}"
        ) from exc
    return (
        lane_priority,
        item.source_id,
        item.artifact_id,
        item.input_id,
    )


def _candidate_from_input_row(
    row: dict,
    item: GlobalManifestInput,
) -> GlobalCandidate:
    if not isinstance(row, dict):
        raise GlobalManifestError(
            f"{item.input_id}: accepted row must be an object"
        )
    expected = {
        "source_id": item.source_id,
        "artifact_id": item.artifact_id,
        "language": item.language,
    }
    for field, value in expected.items():
        if row.get(field) != value:
            raise GlobalManifestError(
                f"{item.input_id}: row {field} does not match registry"
            )
    if "rights_lane" in row and row["rights_lane"] != item.rights_lane:
        raise GlobalManifestError(
            f"{item.input_id}: row rights_lane does not match registry"
        )
    payload = dict(row)
    payload["rights_lane"] = item.rights_lane
    return parse_global_candidate(payload)


def _decision_payload(
    candidate: GlobalCandidate,
    decision: GlobalDecision,
) -> dict:
    return {
        "canonical_id": decision.canonical_id,
        "record_id": decision.record_id,
        "source_id": decision.source_id,
        "artifact_id": decision.artifact_id,
        "source_group_id": candidate.source_group_id,
        "language": candidate.language,
        "partition": decision.partition,
        "rights_lane": decision.rights_lane,
        "primary_category": decision.primary_category,
        "status": decision.status,
        "reason": decision.reason,
        "duplicate_of": decision.duplicate_of,
        "hamming_distance": decision.hamming_distance,
        "token_count": decision.token_count,
    }


def _category_totals() -> dict[str, int]:
    return {category: 0 for category in sorted(_ALLOWED_CATEGORIES)}


def evaluate_global_manifest_mixture(
    report: dict,
    contract: MixtureContract,
) -> dict:
    capacities = report.get("train_capacity")
    if not isinstance(capacities, dict):
        raise GlobalManifestError("global report missing train_capacity")
    result: dict[str, object] = {}
    for checkpoint_lane in ("release_safe", "research_only"):
        available = capacities.get(checkpoint_lane)
        if not isinstance(available, dict):
            raise GlobalManifestError(
                f"global report missing {checkpoint_lane} train capacity"
            )
        result[checkpoint_lane] = asdict(
            evaluate_mixture_readiness(
                contract,
                available_tokens=available,
            )
        )
    result["pretraining_authorized"] = False
    return result


def materialize_global_manifest(
    plan: GlobalManifestInputPlan,
    *,
    external_root: str | Path,
    output_dir: str | Path,
    token_counter: Callable[[str], int],
) -> dict:
    """Verify, globally deduplicate, and account one frozen input plan.

    Raw text is consumed only from the externally stored accepted corpora. The
    emitted decision manifest is metadata-only and follows canonical priority:
    holdout, validation, train; then release-safe before research-only.
    """
    if plan.pretraining_authorized is not False:
        raise GlobalManifestError(
            "global manifest input plan cannot authorize pretraining"
        )
    if plan.near_duplicate_hamming_distance != 3:
        raise GlobalManifestError(
            "L003 global near-duplicate distance must remain 3"
        )
    if not callable(token_counter):
        raise TypeError("token_counter must be callable")

    verified: list[tuple[GlobalManifestInput, Path]] = []
    for item in plan.inputs:
        path = verify_global_manifest_input_file(
            item,
            external_root=external_root,
        )
        verified.append((item, path))

    output_dir = Path(output_dir)
    decisions_path = output_dir / "decisions.jsonl"
    if decisions_path.exists():
        raise GlobalManifestError(
            "global manifest output already exists; use a fresh output directory"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    verified.sort(key=lambda pair: _manifest_input_sort_key(pair[0]))
    records = Counter()
    accepted_tokens: Counter[tuple[str, str, str, str]] = Counter()
    decision_digest = hashlib.sha256()
    decision_bytes = 0

    with decisions_path.open("wb") as decision_stream, CompactGlobalDeduplicator(
        near_duplicate_distance=plan.near_duplicate_hamming_distance,
    ) as dedup:
        for partition in ("holdout", "validation", "train"):
            for item, path in verified:
                with path.open("r", encoding="utf-8") as stream:
                    for line_number, raw in enumerate(stream, start=1):
                        if not raw.strip():
                            continue
                        try:
                            row = json.loads(raw)
                        except json.JSONDecodeError as exc:
                            raise GlobalManifestError(
                                f"{item.input_id}:{line_number}: invalid JSON"
                            ) from exc
                        if row.get("partition") != partition:
                            continue
                        candidate = _candidate_from_input_row(row, item)
                        primary_category = category_for_text(
                            text=candidate.text,
                            language=candidate.language,
                        )
                        token_count = token_counter(candidate.text)
                        if not isinstance(token_count, int) or token_count <= 0:
                            raise GlobalManifestError(
                                f"{item.input_id}:{line_number}: token counter "
                                "must return a positive integer"
                            )
                        decision = dedup.consider(
                            candidate,
                            token_count=token_count,
                            primary_category=primary_category,
                        )
                        records["total"] += 1
                        if decision.status == "accept":
                            records["accepted"] += 1
                            accepted_tokens[
                                (
                                    candidate.partition,
                                    candidate.rights_lane,
                                    candidate.language,
                                    primary_category,
                                )
                            ] += token_count
                        else:
                            records["excluded"] += 1
                            if decision.reason == "cross_source_exact_duplicate":
                                records["exact_duplicates"] += 1
                            elif decision.reason == "cross_source_near_duplicate":
                                records["near_duplicates"] += 1

                        encoded = (
                            json.dumps(
                                _decision_payload(candidate, decision),
                                ensure_ascii=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            )
                            + "\n"
                        ).encode("utf-8")
                        decision_stream.write(encoded)
                        decision_digest.update(encoded)
                        decision_bytes += len(encoded)

    by_partition: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
    for (partition, lane, language, category), tokens in sorted(
        accepted_tokens.items()
    ):
        by_partition.setdefault(partition, {}).setdefault(lane, {}).setdefault(
            language,
            {},
        )[category] = tokens

    release_capacity = _category_totals()
    research_capacity = _category_totals()
    for (partition, lane, _language, category), tokens in accepted_tokens.items():
        if partition != "train":
            continue
        if lane == "release_safe":
            release_capacity[category] += tokens
            research_capacity[category] += tokens
        elif lane == "research_only":
            research_capacity[category] += tokens

    return {
        "schema_version": 1,
        "stage": "L003",
        "tokenizer": plan.tokenizer,
        "technical_classifier": plan.technical_classifier,
        "near_duplicate_hamming_distance": (
            plan.near_duplicate_hamming_distance
        ),
        "inputs_verified": [item.input_id for item, _ in verified],
        "records": {
            "total": records["total"],
            "accepted": records["accepted"],
            "excluded": records["excluded"],
            "exact_duplicates": records["exact_duplicates"],
            "near_duplicates": records["near_duplicates"],
        },
        "accepted_tokens": {"by_partition": by_partition},
        "train_capacity": {
            "release_safe": release_capacity,
            "research_only": research_capacity,
        },
        "decisions": {
            "path": "decisions.jsonl",
            "bytes": decision_bytes,
            "sha256": decision_digest.hexdigest(),
            "contains_raw_text": False,
        },
        "pretraining_authorized": False,
    }
