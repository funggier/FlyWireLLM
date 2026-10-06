from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class RightsPolicyError(ValueError):
    pass


@dataclass(frozen=True)
class SourceRights:
    source_id: str
    lane: str
    reason: str


@dataclass(frozen=True)
class RightsPolicy:
    primary_target_tokens: int
    sources: dict[str, SourceRights]
    known_token_accounting: dict[str, Any]
    pretraining_authorization: dict[str, Any]


_VALID_LANES = {"release_safe", "research_only", "blocked"}


def load_rights_policy(
    path: str | Path,
    *,
    expected_source_ids: set[str] | None = None,
) -> RightsPolicy:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RightsPolicyError("rights policy must be a JSON object")
    if payload.get("schema_version") != 1:
        raise RightsPolicyError("unsupported rights policy schema")
    if payload.get("stage") != "L003":
        raise RightsPolicyError("rights policy stage must be L003")
    if payload.get("primary_target_tokens") != 500_000_000:
        raise RightsPolicyError("primary target must remain 500M tokens")

    lanes = payload.get("lanes")
    if not isinstance(lanes, dict) or set(lanes) != {
        "release_safe",
        "research_only",
    }:
        raise RightsPolicyError("training lane set changed")
    if lanes["release_safe"].get("allowed_source_lanes") != [
        "release_safe"
    ]:
        raise RightsPolicyError(
            "release_safe checkpoint may only use release_safe sources"
        )
    if set(lanes["research_only"].get("allowed_source_lanes", [])) != {
        "release_safe",
        "research_only",
    }:
        raise RightsPolicyError(
            "research_only checkpoint source-lane contract changed"
        )
    if payload.get("blocked_source_lane") != "blocked":
        raise RightsPolicyError("blocked source lane changed")

    promotion = payload.get("promotion_policy")
    if not isinstance(promotion, dict):
        raise RightsPolicyError("promotion_policy must be an object")
    if promotion.get("research_only_to_release_safe") is not False:
        raise RightsPolicyError(
            "research-only lineage must never promote to release-safe"
        )
    if promotion.get("release_safe_to_research_only") is not True:
        raise RightsPolicyError(
            "release-safe lineage may be consumed by research-only runs"
        )
    if (
        promotion.get("checkpoint_lane_immutable_after_first_optimizer_step")
        is not True
    ):
        raise RightsPolicyError(
            "checkpoint lane must be immutable after training starts"
        )

    raw_sources = payload.get("source_assignments")
    if not isinstance(raw_sources, dict) or not raw_sources:
        raise RightsPolicyError("source_assignments must be non-empty")
    sources: dict[str, SourceRights] = {}
    for source_id, raw in raw_sources.items():
        if not isinstance(raw, dict):
            raise RightsPolicyError(
                f"{source_id} rights assignment must be an object"
            )
        lane = raw.get("lane")
        if lane not in _VALID_LANES:
            raise RightsPolicyError(
                f"{source_id} has invalid rights lane {lane!r}"
            )
        reason = raw.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise RightsPolicyError(
                f"{source_id} must have a rights-lane reason"
            )
        sources[source_id] = SourceRights(
            source_id=source_id,
            lane=lane,
            reason=reason.strip(),
        )

    if expected_source_ids is not None:
        if set(sources) != expected_source_ids:
            missing = sorted(expected_source_ids - set(sources))
            extra = sorted(set(sources) - expected_source_ids)
            raise RightsPolicyError(
                "rights/source inventory mismatch "
                f"missing={missing} extra={extra}"
            )

    accounting = payload.get("known_token_accounting")
    if not isinstance(accounting, dict):
        raise RightsPolicyError("known_token_accounting must be an object")
    if accounting.get("tokenizer") != "base50m-unigram-32000-v1":
        raise RightsPolicyError("token accounting tokenizer changed")
    if accounting.get("quality_audit") != (
        "results/l003/acquired-quality-audit.json"
    ):
        raise RightsPolicyError("quality audit path changed")
    if accounting.get("quality_audit_sha256") != (
        "f7e6c3ed4e3d5e915ecc319fc7cec00ea0132e40c9a1ea31769397126f309988"
    ):
        raise RightsPolicyError("quality audit hash changed")

    pre_screen = accounting.get("pre_screen_train")
    screened = accounting.get("screened_train")
    if not isinstance(pre_screen, dict) or not isinstance(screened, dict):
        raise RightsPolicyError(
            "pre_screen_train and screened_train must be objects"
        )

    pre_release = pre_screen.get("release_safe")
    pre_research = pre_screen.get("research_only")
    if not isinstance(pre_release, dict) or not isinstance(pre_research, dict):
        raise RightsPolicyError("pre-screen lane accounting must be objects")
    if int(pre_release.get("total", -1)) != 20_062_981:
        raise RightsPolicyError(
            "pre-screen release-safe baseline token count changed"
        )
    if int(pre_research.get("total", -1)) != 38_312_940:
        raise RightsPolicyError(
            "pre-screen research-only baseline token count changed"
        )
    if pre_screen.get("combined_research_eligible") != 58_375_921:
        raise RightsPolicyError(
            "pre-screen combined research token count changed"
        )

    release = screened.get("release_safe")
    research = screened.get("research_only")
    if not isinstance(release, dict) or not isinstance(research, dict):
        raise RightsPolicyError("screened lane accounting must be objects")
    release_total = int(release.get("total", -1))
    research_total = int(research.get("total", -1))
    if release_total != 20_061_246:
        raise RightsPolicyError(
            "screened release-safe token count changed"
        )
    if research_total != 38_172_496:
        raise RightsPolicyError(
            "screened research-only token count changed"
        )
    if screened.get("combined_research_eligible") != (
        release_total + research_total
    ):
        raise RightsPolicyError(
            "screened combined research token accounting mismatch"
        )
    if screened.get("release_safe_gap_to_500m") != (
        500_000_000 - release_total
    ):
        raise RightsPolicyError(
            "screened release-safe 500M gap mismatch"
        )
    if screened.get("combined_research_gap_to_500m") != (
        500_000_000 - release_total - research_total
    ):
        raise RightsPolicyError(
            "screened research 500M gap mismatch"
        )

    authorization = payload.get("pretraining_authorization")
    if not isinstance(authorization, dict):
        raise RightsPolicyError(
            "pretraining_authorization must be an object"
        )
    if authorization.get("release_safe_requires_tokens") != 500_000_000:
        raise RightsPolicyError(
            "release_safe pretraining budget must remain 500M"
        )
    if authorization.get("research_only_requires_tokens") != 500_000_000:
        raise RightsPolicyError(
            "research_only pretraining budget must remain 500M"
        )
    for field in (
        "source_manifest_frozen",
        "evaluation_manifest_frozen",
        "tokenizer_hash_pinned",
        "checkpoint_lane_declared_before_training",
    ):
        if authorization.get(field) is not True:
            raise RightsPolicyError(f"{field} must remain true")

    return RightsPolicy(
        primary_target_tokens=500_000_000,
        sources=sources,
        known_token_accounting=dict(accounting),
        pretraining_authorization=dict(authorization),
    )


def validate_checkpoint_sources(
    policy: RightsPolicy,
    *,
    checkpoint_lane: str,
    source_ids: Iterable[str],
) -> tuple[str, ...]:
    if checkpoint_lane not in {"release_safe", "research_only"}:
        raise RightsPolicyError(
            "checkpoint_lane must be release_safe or research_only"
        )
    source_ids = tuple(source_ids)
    if not source_ids:
        raise RightsPolicyError("checkpoint source_ids must not be empty")
    if len(source_ids) != len(set(source_ids)):
        raise RightsPolicyError("checkpoint source_ids must be unique")

    for source_id in source_ids:
        source = policy.sources.get(source_id)
        if source is None:
            raise RightsPolicyError(
                f"unknown source_id {source_id!r}"
            )
        if source.lane == "blocked":
            raise RightsPolicyError(
                f"blocked source {source_id!r} cannot enter any checkpoint"
            )
        if (
            checkpoint_lane == "release_safe"
            and source.lane != "release_safe"
        ):
            raise RightsPolicyError(
                f"release_safe checkpoint cannot use "
                f"{source.lane} source {source_id!r}"
            )
    return source_ids


def can_promote_checkpoint(
    *,
    from_lane: str,
    to_lane: str,
) -> bool:
    if from_lane not in {"release_safe", "research_only"}:
        raise RightsPolicyError("invalid from_lane")
    if to_lane not in {"release_safe", "research_only"}:
        raise RightsPolicyError("invalid to_lane")
    if from_lane == to_lane:
        return True
    if from_lane == "release_safe" and to_lane == "research_only":
        return True
    return False


def training_budget_ready(
    policy: RightsPolicy,
    *,
    checkpoint_lane: str,
) -> bool:
    if checkpoint_lane == "release_safe":
        tokens = int(
            policy.known_token_accounting["screened_train"][
                "release_safe"
            ]["total"]
        )
    elif checkpoint_lane == "research_only":
        tokens = int(
            policy.known_token_accounting["screened_train"][
                "combined_research_eligible"
            ]
        )
    else:
        raise RightsPolicyError("invalid checkpoint_lane")
    return tokens >= policy.primary_target_tokens


def token_gap(
    policy: RightsPolicy,
    *,
    checkpoint_lane: str,
) -> int:
    if checkpoint_lane == "release_safe":
        tokens = int(
            policy.known_token_accounting["screened_train"][
                "release_safe"
            ]["total"]
        )
    elif checkpoint_lane == "research_only":
        tokens = int(
            policy.known_token_accounting["screened_train"][
                "combined_research_eligible"
            ]
        )
    else:
        raise RightsPolicyError("invalid checkpoint_lane")
    return max(0, policy.primary_target_tokens - tokens)
