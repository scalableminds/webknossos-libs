import warnings
from collections.abc import Iterable
from typing import Any

import numpy as np

UpdateGroup = tuple[int, list[dict[str, Any]]]


def _iron_out_reverts(
    update_groups_newest_first: Iterable[UpdateGroup],
) -> list[list[dict[str, Any]]]:
    """Drops all update groups that were undone by a revertToVersion action.

    Returns the remaining update groups from oldest to newest.
    """
    collected: list[list[dict[str, Any]]] = []
    next_version: int | None = None
    for version, actions in update_groups_newest_first:
        if next_version is None:
            next_version = version
        if version > next_version:
            continue
        revert_source_versions = [
            action["value"]["sourceVersion"]
            for action in actions
            if action["name"] == "revertToVersion"
        ]
        if revert_source_versions:
            next_version = revert_source_versions[0]
        else:
            collected.append(actions)
            next_version -= 1
    collected.reverse()
    return collected


def _parse_unsigned_long(value: Any) -> int | None:
    # Large ids are encoded as {"customJsonEncoding": "bigint", "value": "<decimal>"}
    if value is None:
        return None
    if isinstance(value, dict):
        return int(value["value"])
    return int(value)


def edited_edges_from_update_groups(
    update_groups_newest_first: Iterable[UpdateGroup], tracing_id: str
) -> tuple[np.ndarray, np.ndarray]:
    """Extracts the merged and split edges of a proofreading tracing from its update action log.

    Returns (edges, is_addition), see RemoteAnnotation.get_edited_edges.
    """
    edges: list[tuple[int, int]] = []
    is_addition: list[bool] = []
    skipped_legacy_actions = 0
    for actions in _iron_out_reverts(update_groups_newest_first):
        for action in actions:
            if action["name"] not in ("mergeAgglomerate", "splitAgglomerate"):
                continue
            value = action["value"]
            if value.get("actionTracingId") != tracing_id:
                continue
            segment_id1 = _parse_unsigned_long(value.get("segmentId1"))
            segment_id2 = _parse_unsigned_long(value.get("segmentId2"))
            if segment_id1 is None or segment_id2 is None:
                # Old actions only store positions instead of segment ids
                skipped_legacy_actions += 1
                continue
            edges.append((segment_id1, segment_id2))
            is_addition.append(action["name"] == "mergeAgglomerate")
    if skipped_legacy_actions > 0:
        warnings.warn(
            f"[WARNING] Skipped {skipped_legacy_actions} proofreading actions that "
            "reference segments by position instead of segment id.",
            UserWarning,
        )
    return (
        np.array(edges, dtype=np.uint64).reshape(-1, 2),
        np.array(is_addition, dtype=bool),
    )
