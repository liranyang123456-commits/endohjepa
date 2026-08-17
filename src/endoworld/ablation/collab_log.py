"""Privacy-conscious event logging for rehearsal usability studies."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any


_SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,80}$")
_EVENT_TYPES = {
    "task_started",
    "plan_generated",
    "plan_accepted",
    "plan_rejected",
    "plan_edited",
    "needle_moved",
    "ablation_started",
    "ablation_ended",
    "sensitivity_viewed",
    "task_completed",
    "sus_submitted",
}


@dataclass(frozen=True)
class CollabEvent:
    """A non-identifying interaction event suitable for a feasibility study."""

    session_id: str
    event_type: str
    payload: dict[str, Any]
    timestamp_utc: str
    user_id: str | None = None


def _validate_session_id(session_id: str) -> str:
    if not _SESSION_RE.fullmatch(session_id):
        raise ValueError("session_id must contain only letters, digits, '_' or '-'.")
    return session_id


def append_event(
    root: str | Path,
    *,
    session_id: str,
    event_type: str,
    payload: dict[str, Any] | None = None,
    user_id: str | None = None,
) -> CollabEvent:
    """Append one event to an isolated session JSONL file."""
    session_id = _validate_session_id(str(session_id))
    if event_type not in _EVENT_TYPES:
        raise ValueError(f"Unsupported event_type {event_type!r}.")
    event = CollabEvent(
        session_id=session_id,
        event_type=event_type,
        payload=payload or {},
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        user_id=user_id,
    )
    directory = Path(root) / session_id
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
    return event


def _read_events(root: str | Path, session_id: str) -> list[dict[str, Any]]:
    session_id = _validate_session_id(str(session_id))
    path = Path(root) / session_id / "events.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def study_summary(root: str | Path, session_id: str) -> dict[str, Any]:
    """Derive pre-specified usability outcomes without retaining patient data."""
    events = _read_events(root, session_id)
    counts = {event_type: 0 for event_type in _EVENT_TYPES}
    for event in events:
        if event["event_type"] in counts:
            counts[event["event_type"]] += 1
    task_start = next((item for item in events if item["event_type"] == "task_started"), None)
    task_end = next(
        (item for item in reversed(events) if item["event_type"] == "task_completed"),
        None,
    )
    duration_s = None
    if task_start and task_end:
        start = datetime.fromisoformat(task_start["timestamp_utc"])
        end = datetime.fromisoformat(task_end["timestamp_utc"])
        duration_s = round((end - start).total_seconds(), 2)
    sus = next(
        (
            item["payload"].get("score")
            for item in reversed(events)
            if item["event_type"] == "sus_submitted"
        ),
        None,
    )
    return {
        "session_id": session_id,
        "event_count": len(events),
        "task_duration_s": duration_s,
        "plan_acceptance_count": counts["plan_accepted"],
        "plan_rejection_count": counts["plan_rejected"],
        "plan_override_count": counts["plan_edited"] + counts["needle_moved"],
        "task_completed": counts["task_completed"] > 0,
        "sus_score": sus,
        "event_counts": counts,
        "scope_note": (
            "Interaction metrics support workflow feasibility and usability "
            "assessment; they do not establish clinical effectiveness."
        ),
    }


def export_study_bundle(root: str | Path, session_id: str) -> dict[str, Any]:
    """Return an API-ready de-identified research bundle for one session."""
    return {
        "manifest": {
            "format": "rehearsal-study-bundle-v1",
            "session_id": _validate_session_id(str(session_id)),
            "contains_identifiers": False,
            "files": ["events.jsonl"],
        },
        "summary": study_summary(root, session_id),
        "events": _read_events(root, session_id),
    }
