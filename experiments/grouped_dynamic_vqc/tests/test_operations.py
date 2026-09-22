"""A resumed worker must not inherit the monitor's old terminal state."""

from __future__ import annotations

from datetime import UTC, datetime

from ..operations import status_table
from ..runtime import atomic_json


def test_monitor_waits_for_fresh_heartbeat_after_resume(tmp_path):
    now = datetime.now(UTC)
    heartbeat = {
        "updated_at": now.isoformat(),
        "status": "paused",
        "pid": 999999999,
        "epoch_completed": 0,
        "epochs_total": 10,
        "global_step": 3,
        "offset": 3072,
        "device": "CUDA",
    }
    atomic_json(tmp_path / "heartbeat.json", heartbeat)
    assert status_table(tmp_path)[1] == "paused"
    assert status_table(tmp_path, since=now.timestamp() + 1)[1] == "starting"
