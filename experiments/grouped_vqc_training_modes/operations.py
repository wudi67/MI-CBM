"""Rich launch/status views and detached tmux worker lifecycle."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = Path(__file__).resolve().parent / "scripts"
console = Console(highlight=False)


def status_table(output: Path, since: float = 0.0) -> tuple[Table, str]:
    table = Table(title="Grouped VQC: Joint / Sequential", show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value", overflow="fold")
    table.add_row("Output", str(output))
    path = output / "heartbeat.json"
    if not path.exists():
        table.add_row("Status", "Waiting for worker initialization")
        return table, "starting"
    heartbeat = json.loads(path.read_text(encoding="utf-8"))
    updated = datetime.fromisoformat(heartbeat["updated_at"])
    if updated.timestamp() < since:
        table.add_row("Status", "Waiting for the newly launched worker heartbeat")
        return table, "starting"
    state = heartbeat["status"]
    age = (datetime.now(UTC) - updated).total_seconds()
    pid = heartbeat["pid"]
    alive = Path(f"/proc/{pid}").exists()
    table.add_row("Status", state)
    table.add_row(
        "Route / phase",
        f"{heartbeat.get('route', 'pending')} / {heartbeat.get('phase', 'pending')}",
    )
    completed = sum(
        (output / route / "result.json").exists() for route in ("joint", "sequential")
    )
    table.add_row("Completed route artifacts", f"{completed} / 2")
    table.add_row("Worker", f"PID {pid}, process {'present' if alive else 'absent'}")
    table.add_row("Heartbeat age", f"{age:.1f} seconds")
    table.add_row(
        "Epoch", f"{heartbeat['epoch_completed']} / {heartbeat['epochs_total']}"
    )
    table.add_row("Optimizer steps", str(heartbeat["global_step"]))
    table.add_row("Rows in current epoch", str(heartbeat["offset"]))
    table.add_row("Device", heartbeat["device"])
    if "batch_loss" in heartbeat:
        table.add_row("Batch loss", f"{heartbeat['batch_loss']:.5f}")
    if "error" in heartbeat:
        table.add_row("Error", heartbeat["error"])
    if "validation" in heartbeat:
        validation = heartbeat["validation"]
        table.add_row(
            "Validation label accuracy", f"{validation['label']['accuracy']:.2%}"
        )
        table.add_row(
            "Validation joint concept MAP",
            f"{validation['concept']['joint_map_accuracy']:.2%}",
        )
    if not alive and state not in {"complete", "paused", "failed"}:
        table.add_row(
            "Recovery", "Worker absent; inspect train.log and route resume.pt files"
        )
        state = "absent"
    return table, state


def show_status(output: Path, watch: bool, since: float = 0.0) -> None:
    table, state = status_table(output, since)
    if not watch:
        console.print(table)
        return
    with Live(table, console=console, refresh_per_second=1) as live:
        while state not in {"complete", "paused", "failed", "absent"}:
            time.sleep(2)
            table, state = status_table(output, since)
            live.update(table)


def worker(output: Path, arguments: list[str]) -> int:
    """Keep both traceback and Rich text in the log and upper tmux pane."""
    environment = dict(os.environ)
    environment.pop("GROUPED_VQC_LOG", None)
    command = [str(SCRIPTS / "run.sh"), "--out", str(output), *arguments]
    with (output / "train.log").open("a", encoding="utf-8") as log:
        with subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                console.print(line, end="", markup=False)
            return process.wait()


def launch(output: Path, session: str, arguments: list[str]) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", session):
        raise ValueError("Use letters, numbers, underscore or dash for tmux session")
    tmux = Path(os.environ.get("VQC_PREFIX", "/root/miniforge3/envs/VQC")) / "bin/tmux"
    if not tmux.is_file():
        raise FileNotFoundError(f"tmux not found: {tmux}")
    if (
        subprocess.run(
            [str(tmux), "has-session", "-t", session], capture_output=True, check=False
        ).returncode
        == 0
    ):
        raise ValueError(f"tmux session already exists: {session}")
    if (output / "config.json").exists() and "--resume" not in arguments:
        raise FileExistsError("Run exists; add --resume or select a fresh --out")
    if "--resume" in arguments and not (output / "manifest.json").exists():
        raise FileNotFoundError("No shared manifest in selected output directory")
    output.mkdir(parents=True, exist_ok=True)
    command = shlex.join(
        [
            sys.executable,
            "-m",
            "experiments.grouped_vqc_training_modes.operations",
            "worker",
            "--out",
            str(output),
            *arguments,
        ]
    )
    launched_at = time.time()
    subprocess.run(
        [
            str(tmux),
            "new-session",
            "-d",
            "-x",
            "160",
            "-y",
            "40",
            "-s",
            session,
            "-c",
            str(ROOT),
            command,
        ],
        check=True,
    )
    # Worker and monitor exit on completion; discard their panes automatically.
    subprocess.run(
        [str(tmux), "set-window-option", "-t", session, "remain-on-exit", "off"],
        check=True,
    )
    monitor = shlex.join(
        [
            str(SCRIPTS / "status.sh"),
            "--out",
            str(output),
            "--watch",
            "--since",
            str(launched_at),
        ]
    )
    subprocess.run(
        [
            str(tmux),
            "split-window",
            "-v",
            "-p",
            "40",
            "-t",
            session,
            "-c",
            str(ROOT),
            monitor,
        ],
        check=True,
    )
    subprocess.run([str(tmux), "select-pane", "-t", f"{session}:0.0"], check=True)
    table = Table(show_header=False)
    table.add_column("Field", style="cyan")
    table.add_column("Value", overflow="fold")
    rows = {
        "Session": session,
        "Output": str(output),
        "Log": str(output / "train.log"),
        "Attach": shlex.join([str(tmux), "attach", "-t", session]),
        "Status": shlex.join([str(SCRIPTS / "status.sh"), "--out", str(output)]),
        "Follow": shlex.join(["tail", "-f", str(output / "train.log")]),
        "Resume": shlex.join(
            [
                str(SCRIPTS / "launch.sh"),
                "--session",
                session + "_resume",
                "--out",
                str(output),
                "--resume",
            ]
        ),
    }
    for key, value in rows.items():
        table.add_row(key, value)
    console.print(
        Panel(table, title="Started paired training experiment", border_style="green")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("launch", "status", "worker"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--session", default="grouped_modes_l4")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--since", type=float, default=0.0)
    args, remainder = parser.parse_known_args()
    output = args.out.resolve()
    if args.action == "launch":
        launch(output, args.session, remainder)
    elif args.action == "status":
        if remainder:
            parser.error(f"Unexpected arguments: {remainder}")
        show_status(output, args.watch, args.since)
    else:
        raise SystemExit(worker(output, remainder))


if __name__ == "__main__":
    main()
