"""Run the paired joint and concept-then-label experiment on CUDA."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
from dataclasses import asdict
from pathlib import Path

from experiments.grouped_dynamic_vqc.runtime import atomic_json, report, utc_now

from .protocol import ROUTES, ExperimentConfig, SharedExperiment
from .results import summarize, verify_complete
from .runner import RouteRun


def run_experiment(shared: SharedExperiment, max_steps: int | None = None) -> dict:
    """A single worker runs both routes; never compete for CUDA memory."""
    for route in ROUTES:
        directory = shared.output / route
        if (directory / "result.json").exists():
            verify_complete(shared, route)
            report(f"[green]Verified completed route:[/green] {route}")
            continue
        runner = RouteRun(shared, route, resume=(directory / "resume.pt").exists())

        previous = {
            signum: signal.signal(signum, runner.request_stop)
            for signum in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            report(
                f"[cyan]Route {route}[/cyan] | {shared.runtime['device']} | "
                f"batch {shared.config.batch_size} | epochs {shared.config.epochs}"
            )
            result = runner.run(max_steps)
        except Exception as error:
            runner.heartbeat("failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        if result["status"] == "paused":
            report(
                f"[yellow]Paused[/yellow] {route}; resume the same experiment directory"
            )
            return result
        if runner.stop_requested:
            runner.heartbeat("paused")
            return {"status": "paused", "route": route}
        del runner
    result = summarize(shared)
    heartbeat = json.loads((shared.output / "heartbeat.json").read_text())
    atomic_json(
        shared.output / "heartbeat.json",
        {**heartbeat, "status": "complete", "updated_at": utc_now()},
    )
    report(
        "[green]Both routes complete and pairing verified.[/green] "
        f"{shared.output / 'summary.md'}"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Pause at this cumulative step in each route; smoke/recovery only",
    )
    for key, value in asdict(ExperimentConfig()).items():
        parser.add_argument(
            "--" + key.replace("_", "-"), type=type(value), default=None
        )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.out.resolve()
    values = (
        json.loads((output / "config.json").read_text())
        if args.resume
        else asdict(ExperimentConfig())
    )
    for key in values:
        if getattr(args, key) is not None:
            values[key] = getattr(args, key)
    config = ExperimentConfig(**values)
    config.validate()
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max_steps must be positive")
    if args.plan_only:
        report(
            f"[cyan]Joint:[/cyan] {config.epochs} epochs; [cyan]Sequential:[/cyan] "
            f"{config.concept_epochs} concept + "
            f"{config.epochs - config.concept_epochs} label epochs\n"
            f"CUDA, L{config.front_layers}, head L{config.label_layers}, "
            f"batch {config.batch_size}, "
            f"Adam {config.learning_rate}, seed {config.seed}\nOutput: {output}"
        )
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".worker.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Another worker owns this experiment directory"
            ) from error
        try:
            # Keep the last good progress available during a resume startup failure.
            previous = (
                json.loads((output / "heartbeat.json").read_text())
                if (output / "heartbeat.json").exists()
                else {}
            )
            atomic_json(
                output / "heartbeat.json",
                {
                    **previous,
                    "updated_at": utc_now(),
                    "pid": os.getpid(),
                    "status": "initializing",
                    "epoch_completed": previous.get("epoch_completed", 0),
                    "epochs_total": config.epochs,
                    "global_step": previous.get("global_step", 0),
                    "offset": previous.get("offset", 0),
                    "device": "CUDA initialization",
                    "route": previous.get("route", "pending"),
                },
            )
            shared = SharedExperiment(config, output, args.resume)
            run_experiment(shared, args.max_steps)
        except Exception as error:
            heartbeat = json.loads((output / "heartbeat.json").read_text())
            atomic_json(
                output / "heartbeat.json",
                {
                    **heartbeat,
                    "status": "failed",
                    "updated_at": utc_now(),
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            raise


if __name__ == "__main__":
    main()
