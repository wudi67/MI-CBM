"""Run two small MLP controls and a frozen learning-rate grid on CUDA."""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
from pathlib import Path

from experiments.grouped_dynamic_vqc.runtime import atomic_json, report, utc_now

from .model import TASKS
from .protocol import Config, Experiment, cell_name, read_json
from .results import summarize
from .runner import CellRun, verify_complete


def run_experiment(experiment: Experiment, max_steps: int | None = None) -> dict:
    for task in TASKS:
        for lr in experiment.config.learning_rates:
            directory = experiment.output / cell_name(task, lr)
            if (directory / "result.json").exists():
                verify_complete(experiment, task, lr)
                report(f"[green]Verified completed cell:[/green] {cell_name(task, lr)}")
                continue
            runner = CellRun(
                experiment, task, lr, resume=(directory / "resume.pt").exists()
            )
            previous = {
                signum: signal.signal(signum, runner.request_stop)
                for signum in (signal.SIGINT, signal.SIGTERM)
            }
            try:
                report(
                    f"[cyan]{runner.name}[/cyan] | {experiment.runtime['device']} | "
                    f"batch {experiment.config.batch_size} | "
                    f"epochs {experiment.epochs_for(task)}"
                )
                result = runner.run(max_steps)
            except Exception as error:
                runner.heartbeat("failed", error=f"{type(error).__name__}: {error}")
                raise
            finally:
                for signum, handler in previous.items():
                    signal.signal(signum, handler)
            if result["status"] == "paused" or runner.stop_requested:
                runner.heartbeat("paused")
                report(f"[yellow]Paused[/yellow] {runner.name}; use --resume")
                return {"status": "paused", "cell": runner.name}
            del runner
    result = summarize(experiment)
    heartbeat = read_json(experiment.output / "heartbeat.json")
    atomic_json(
        experiment.output / "heartbeat.json",
        {**heartbeat, "status": "complete", "updated_at": utc_now()},
    )
    report(f"[green]All MLP cells verified.[/green] {experiment.output / 'summary.md'}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Cumulative step limit per unfinished cell; recovery checks only",
    )
    for key, value in Config().to_dict().items():
        if key == "learning_rates":
            parser.add_argument("--learning-rates", type=float, nargs="+", default=None)
        else:
            parser.add_argument(
                "--" + key.replace("_", "-"), type=type(value), default=None
            )
    args = parser.parse_args()
    output = args.out.resolve()
    values = read_json(output / "config.json") if args.resume else Config().to_dict()
    for key in values:
        if getattr(args, key) is not None:
            values[key] = getattr(args, key)
    values["reference"] = str(Path(values["reference"]).resolve())
    values["learning_rates"] = tuple(values["learning_rates"])
    config = Config(**values)
    config.validate()
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max_steps must be positive")
    if args.plan_only:
        report(
            "[cyan]Concept:[/cyan] 40→3→32, 251 parameters, "
            f"{config.concept_epochs} epochs\n"
            "[cyan]Label:[/cyan] 40→6→1, 253 parameters, "
            f"{config.label_epochs} epochs\n"
            f"Tanh; LR={config.learning_rates}; batch={config.batch_size}; "
            f"seed={config.seed}\n"
            f"Reference: {config.reference}\nOutput: {output}"
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
        previous = (
            read_json(output / "heartbeat.json")
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
                "epochs_total": previous.get("epochs_total", config.concept_epochs),
                "global_step": previous.get("global_step", 0),
                "offset": previous.get("offset", 0),
                "device": "CUDA initialization",
                "total_cells": 2 * len(config.learning_rates),
            },
        )
        try:
            run_experiment(Experiment(config, output, args.resume), args.max_steps)
        except Exception as error:
            heartbeat = read_json(output / "heartbeat.json")
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
