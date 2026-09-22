"""Run validation shot curves, seal the protocol, then evaluate the final split."""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
from pathlib import Path

from experiments.grouped_dynamic_vqc.runtime import atomic_json, report, utc_now
from experiments.grouped_robot_pilot.protocol import read_json

from .protocol import DEFAULT_OUTPUT, Config, check_output
from .reference import Sources
from .runner import Experiment, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--max-conditions",
        type=int,
        help="Pause after this many NEW conditions; "
        "resume skips verified completed conditions",
    )
    for key, value in Config().to_dict().items():
        if isinstance(value, bool):
            parser.add_argument(
                "--" + key.replace("_", "-"), action="store_true", default=None
            )
        else:
            parser.add_argument(
                "--" + key.replace("_", "-"), type=type(value), default=None
            )
    return parser


def parse_config(args: argparse.Namespace) -> Config:
    values = read_json(args.out / "config.json") if args.resume else Config().to_dict()
    for key in values:
        if getattr(args, key) is not None:
            values[key] = getattr(args, key)
    values["reference"] = str(Path(values["reference"]).resolve())
    config = Config(**values)
    config.validate()
    if args.max_conditions is not None and args.max_conditions < 1:
        raise ValueError("max_conditions must be positive")
    check_output(config, args.out)
    return config


def main() -> None:
    args = build_parser().parse_args()
    config = parse_config(args)
    output = args.out.resolve()
    if args.plan_only:
        sources = Sources(config)
        report(
            f"[cyan]Validation shots -> fixed protocol -> final test[/cyan]\n"
            f"Source: {config.reference}\nSeeds: {config.seed_list()}\n"
            f"Shots: {config.shot_list()}; repetitions: {config.repeats}\n"
            f"CUDA: {sources.runtime['device']}; batch: {config.eval_batch_size}\n"
            f"Training: none; 9 conditions per seed per stage.\n"
            f"Development proxy: {config.development}; output: {output}\n"
            "Plan only: final split is not preprocessed or evaluated."
        )
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".worker.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another evaluation worker owns this output") from error
        atomic_json(
            output / "heartbeat.json",
            {
                "status": "verifying_sources",
                "updated_at": utc_now(),
                "pid": os.getpid(),
                "total_conditions": 18 * len(config.seed_list()),
            },
        )
        shared = None
        previous = {}
        try:
            shared = Experiment(config, output, args.resume)

            def stop(_sig: int, _frame: object) -> None:
                shared.stop_requested = True

            previous = {
                sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)
            }
            run_experiment(shared, args.max_conditions, args.preflight_only)
        except InterruptedError:
            if shared is not None:
                shared.heartbeat("paused")
        except Exception as error:
            atomic_json(
                output / "heartbeat.json",
                {
                    "status": "failed",
                    "updated_at": utc_now(),
                    "pid": os.getpid(),
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            raise
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    main()
