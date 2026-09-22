"""Run the isolated Robot P0 concept/Independent/zero-feedback validation pilot."""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
from pathlib import Path

from experiments.dynamic_vqc.robot.data import load_tables
from experiments.grouped_dynamic_vqc.runtime import (
    atomic_json,
    cuda_runtime,
    report,
    utc_now,
)

from .protocol import DEFAULT_OUTPUT, Config, check_output, read_json
from .runner import Experiment, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--max-steps", type=int, help="Pause after this many NEW Adam updates"
    )
    parser.add_argument(
        "--max-conditions",
        type=int,
        help="Pause after this many NEW evaluation conditions",
    )
    for key, value in Config().to_dict().items():
        name = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(name, action="store_true", default=None)
        else:
            parser.add_argument(name, type=type(value), default=None)
    return parser


def parse_config(args: argparse.Namespace) -> Config:
    values = read_json(args.out / "config.json") if args.resume else Config().to_dict()
    for key in values:
        if getattr(args, key) is not None:
            values[key] = getattr(args, key)
    values["dataset"] = str(Path(values["dataset"]).resolve())
    config = Config(**values)
    config.validate()
    check_output(config, args.out)
    for limit in (args.max_steps, args.max_conditions):
        if limit is not None and limit < 1:
            raise ValueError("Step and condition limits must be positive")
    return config


def main() -> None:
    args = build_parser().parse_args()
    config = parse_config(args)
    output = args.out.resolve()
    if args.plan_only:
        runtime = cuda_runtime(config.seed)
        groups, _, _ = load_tables(Path(config.dataset))
        report(
            f"[cyan]Robot P0: A5+B5+readout, Fusion L4, 264 parameters[/cyan]\n"
            f"Concept training: {config.concept_epochs} epochs; "
            f"two label routes: {config.head_epochs} each\n"
            f"Train rows: {len(groups['train']['labels'])}; "
            f"validation rows: {len(groups['validation']['labels'])}\n"
            f"Device: {runtime['device']}; batch={config.batch_size}\n"
            f"Eight exact validation conditions; no test evaluation. Output: {output}\n"
            "Plan only: no output files or image features created."
        )
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".worker.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another worker owns this output") from error
        control = {"stop": False}

        def stop(_sig, _frame):
            control["stop"] = True

        previous = {
            sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)
        }
        shared = None
        atomic_json(
            output / "heartbeat.json",
            {"status": "initializing", "updated_at": utc_now(), "pid": os.getpid()},
        )
        try:
            shared = Experiment(config, output, args.resume, control)
            run_experiment(
                shared, args.max_steps, args.max_conditions, args.preflight_only
            )
        except InterruptedError as error:
            if shared is not None:
                shared.heartbeat("paused", reason=str(error))
            else:
                atomic_json(
                    output / "heartbeat.json",
                    {
                        "status": "paused",
                        "updated_at": utc_now(),
                        "pid": os.getpid(),
                        "reason": str(error),
                    },
                )
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
