"""CLI for two isolated duration-only Robot continuations."""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
from pathlib import Path

from experiments.grouped_dynamic_vqc.runtime import (
    atomic_json,
    cuda_runtime,
    report,
    utc_now,
)
from experiments.grouped_robot_pilot.protocol import Config as PilotConfig
from experiments.grouped_robot_pilot.protocol import read_json, verify_files

from .protocol import DEFAULT_OUTPUT, Config, check_output
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
    values["reference"] = str(Path(values["reference"]).resolve())
    config = Config(**values)
    source = PilotConfig(**read_json(Path(config.reference) / "config.json"))
    config.validate(source)
    check_output(config, args.out, source)
    for budget in (args.max_steps, args.max_conditions):
        if budget is not None and budget < 1:
            raise ValueError("Pause budgets must be positive")
    return config


def main() -> None:
    args = build_parser().parse_args()
    config = parse_config(args)
    output = args.out.resolve()
    if args.plan_only:
        root = Path(config.reference)
        original = PilotConfig(**read_json(root / "config.json"))
        runtime = cuda_runtime(original.seed)
        verify_files(root, read_json(root / "result_lock.json")["artifacts"])
        report(
            f"[cyan]Robot duration-only continuation[/cyan]\n"
            f"Reference: {root}\n"
            f"Long concept: {original.concept_epochs}->{config.concept_epochs}, "
            f"then two freshly initialized {original.head_epochs}-epoch "
            f"label circuits.\n"
            f"Long label: original concept endpoint, each original label circuit "
            f"{original.head_epochs}->{config.head_epochs} with full Adam.\n"
            f"Five training cells; 24 validation conditions "
            f"plus 9 matched training diagnostics.\n"
            f"CUDA: {runtime['device']}; batch={original.batch_size}; "
            f"learning rate={original.learning_rate}. No test evaluation.\n"
            f"Output: {output}\nPlan only: no output files or training."
        )
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".worker.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Another worker owns this continuation output"
            ) from error
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
