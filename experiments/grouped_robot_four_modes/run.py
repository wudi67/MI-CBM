"""Complete Robot Standard and Joint using fresh paired full-circuit training."""

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
        "--max-steps",
        type=int,
        help="Pause after this many NEW optimizer updates",
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
    check_output(config, args.out)
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("Pause step budget must be positive")
    return config


def main() -> None:
    args = build_parser().parse_args()
    config = parse_config(args)
    output = args.out.resolve()
    if args.plan_only:
        root = Path(config.reference)
        verify_files(root, read_json(root / "result_lock.json")["artifacts"])
        _, _, pilot = config.sources()
        runtime = cuda_runtime(0)
        report(
            "[cyan]Robot Standard / Joint five-seed completion[/cyan]\n"
            f"Seeds: {config.seed_values}; {3 * len(config.seed_values)} new jobs.\n"
            "Standard (BCE), Joint (NLL+BCE), Joint no-feedback (NLL+BCE).\n"
            "Both modules train from the ORIGINAL paired uniform/Gaussian weights.\n"
            "Reuse completed Independent/Sequential predictions read-only.\n"
            f"Epochs={config.epochs}; batch={pilot.batch_size}; "
            f"eval={pilot.eval_batch_size}; lr={pilot.learning_rate}.\n"
            f"Joint loss weights: concept={config.concept_weight}, "
            f"label={config.label_weight}.\n"
            "Match total staged update count, not per-module updates or runtime.\n"
            "Train/validation: four modes, Joint feedback and all-concept correction.\n"
            f"CUDA: {runtime['device']}\nOutput: {output}\n"
            "Plan only: no training or output creation. No test access."
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
            {"status": "initializing", "pid": os.getpid(), "updated_at": utc_now()},
        )
        try:
            shared = Experiment(config, output, args.resume, control)
            run_experiment(shared, args.max_steps, args.preflight_only)
        except InterruptedError as error:
            if shared is not None:
                shared.heartbeat("paused", reason=str(error))
            else:
                atomic_json(
                    output / "heartbeat.json",
                    {
                        "status": "paused",
                        "pid": os.getpid(),
                        "updated_at": utc_now(),
                        "reason": str(error),
                    },
                )
        except Exception as error:
            atomic_json(
                output / "heartbeat.json",
                {
                    "status": "failed",
                    "pid": os.getpid(),
                    "updated_at": utc_now(),
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            raise
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    main()
