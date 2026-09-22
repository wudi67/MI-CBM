"""One CUDA worker: Standard, frozen heads, then unified interventions."""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
from pathlib import Path

from experiments.grouped_dynamic_vqc.runtime import atomic_json, report, utc_now

from .protocol import DEFAULT_OUTPUT, Config, Experiment, read_json
from .results import evaluate_models, summarize
from .runner import RouteRun, verify_complete


def run_experiment(shared: Experiment, max_steps: int | None = None) -> dict:
    for route in shared.routes:
        directory = shared.output / route
        if (directory / "result.json").exists():
            verify_complete(shared, route)
            report(f"[green]Verified completed route:[/green] {route}")
            continue
        runner = RouteRun(shared, route, resume=(directory / "resume.pt").exists())
        previous = {
            sig: signal.signal(sig, runner.request_stop)
            for sig in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            report(
                f"[cyan]{route}[/cyan] | {runner.epochs} epochs | "
                f"CUDA {shared.runtime['device']}"
            )
            result = runner.run(max_steps)
        except Exception as error:
            runner.heartbeat("failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        if result["status"] == "paused" or runner.stop_requested:
            runner.heartbeat("paused")
            return {"status": "paused", "route": route}
        del runner
    stopped = False

    def request_stop(_sig: int, _frame: object) -> None:
        nonlocal stopped
        stopped = True

    previous = {
        sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        evaluate_models(shared, lambda: stopped)
        if stopped:
            raise InterruptedError("Paused before summary")
        result = summarize(shared)
    except InterruptedError:
        heartbeat = read_json(shared.output / "heartbeat.json")
        atomic_json(
            shared.output / "heartbeat.json",
            {**heartbeat, "status": "paused", "updated_at": utc_now()},
        )
        return {"status": "paused"}
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    heartbeat = read_json(shared.output / "heartbeat.json")
    atomic_json(
        shared.output / "heartbeat.json",
        {**heartbeat, "status": "complete", "updated_at": utc_now()},
    )
    report(
        "[green]All training and diagnostics complete.[/green] "
        f"{shared.output / 'summary.md'}"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Pause at cumulative optimizer step per unfinished route",
    )
    for key, value in Config().to_dict().items():
        parser.add_argument(
            "--" + key.replace("_", "-"), type=type(value), default=None
        )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = args.out.resolve()
    values = read_json(output / "config.json") if args.resume else Config().to_dict()
    for key in values:
        if getattr(args, key) is not None:
            values[key] = getattr(args, key)
    config = Config(**values)
    config.validate()
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("max_steps must be positive")
    if args.plan_only:
        report(
            f"[cyan]L4 Standard:[/cyan] {config.epochs} epochs, Label BCE only\n"
            "[cyan]Frozen frontend heads:[/cyan] true / zero, "
            f"{config.head_epochs} epochs each\n"
            "Measured head reused with matching training budget; "
            "otherwise retrained.\n"
            "Six controls, exact random average, "
            "18-group diagnostics and finite shots.\n"
            f"CUDA, Adam {config.learning_rate}, batch {config.batch_size}, "
            f"seed {config.seed}\n"
            f"Reference: {config.reference}\nOutput: {output}"
        )
        return
    reference = Path(config.reference).resolve()
    if output == reference or output.is_relative_to(reference):
        raise ValueError("Use an output directory outside the immutable reference run")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".worker.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another worker owns this output directory") from error
        prior = (
            read_json(output / "heartbeat.json")
            if (output / "heartbeat.json").exists()
            else {}
        )
        atomic_json(
            output / "heartbeat.json",
            {
                **prior,
                "updated_at": utc_now(),
                "pid": os.getpid(),
                "status": "initializing",
                "cell": prior.get("cell", "pending"),
                "epoch_completed": prior.get("epoch_completed", 0),
                "epochs_total": config.epochs,
                "global_step": prior.get("global_step", 0),
                "offset": prior.get("offset", 0),
                "device": "CUDA initialization",
            },
        )
        try:
            shared = Experiment(config, output, args.resume)
            run_experiment(shared, args.max_steps)
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
