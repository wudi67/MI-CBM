"""Train Independent classification circuits, then compare four interventions."""

from __future__ import annotations

import argparse
import fcntl
import os
import signal
from pathlib import Path

from experiments.grouped_dynamic_vqc.runtime import atomic_json, report, utc_now
from experiments.grouped_feedback_ablation.protocol import read_json
from experiments.grouped_sequential_intervention.protocol import MODES

from .artifacts import import_historical, verify_complete
from .evaluation import directory, evaluate_model
from .protocol import DEFAULT_OUTPUT, Config, Experiment, check_output
from .results import summarize
from .trainer import IndependentRun


def run_experiment(
    shared: Experiment, max_steps: int | None = None, max_seeds: int | None = None
) -> dict:
    newly_completed = 0
    for seed in shared.config.seed_list():
        if shared.stop_requested:
            raise InterruptedError("Paused between seeds")
        was_complete = all(
            (directory(shared, training, seed, mode) / "evaluation_lock.json").exists()
            for training in ("independent", "sequential")
            for mode in MODES
        )
        path = shared.output / f"independent/seed{seed}/result.json"
        if path.exists():
            verify_complete(shared, seed)
        elif shared.true_reference["reuse_seed"] == seed:
            import_historical(shared, seed)
            report(
                f"[green]Independent seed {seed}: 已复用核验后的历史最终模型[/green]"
            )
        else:
            run = IndependentRun(shared, seed)
            result = run.run(max_steps)
            del run
            if result["status"] == "paused":
                summarize(shared)
                return result
            verify_complete(shared, seed)
        evaluate_model(shared, "sequential", seed)
        evaluate_model(shared, "independent", seed)
        result = summarize(shared)
        if not was_complete:
            newly_completed += 1
        if max_seeds is not None and newly_completed >= max_seeds:
            shared.verify_sources()
            shared.heartbeat("complete" if result["status"] == "complete" else "paused")
            return result
    shared.verify_sources()
    result = summarize(shared)
    shared.heartbeat("complete", cell="all training and evaluation complete")
    report(f"[green]全部完成：[/green]{shared.output / 'summary.md'}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Pause at cumulative updates in first unfinished training cell",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        help="Pause after this many newly completed seed comparisons",
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
    config = Config(**values)
    config.validate()
    for value in (args.max_steps, args.max_seeds):
        if value is not None and value < 1:
            raise ValueError("Pause limits must be positive")
    check_output(config, args.out)
    return config


def main() -> None:
    args = build_parser().parse_args()
    config = parse_config(args)
    output = args.out.resolve()
    if args.plan_only:
        report(
            f"[cyan]Independent 五种子训练与概念干预[/cyan]\n"
            f"Seeds: {config.seed_list()}; frozen Fusion L4 frontends.\n"
            "Train only 24 quantum classification parameters "
            "using true Shape/Scale controls.\n"
            f"Fresh Adam {config.learning_rate}; batch {config.batch_size}; "
            f"{config.head_epochs} epochs.\n"
            "Evaluate Sequential/Independent, four record conditions, "
            f"{config.shots} joint shots.\n"
            "Preserve actual branches and retained B states; no test evaluation.\n"
            "Reuse compatible historical Independent seed 0 "
            "and Sequential evaluations after verification.\n"
            f"Development: {config.development}; output: {output}"
        )
        return
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".worker.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another worker owns this output directory") from error
        atomic_json(
            output / "heartbeat.json",
            {
                "status": "verifying_sources",
                "pid": os.getpid(),
                "updated_at": utc_now(),
                "epoch_completed": 0,
                "epochs_total": config.head_epochs,
                "global_step": 0,
                "offset": 0,
                "device": "initializing CUDA",
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
            run_experiment(shared, args.max_steps, args.max_seeds)
        except InterruptedError:
            if shared is not None:
                shared.heartbeat("paused")
        except Exception as error:
            atomic_json(
                output / "heartbeat.json",
                {
                    "status": "failed",
                    "pid": os.getpid(),
                    "updated_at": utc_now(),
                    "epoch_completed": 0,
                    "epochs_total": config.head_epochs,
                    "global_step": 0,
                    "offset": 0,
                    "device": "CUDA worker",
                    "error": f"{type(error).__name__}: {error}",
                },
            )
            raise
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    main()
