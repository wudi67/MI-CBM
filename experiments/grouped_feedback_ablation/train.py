"""Run a Joint seed-0 pilot, then paired Sequential repeats, on one CUDA worker."""

from __future__ import annotations

import argparse
import fcntl
import signal
from pathlib import Path

from experiments.grouped_dynamic_vqc.runtime import report

from .protocol import (
    DEFAULT_OUTPUT,
    Config,
    Experiment,
    cell_spec,
    cells,
    check_output,
    read_json,
)
from .results import evaluate_cell, summarize
from .runner import RouteRun, import_historical, verify_complete


def run_experiment(
    shared: Experiment, stage: str = "all", max_steps: int | None = None
) -> dict:
    for cell in cells(shared.config, stage):
        if shared.stop_requested:
            raise InterruptedError("Paused between cells")
        path = shared.output / cell
        if (path / "result.json").exists():
            verify_complete(shared, cell)
            report(f"[green]Verified completed cell:[/green] {cell}")
        elif cell in shared.manifest["reuse"]:
            import_historical(shared, cell)
        else:
            runner = RouteRun(shared, cell)
            result = runner.run(max_steps)
            del runner
            if result["status"] == "paused":
                return result
        if cell_spec(shared.config, cell)["phase"] != "concept":
            evaluate_cell(shared, cell)
        if cell.endswith("/no_feedback"):
            summarize(shared)
    result = summarize(shared)
    shared.heartbeat(
        "complete",
        cell=f"requested stage: {stage}",
        suite_complete=result["status"] == "complete",
    )
    report(
        f"[green]Requested stage complete:[/green] {stage}\n"
        f"{shared.output / 'summary.md'}"
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--stage", choices=("all", "joint", "sequential"), default="all"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Pause at cumulative steps in the first unfinished cell",
    )
    for key, value in Config().to_dict().items():
        option = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            parser.add_argument(option, action="store_true", default=None)
        else:
            parser.add_argument(option, type=type(value), default=None)
    return parser


def show_plan(config: Config, stage: str, output: Path) -> None:
    report(
        f"[cyan]Stage:[/cyan] {stage}; Joint seed {config.joint_seed}, "
        f"Sequential seeds {config.seed_list()}\n"
        f"Joint: concept NLL + label BCE, {config.joint_epochs} epochs per variant.\n"
        f"Sequential: new concept frontend per seed, {config.concept_epochs} epochs; "
        f"freeze, then {config.head_epochs} epochs per variant.\n"
        "Variants: measured-record X feedback / zero feedback, separately trained.\n"
        "L4 Fusion data re-uploading; 240 frontend + 24 classification parameters.\n"
        f"CUDA required; Adam {config.learning_rate}, batch {config.batch_size}; "
        f"{config.shots} joint (m,y) shots per image, each variant's own controls.\n"
        "Reuse compatible verified historical seed 0: "
        f"{not config.retrain_reference}.\n"
        "Fixed data/split, initialization and sample order within each pair.\n"
        "Fixed final endpoints; validation only; no automatic Joint selection.\n"
        f"Scheduled cells: {len(cells(config, stage))}\nOutput: {output}"
    )


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
        show_plan(config, args.stage, output)
        return
    check_output(config, output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".worker.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another worker owns this output directory") from error
        shared = Experiment(config, output, args.resume)
        shared.heartbeat("initializing")

        def stop(_sig: int, _frame: object) -> None:
            shared.stop_requested = True

        previous = {
            sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)
        }
        try:
            run_experiment(shared, args.stage, args.max_steps)
        except InterruptedError:
            shared.heartbeat("paused")
        except Exception as error:
            shared.heartbeat("failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    main()
