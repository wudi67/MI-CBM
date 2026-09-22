"""Resumable validation -> protocol seal -> test, without parameter updates."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from experiments.grouped_dynamic_vqc.runtime import (
    ROOT,
    atomic_checkpoint,
    atomic_json,
    report,
    sha256,
    utc_now,
)
from experiments.grouped_robot_pilot.protocol import read_json
from experiments.grouped_robot_pilot.protocol import verify_files as verify_artifacts
from experiments.grouped_robot_pilot.training import load as load_checkpoint
from experiments.grouped_vqc_training_modes.protocol import state_hash

from .data import data_hashes, read_test, subset, verify_test_gate
from .evaluation import exact_metrics, infer, sample_repeats, sampled_metrics
from .protocol import CONDITIONS, Config, check_output, source_hashes
from .reference import Sources


class Experiment:
    def __init__(self, config: Config, output: Path, resume: bool = False) -> None:
        config.validate()
        check_output(config, output)
        self.config, self.output = config, output.resolve()
        self.stop_requested = False
        self.stage, self.cell = "validation", "initialization"
        self.data: dict = {}
        self.runtime = {"device": "initializing CUDA"}
        self.cached_frontend = None
        self.state_cache = None
        self.output.mkdir(parents=True, exist_ok=True)
        self.sources = Sources(config, self.tick)
        self.runtime = self.sources.runtime
        self.stage = "validation"
        self.cell = "initialization"
        self.data: dict = {}
        self.new_conditions = 0
        self.output.mkdir(parents=True, exist_ok=True)
        reference = {"artifacts": self.sources.hashes}
        self.manifest: dict
        if resume:
            self.manifest = read_json(self.output / "manifest.json")
            if (
                self.manifest["config"] != config.to_dict()
                or self.manifest["sources"] != source_hashes()
            ):
                raise ValueError("Resume requires unchanged config and source hashes")
            if self.manifest["runtime"] != self.runtime:
                raise ValueError("Resume requires original CUDA/software runtime")
            verify_artifacts(self.output, self.manifest["artifacts"])
            if read_json(self.output / "reference_lock.json") != reference:
                raise ValueError("Pinned training or evaluation sources changed")
        else:
            if (self.output / "config.json").exists() or (
                self.output / "manifest.json"
            ).exists():
                raise FileExistsError("Run exists; use --resume or a fresh --out")
            atomic_json(self.output / "config.json", config.to_dict())
            atomic_json(self.output / "reference_lock.json", reference)
            self.manifest = {
                "schema": "grouped_robot_shots_final.v1",
                "created_at": utc_now(),
                "config": config.to_dict(),
                "sources": source_hashes(),
                "runtime": self.runtime,
                "artifacts": {
                    n: sha256(self.output / n)
                    for n in ("config.json", "reference_lock.json")
                },
                "stages": ["validation", self.final_role],
                "conditions": [list(value) for value in CONDITIONS],
                "selection": (
                    "fixed final checkpoints; "
                    "no result-dependent model or seed selection"
                ),
                "training": (
                    "none; frozen Fusion L4 plus B5/readout L5 (352 parameters)"
                ),
                "evaluation": (
                    "all 32 physical branches; own measured/corrected/zero X controls"
                ),
                "sampling": (
                    "joint (m,y); nested shot prefixes; independent repeated draws"
                ),
                "label_decision": "probability >= 0.5; ties predict 1",
                "concept_decision": (
                    "five binary marginals >= 0.5; joint MAP uses all 32 legal codes; "
                    "argmax ties use lowest index"
                ),
                "concept_reporting": (
                    "original frontend predictions; Standard diagnostic only; "
                    "Independent/Sequential share the "
                    "Independent/measured concept report"
                ),
                "replication_unit": (
                    "training seed; average Monte Carlo repetitions within each seed"
                ),
                "engineering_subset": config.development,
            }
            atomic_json(self.output / "manifest.json", self.manifest)
        self.manifest_hash = sha256(self.output / "manifest.json")
        self.verify_report_lock(self.output)

    @property
    def final_role(self) -> str:
        return "test_proxy" if self.config.development else "test"

    def heartbeat(self, status: str, **details: object) -> None:
        atomic_json(
            self.output / "heartbeat.json",
            {
                "status": status,
                "updated_at": utc_now(),
                "pid": os.getpid(),
                "stage": self.stage,
                "cell": self.cell,
                "device": self.runtime["device"],
                "completed_conditions": len(
                    list(self.output.glob("*/seed*/*/*/evaluation_lock.json"))
                ),
                "total_conditions": 2 * len(CONDITIONS) * len(self.config.seed_list()),
                "stage_samples": len(self.data.get("labels", [])),
                "test_read": (self.output / "test/test_inputs.json").exists(),
                "test_evaluated": (self.output / "test/result_lock.json").exists(),
                **details,
            },
        )

    def tick(self, status: str, **details: object) -> None:
        self.heartbeat(status, **details)
        if self.stop_requested:
            raise InterruptedError(
                "Pause requested; completed conditions are preserved"
            )

    def verify_report_lock(self, root: Path) -> dict | None:
        path = root / "result_lock.json"
        if not path.exists():
            return None
        lock = read_json(path)
        if lock["manifest_sha256"] != self.manifest_hash:
            raise ValueError("Report manifest mismatch")
        verify_artifacts(root, lock["artifacts"])
        return lock

    def prepare_data(self) -> None:
        if self.stage == "test":
            verify_test_gate(self.output, self.manifest_hash)
        root = self.output / self.stage
        root.mkdir(parents=True, exist_ok=True)
        path = root / "data_lock.json"
        dependency = (
            self.manifest_hash
            if self.stage == "validation"
            else sha256(self.output / "protocol_lock.json")
        )
        expected = {
            "manifest_sha256": self.manifest_hash,
            "role": self.stage,
            "dependency_sha256": dependency,
        }
        if path.exists():
            lock = read_json(path)
            if any(lock[k] != v for k, v in expected.items()):
                raise ValueError("Data cache provenance mismatch")
            verify_artifacts(root, lock["artifacts"])
            if self.stage == "test":
                verify_artifacts(
                    Path("/"), read_json(root / "test_inputs.json")["artifacts"]
                )
            self.data = load_checkpoint(root / "data.pt")
            if data_hashes(self.data) != lock["data_hashes"]:
                raise ValueError("Cached data content changed")
            return
        self.tick("preprocessing")
        if self.stage == "validation":
            self.data = subset(self.sources.data["validation"], self.config.val_limit)
        elif self.stage == "test_proxy":
            self.data = subset(
                self.sources.data["validation"], self.config.val_limit, seed=1
            )
        else:
            self.data, inputs = read_test(
                self.sources, self.output, self.manifest_hash, self.tick
            )
            atomic_json(root / "test_inputs.json", inputs)
        if not self.config.development and len(self.data["labels"]) != 6144:
            raise ValueError("Formal Robot evaluation requires the complete split")
        atomic_checkpoint(root / "data.pt", self.data)
        atomic_json(
            path,
            {
                **expected,
                "data_hashes": data_hashes(self.data),
                "n_samples": len(self.data["labels"]),
                "preprocessing_sha256": sha256(self.sources.preprocessing_path),
                "routing": "reuse frozen training permutation; no fitting",
                "artifacts": {
                    "data.pt": sha256(root / "data.pt"),
                    **(
                        {"test_inputs.json": sha256(root / "test_inputs.json")}
                        if self.stage == "test"
                        else {}
                    ),
                },
            },
        )

    def directory(self, seed: int, training: str, mode: str) -> Path:
        return self.output / self.stage / f"seed{seed}" / training / mode

    def expected(self, seed: int, training: str, mode: str) -> dict:
        return {
            "manifest_sha256": self.manifest_hash,
            "data_lock_sha256": sha256(self.output / self.stage / "data_lock.json"),
            "checkpoint_sha256": sha256(self.sources.model_path(seed, training)),
            "role": self.stage,
            "seed": seed,
            "training": training,
            "control_mode": mode,
        }

    def verify_condition(self, seed: int, training: str, mode: str) -> dict | None:
        root = self.directory(seed, training, mode)
        path = root / "evaluation_lock.json"
        if not path.exists():
            return None
        lock = read_json(path)
        if any(lock[k] != v for k, v in self.expected(seed, training, mode).items()):
            raise ValueError("Condition provenance mismatch")
        verify_artifacts(root, lock["artifacts"])
        raw, samples = (
            load_checkpoint(root / "joint.pt"),
            load_checkpoint(root / "samples.pt"),
        )
        metrics = read_json(root / "evaluation.json")
        if (
            samples["budgets"] != self.config.shot_list()
            or len(samples["seeds"]) != self.config.repeats
        ):
            raise ValueError("Cached sampling budget changed")
        for key in ("labels", "concepts", "source_index", "robot_ids"):
            if not torch.equal(raw[key], self.data[key]):
                raise ValueError("Cached condition sample identities changed")
        if metrics["exact"] != exact_metrics(raw) or metrics[
            "repeats"
        ] != sampled_metrics(raw, samples):
            raise ValueError(
                "Persisted metrics do not reproduce from saved observations"
            )
        return metrics

    def compare_validation(
        self, raw: dict, seed: int, training: str, mode: str
    ) -> None:
        if self.stage == "test":
            return
        old = load_checkpoint(self.sources.reference_predictions(seed, training, mode))
        position = {int(index): i for i, index in enumerate(old["source_index"])}
        indices = [position[int(index)] for index in raw["source_index"]]
        for key in ("labels", "concepts", "robot_ids"):
            if not torch.equal(raw[key], old[key][indices]):
                raise ValueError("Reference evaluation targets differ")
        for actual, previous in (
            (raw["concept_probabilities"], old["concept_probabilities"][indices]),
            (raw["branch_label_mass"], old["branch_label_mass"][indices]),
        ):
            if (actual - previous).abs().max() > 2e-6:
                raise ValueError(
                    "CUDA inference differs from existing validation probabilities"
                )

    @torch.no_grad()
    def states(self, seed: int, training: str) -> torch.Tensor:
        weights = load_checkpoint(self.sources.model_path(seed, training))["model"]
        model = self.sources.make_model(weights)
        fingerprint = state_hash(model.frontend.state_dict())
        if fingerprint == self.cached_frontend and self.state_cache is not None:
            return self.state_cache
        chunks = []
        batch = self.config.eval_batch_size
        for start in range(0, len(self.data["angles"]), batch):
            chunks.append(
                model.frontend(self.data["angles"][start : start + batch].cuda())
            )
            self.tick(
                "caching_frontend", offset=min(start + batch, len(self.data["angles"]))
            )
        self.cached_frontend = fingerprint
        self.state_cache = torch.cat(chunks)
        return self.state_cache

    @torch.no_grad()
    def evaluate_condition(
        self, seed: int, training: str, mode: str, states: torch.Tensor
    ) -> None:
        root = self.directory(seed, training, mode)
        root.mkdir(parents=True, exist_ok=True)
        expected = self.expected(seed, training, mode)
        if (root / "joint_lock.json").exists():
            lock = read_json(root / "joint_lock.json")
            if any(lock[k] != v for k, v in expected.items()):
                raise ValueError("Joint cache provenance mismatch")
            verify_artifacts(root, lock["artifacts"])
            raw = load_checkpoint(root / "joint.pt")
        else:
            weights = load_checkpoint(self.sources.model_path(seed, training))["model"]
            model = self.sources.make_model(weights)
            before = state_hash(model.state_dict())
            if state_hash(model.frontend.state_dict()) != self.cached_frontend:
                raise ValueError("Cached state belongs to another frontend")
            raw = infer(
                model, self.data, states, mode, self.config.eval_batch_size, self.tick
            )
            if state_hash(model.state_dict()) != before:
                raise ValueError("Frozen model changed during evaluation")
            self.compare_validation(raw, seed, training, mode)
            atomic_checkpoint(root / "joint.pt", raw)
            atomic_json(
                root / "joint_lock.json",
                {**expected, "artifacts": {"joint.pt": sha256(root / "joint.pt")}},
            )
        samples = sample_repeats(
            raw, self.config, (self.stage, seed, training, mode), self.tick
        )
        metrics = {
            **expected,
            "n_samples": len(self.data["labels"]),
            "test_evaluated": self.stage == "test",
            "engineering_subset": self.config.development,
            "concept_supervised": training != "standard",
            "exact": exact_metrics(raw),
            "repeats": sampled_metrics(raw, samples),
        }
        atomic_checkpoint(root / "samples.pt", samples)
        atomic_json(root / "evaluation.json", metrics)
        atomic_json(
            root / "evaluation_lock.json",
            {
                **expected,
                "artifacts": {
                    name: sha256(root / name)
                    for name in (
                        "joint.pt",
                        "joint_lock.json",
                        "samples.pt",
                        "evaluation.json",
                    )
                },
            },
        )
        self.new_conditions += 1
        report(
            f"[green]{self.stage}/{self.cell} complete[/green] "
            f"exact label={metrics['exact']['label']['accuracy']:.2%}"
        )

    def seal_validation(self) -> None:
        root = self.output / "validation"
        lock = self.verify_report_lock(root)
        if lock is None or read_json(root / "summary.json")["status"] != "complete":
            raise ValueError(
                "Cannot enter final evaluation before validation is complete"
            )
        self.sources.verify_unchanged()
        verify_artifacts(ROOT, self.manifest["sources"])
        value = {
            "manifest_sha256": self.manifest_hash,
            "validation_result_lock_sha256": sha256(root / "result_lock.json"),
            "development": self.config.development,
            "config": self.config.to_dict(),
            "transition": (
                "all planned validation cells verified; no accuracy-dependent gate"
            ),
        }
        path = self.output / "protocol_lock.json"
        if path.exists():
            if read_json(path) != value:
                raise ValueError("Locked validation protocol changed")
        else:
            atomic_json(path, value)
        access = self.output / "test_access.json"
        expected = {"protocol_sha256": sha256(path), "role": self.final_role}
        if access.exists():
            if any(read_json(access)[k] != v for k, v in expected.items()):
                raise ValueError("Test access protocol changed")
        else:
            atomic_json(
                access,
                {
                    **expected,
                    "started_at": utc_now(),
                    "real_test_values_requested": not self.config.development,
                },
            )


def run_experiment(
    shared: Experiment, max_conditions: int | None = None, preflight_only: bool = False
) -> None:
    from .results import (  # pylint: disable=import-outside-toplevel
        finish_pipeline,
        summarize_stage,
    )

    if preflight_only:
        shared.heartbeat(
            "paused", reason="Source preflight complete; resume to evaluate"
        )
        return
    for role in ("validation", shared.final_role):
        shared.stage = role
        if role != "validation":
            shared.seal_validation()
        shared.prepare_data()
        stage_root = shared.output / role
        existing = shared.verify_report_lock(stage_root)
        evaluations = []
        for seed in shared.config.seed_list():
            states = None
            for training, mode, _mask in CONDITIONS:
                shared.cell = f"seed{seed}/{training}/{mode}"
                shared.tick("verifying_condition")
                metric = shared.verify_condition(seed, training, mode)
                if metric is None:
                    if existing is not None:
                        raise ValueError("Completed stage is missing a condition")
                    if (
                        max_conditions is not None
                        and shared.new_conditions >= max_conditions
                    ):
                        raise InterruptedError("Configured condition limit reached")
                    states = shared.states(seed, training)
                    shared.evaluate_condition(seed, training, mode, states)
                    metric = shared.verify_condition(seed, training, mode)
                if metric is None:
                    raise RuntimeError("Condition was not committed after evaluation")
                evaluations.append(metric)
            del states
            shared.cached_frontend, shared.state_cache = None, None
        if existing is None:
            shared.tick("writing_reports")
            summarize_stage(shared, evaluations)
        if (
            max_conditions is not None
            and shared.new_conditions >= max_conditions
            and role == "validation"
        ):
            raise InterruptedError(
                "Paused after validation; final split remains unopened"
            )
    shared.sources.verify_unchanged()
    verify_artifacts(ROOT, shared.manifest["sources"])
    finish_pipeline(shared)
    shared.heartbeat("complete")
