"""Read-only training audits and immutable provenance for a shared zero baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.grouped_dynamic_vqc.runtime import ROOT, sha256
from experiments.grouped_feedback_ablation.protocol import (
    cell_name,
    load_checkpoint,
    read_json,
)
from experiments.grouped_feedback_ablation.runner import verify_complete as verify_zero
from experiments.grouped_independent_intervention.artifacts import verify_complete
from experiments.grouped_independent_intervention.evaluation import verify_condition
from experiments.grouped_independent_intervention.protocol import (
    DEFAULT_OUTPUT as DEFAULT_REFERENCE,
)
from experiments.grouped_independent_intervention.protocol import (
    Config as IndependentConfig,
)
from experiments.grouped_independent_intervention.protocol import (
    Experiment as IndependentExperiment,
)
from experiments.grouped_independent_intervention.protocol import (
    source_hashes as upstream_hashes,
)
from experiments.grouped_sequential_intervention.protocol import verify_artifacts

PACKAGE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = (
    ROOT / "outputs/grouped_independent_feedback_ablation/dsprites_l4_paired"
)
VARIANTS = ("feedback", "no_feedback")
CONTROLS = {"feedback": "measured", "no_feedback": "zero"}


@dataclass(frozen=True)
class Config:
    reference: str = str(DEFAULT_REFERENCE)
    seeds: str = "0,1,2,3,4"
    eval_batch_size: int = 2048
    shots: int = 256
    val_limit: int = 0
    development: bool = False
    recompute: bool = False

    def seed_list(self) -> list[int]:
        return [int(value) for value in self.seeds.split(",")]

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        seeds = self.seed_list()
        if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
            raise ValueError("Provide distinct nonnegative training seeds")
        if min(self.shots, self.eval_batch_size) < 1:
            raise ValueError("Shots and batch size must be positive")
        if self.val_limit < 0 or 0 < self.val_limit < 18:
            raise ValueError("val_limit must be zero or at least 18")
        if self.val_limit and not self.development:
            raise ValueError("Validation subsets require --development")


def check_output(config: Config, output: Path) -> None:
    """Reject reference ancestors/descendants, including indirect historical roots."""
    pending, seen = [Path(config.reference).resolve()], set()
    output = output.resolve()
    while pending:
        root = pending.pop()
        if root in seen:
            continue
        seen.add(root)
        if output == root or output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError("Output must be isolated from all source experiments")
        if (root / "config.json").exists():
            values = read_json(root / "config.json")
            pending.extend(
                Path(value).resolve()
                for key, value in values.items()
                if (key == "reference" or key.endswith("_reference")) and value
            )


def source_hashes() -> dict:
    paths = [*PACKAGE.rglob("*.py"), *(PACKAGE / "scripts").glob("*.sh")]
    return {**upstream_hashes(), **{str(p.relative_to(ROOT)): sha256(p) for p in paths}}


class Sources:
    """Open completed sources without their writers, trainers or heartbeat methods."""

    def __init__(self, config: Config) -> None:
        self.config = config
        root = Path(config.reference).resolve()
        old = IndependentConfig(**read_json(root / "config.json"))
        if not set(config.seed_list()).issubset(old.seed_list()):
            raise ValueError("Requested seeds absent from Independent source")
        self.independent = IndependentExperiment(old, root, resume=True)
        self.baseline = self.independent.source
        self.runtime = self.independent.runtime
        self.independent.verify_sources()
        if (
            not self.independent.paired_training
            or self.independent.manifest["engineering_subset"]
            or old.val_limit
            or self.baseline.config.label_weight != 1.0
            or self.independent.manifest["data_indices"]
            != self.baseline.manifest["data_indices"]
        ):
            raise ValueError(
                "Expected full, matched Independent and no-feedback training"
            )
        self.hashes = read_json(root / "reference_lock.json")["artifacts"].copy()
        self.hashes.update(self.baseline.reference["artifacts"])
        self.pairing = {}
        for source in (self.independent, self.baseline):
            self.pin(source.output, ["manifest.json", "config.json"])
            self.pin(source.output, source.manifest["artifacts"])
        for seed in config.seed_list():
            self.audit_pair(seed)

    def pin(self, root: Path, names) -> None:
        self.hashes.update({str(root / name): sha256(root / name) for name in names})

    def model_path(self, seed: int, variant: str) -> Path:
        if variant == "feedback":
            return self.independent.model_path("independent", seed)
        return (
            self.baseline.output
            / cell_name("sequential", seed, variant)
            / "endpoint.pt"
        )

    def evaluation_path(self, seed: int, variant: str) -> Path:
        path = self.model_path(seed, variant).parent
        return path / "evaluation/measured" if variant == "feedback" else path

    def audit_pair(self, seed: int) -> None:
        a = verify_complete(self.independent, seed)
        b = verify_zero(self.baseline, cell_name("sequential", seed, "no_feedback"))
        for key in ("initial_model_sha256", "frontend_sha256", "global_step"):
            if a[key] != b[key]:
                raise ValueError(f"Training pairing mismatch: seed {seed}, {key}")
        if a["spec"]["control_mode"] != "both" or b["spec"]["control_mode"] != "zero":
            raise ValueError(
                "Expected true-record training versus separately trained zero"
            )
        histories = []
        for variant, result in zip(VARIANTS, (a, b), strict=True):
            directory = self.model_path(seed, variant).parent
            self.pin(directory, ["result.json", *result["artifacts"]])
            histories.append(read_json(directory / "history.json"))
            checkpoint = load_checkpoint(directory / "endpoint.pt")
            progress, optimizer = checkpoint["progress"], checkpoint["optimizer"]
            if (
                progress["history"] != histories[-1]
                or progress["global_step"] != result["global_step"]
                or progress["completed_epoch"] != result["spec"]["epochs"]
                or progress["order"] is not None
                or progress["offset"] != 0
                or len(optimizer["state"]) != 5
                or any(
                    int(s["step"]) != result["global_step"]
                    for s in optimizer["state"].values()
                )
            ):
                raise ValueError("Completed optimizer/history budget mismatch")
            path = self.evaluation_path(seed, variant)
            lock = read_json(path / "evaluation_lock.json")
            if lock["checkpoint_sha256"] != sha256(directory / "endpoint.pt"):
                raise ValueError("Evaluation checkpoint mismatch")
            if variant == "feedback":
                metrics = verify_condition(
                    self.independent, "independent", seed, "measured"
                )
            else:
                if lock["manifest_sha256"] != self.baseline.manifest_hash or lock[
                    "result_sha256"
                ] != sha256(directory / "result.json"):
                    raise ValueError("No-feedback evaluation provenance mismatch")
                verify_artifacts(path, lock["artifacts"])
                metrics = read_json(path / "evaluation.json")
            if (
                metrics is None
                or metrics["control_mode"] != CONTROLS[variant]
                or metrics["test_evaluated"]
            ):
                raise ValueError("Expected unassisted validation evaluation")
            self.pin(path, ["evaluation_lock.json", *lock["artifacts"]])
        orders = [[row["order_sha256"] for row in h] for h in histories]
        if orders[0] != orders[1]:
            raise ValueError("Paired sample orders differ")
        self.pairing[str(seed)] = {
            "initialization_equal": True,
            "frontend_equal": True,
            "sample_orders_equal": True,
            "optimizer_steps_equal": True,
            "global_step": a["global_step"],
            "initial_model_sha256": a["initial_model_sha256"],
            "frontend_sha256": a["frontend_sha256"],
            "checkpoints": {v: str(self.model_path(seed, v)) for v in VARIANTS},
        }

    def can_reuse(self) -> bool:
        return (
            not self.config.recompute
            and not self.config.val_limit
            and all(
                self.config.shots == source.config.shots
                and self.config.eval_batch_size == source.config.eval_batch_size
                for source in (self.independent, self.baseline)
            )
        )

    def verify(self) -> None:
        verify_artifacts(Path("/"), self.hashes)
