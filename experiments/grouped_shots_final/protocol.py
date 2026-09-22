"""Fixed evaluation protocol and read-only, paired training sources."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.grouped_dynamic_vqc.runtime import ROOT, sha256
from experiments.grouped_feedback_ablation.protocol import cell_name, read_json
from experiments.grouped_feedback_ablation.runner import verify_complete
from experiments.grouped_independent_feedback_ablation.protocol import (
    Config as SourceConfig,
)
from experiments.grouped_independent_feedback_ablation.protocol import (
    Sources as PairedSources,
)
from experiments.grouped_independent_feedback_ablation.protocol import (
    check_output as check_source_output,
)
from experiments.grouped_independent_feedback_ablation.protocol import (
    source_hashes as upstream_hashes,
)
from experiments.grouped_sequential_intervention.protocol import verify_artifacts

PACKAGE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs/grouped_shots_final/dsprites_l4_five_seeds"
MODES = ("measured", "shape", "scale", "both")
CONDITIONS = tuple(
    (training, mode) for training in ("independent", "sequential") for mode in MODES
) + (("no_feedback", "zero"),)


@dataclass(frozen=True)
class Config:
    reference: str = SourceConfig().reference
    seeds: str = "0,1,2,3,4"
    shots: str = "64,128,256,512,1024"
    repeats: int = 10
    sampling_seed: int = 20270914
    eval_batch_size: int = 2048
    val_limit: int = 0
    development: bool = False

    def seed_list(self) -> list[int]:
        return [int(x) for x in self.seeds.split(",")]

    def shot_list(self) -> list[int]:
        return [int(x) for x in self.shots.split(",")]

    def to_dict(self) -> dict:
        return asdict(self)

    def source_config(self) -> SourceConfig:
        return SourceConfig(reference=self.reference, seeds=self.seeds)

    def validate(self) -> None:
        self.source_config().validate()
        budgets = self.shot_list()
        if budgets != sorted(set(budgets)) or min(budgets) < 1:
            raise ValueError("Shots must be distinct, positive and increasing")
        if min(self.repeats, self.eval_batch_size) < 1 or self.sampling_seed < 0:
            raise ValueError(
                "Positive repeats/batch and nonnegative sampling seed required"
            )
        if self.val_limit < 0 or 0 < self.val_limit < 18:
            raise ValueError("val_limit must be zero or at least 18")
        if not self.development and (
            self.val_limit or self.seed_list() != [0, 1, 2, 3, 4]
        ):
            raise ValueError(
                "Formal evaluation requires all five seeds and full splits"
            )


def check_output(config: Config, output: Path) -> None:
    check_source_output(config.source_config(), output)
    resolved = output.resolve()
    for protected in (ROOT / "data", ROOT / "experiments"):
        if resolved == protected or resolved.is_relative_to(protected):
            raise ValueError(
                "Evaluation outputs cannot be written into data or source trees"
            )


def source_hashes() -> dict:
    files = [*PACKAGE.rglob("*.py"), *(PACKAGE / "scripts").glob("*.sh")]
    return {**upstream_hashes(), **{str(p.relative_to(ROOT)): sha256(p) for p in files}}


class Sources:
    """Reuse the audited Independent/zero pair and add its paired Sequential model."""

    def __init__(self, config: Config) -> None:
        self.paired = PairedSources(config.source_config())
        self.runtime = self.paired.runtime
        self.audit = self.paired.baseline.audit
        self.hashes = self.paired.hashes.copy()
        self.pairing = self.paired.pairing.copy()
        self.data = self.paired.baseline.data["val"]
        for seed in config.seed_list():
            root = self.model_path(seed, "sequential").parent
            result = verify_complete(
                self.paired.baseline, cell_name("sequential", seed, "feedback")
            )
            expected = self.pairing[str(seed)]
            for key in ("initial_model_sha256", "frontend_sha256", "global_step"):
                if result[key] != expected[key]:
                    raise ValueError(f"Sequential pairing mismatch: seed {seed}, {key}")
            if result["spec"]["control_mode"] != "measured":
                raise ValueError("Expected Sequential training with measured controls")
            orders = [r["order_sha256"] for r in read_json(root / "history.json")]
            zero = self.model_path(seed, "no_feedback").parent
            if orders != [r["order_sha256"] for r in read_json(zero / "history.json")]:
                raise ValueError("Sequential and zero sample orders differ")
            self.pin(root, ["result.json", *result["artifacts"]])
            self.pairing[str(seed)] = {
                **expected,
                "sequential_pairing_verified": True,
                "sequential_checkpoint": str(root / "endpoint.pt"),
            }
            for training, mode in CONDITIONS:
                directory = self.reference_predictions(seed, training, mode).parent
                lock = read_json(directory / "evaluation_lock.json")
                if lock["checkpoint_sha256"] != sha256(self.model_path(seed, training)):
                    raise ValueError("Reference predictions use a different checkpoint")
                verify_artifacts(directory, lock["artifacts"])
                self.pin(directory, ["evaluation_lock.json", *lock["artifacts"]])
        for key, hash_key in (
            ("dataset", "dataset_sha256"),
            ("admission", "admission_sha256"),
        ):
            path = Path(self.audit[key])
            if sha256(path) != self.audit[hash_key]:
                raise ValueError(f"Pinned {key} changed")
            self.hashes[str(path)] = self.audit[hash_key]
        manifest = Path(self.audit["dataset"]).with_suffix(".json")
        if sha256(manifest) != self.audit["dataset_manifest_sha256"]:
            raise ValueError("Dataset manifest changed")
        self.hashes[str(manifest)] = sha256(manifest)

    def pin(self, root: Path, names) -> None:
        self.hashes.update({str(root / name): sha256(root / name) for name in names})

    def model_path(self, seed: int, training: str) -> Path:
        if training == "sequential":
            return self.paired.independent.model_path("sequential", seed)
        if training not in {"independent", "no_feedback"}:
            raise ValueError(f"Unknown training mode: {training}")
        return self.paired.model_path(
            seed, "feedback" if training == "independent" else "no_feedback"
        )

    def reference_predictions(self, seed: int, training: str, mode: str) -> Path:
        if training == "no_feedback":
            return self.model_path(seed, training).parent / "predictions.pt"
        if training == "independent":
            return (
                self.model_path(seed, training).parent
                / "evaluation"
                / mode
                / "predictions.pt"
            )
        return (
            Path(self.paired.independent.config.sequential_reference)
            / f"seed{seed}"
            / mode
            / "predictions.pt"
        )

    def make_model(self, weights: dict):
        return self.paired.baseline.make_model(weights).eval().requires_grad_(False)
