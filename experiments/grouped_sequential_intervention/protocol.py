"""Read-only source verification and isolated evaluation provenance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from experiments.grouped_dynamic_vqc.runtime import ROOT, array_hash, sha256
from experiments.grouped_feedback_ablation.protocol import (
    DEFAULT_OUTPUT as DEFAULT_REFERENCE,
)
from experiments.grouped_feedback_ablation.protocol import (
    Config as SourceConfig,
)
from experiments.grouped_feedback_ablation.protocol import (
    Experiment as SourceExperiment,
)
from experiments.grouped_feedback_ablation.protocol import (
    cell_name,
    read_json,
)
from experiments.grouped_feedback_ablation.protocol import (
    source_hashes as upstream_hashes,
)
from experiments.grouped_feedback_ablation.runner import verify_complete

PACKAGE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs/grouped_sequential_intervention/dsprites_l4_five_seeds"
MODES = ("measured", "shape", "scale", "both")
LABELS = {
    "measured": "不纠正",
    "shape": "纠正 Shape",
    "scale": "纠正 Scale",
    "both": "全部纠正",
}


@dataclass(frozen=True)
class Config:
    reference: str = str(DEFAULT_REFERENCE)
    seeds: str = "0,1,2,3,4"
    eval_batch_size: int = 2048
    shots: int = 256
    val_limit: int = 0

    def seed_list(self) -> list[int]:
        return [int(item) for item in self.seeds.split(",")]

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> None:
        seeds = self.seed_list()
        if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
            raise ValueError("Provide distinct nonnegative training seeds")
        if min(self.eval_batch_size, self.shots) < 1:
            raise ValueError("Batch size and shots must be positive")
        if self.val_limit < 0 or 0 < self.val_limit < 18:
            raise ValueError("val_limit must be zero or at least 18")


def check_output(config: Config, output: Path) -> None:
    """Protect the source and its historical references before creating anything."""
    source = Path(config.reference).resolve()
    references = [source]
    if (source / "config.json").exists():
        old = read_json(source / "config.json")
        references += [Path(old[k]).resolve() for k in ("reference", "zero_reference")]
    output = output.resolve()
    for reference in references:
        if (
            output == reference
            or output.is_relative_to(reference)
            or reference.is_relative_to(output)
        ):
            raise ValueError(
                "Evaluation output must be isolated from source experiments"
            )


def source_hashes() -> dict:
    paths = [*PACKAGE.rglob("*.py"), *(PACKAGE / "scripts").glob("*.sh")]
    return {**upstream_hashes(), **{str(p.relative_to(ROOT)): sha256(p) for p in paths}}


def verify_artifacts(root: Path, hashes: dict) -> None:
    for name, digest in hashes.items():
        if sha256(root / name) != digest:
            raise ValueError(f"Locked artifact changed: {root / name}")


def open_reference(config: Config) -> tuple[SourceExperiment, dict]:
    """Only the existing resume constructor and pure verifiers are called."""
    root = Path(config.reference).resolve()
    old_config = SourceConfig(**read_json(root / "config.json"))
    if not set(config.seed_list()).issubset(old_config.seed_list()):
        raise ValueError("Requested seeds are absent from the training manifest")
    source = SourceExperiment(old_config, root, resume=True)
    names = {
        "manifest.json",
        "config.json",
        "reference_lock.json",
        *source.manifest["artifacts"],
    }
    for seed in config.seed_list():
        for variant in ("concept", "feedback"):
            cell = cell_name("sequential", seed, variant)
            result = verify_complete(source, cell)
            names.add(f"{cell}/result.json")
            names.update(f"{cell}/{name}" for name in result["artifacts"])
        cell = cell_name("sequential", seed, "feedback")
        directory = root / cell
        lock = read_json(directory / "evaluation_lock.json")
        expected = {
            "manifest_sha256": source.manifest_hash,
            "checkpoint_sha256": sha256(directory / "endpoint.pt"),
            "result_sha256": sha256(directory / "result.json"),
        }
        if any(lock[k] != v for k, v in expected.items()):
            raise ValueError("Source evaluation provenance mismatch")
        verify_artifacts(directory, lock["artifacts"])
        names.add(f"{cell}/evaluation_lock.json")
        names.update(f"{cell}/{name}" for name in lock["artifacts"])
        metrics = read_json(directory / "evaluation.json")
        if metrics["control_mode"] != "measured" or metrics["test_evaluated"]:
            raise ValueError("Expected validation-only Sequential measured controls")
    for role, data in source.data.items():
        if (
            array_hash(data["source_index"].cpu().numpy())
            != source.manifest["data_indices"][role]
        ):
            raise ValueError("Source data row selection differs from training manifest")
    reference_lock = {
        "root": str(root),
        "artifacts": {name: sha256(root / name) for name in sorted(names)},
        "training_config": old_config.to_dict(),
        "test_evaluated": False,
    }
    return source, reference_lock
