"""Read and audit the completed long-concept arm without writing upstream files."""

from pathlib import Path

from experiments.grouped_dynamic_vqc.runtime import ROOT, sha256
from experiments.grouped_robot_continuation.protocol import JOBS
from experiments.grouped_robot_continuation.protocol import Config as SourceConfig
from experiments.grouped_robot_continuation.protocol import source_hashes as old_sources
from experiments.grouped_robot_continuation.reference import Reference as PilotReference
from experiments.grouped_robot_continuation.runner import (
    TRAIN_CONDITIONS,
    verify_pair,
)
from experiments.grouped_robot_continuation.runner import (
    Experiment as SourceExperiment,
)
from experiments.grouped_robot_continuation.training import Job, verify_job
from experiments.grouped_robot_pilot.protocol import CONDITIONS, read_json, verify_files


class Source(SourceExperiment):
    """Use only the parent's read-only verification helpers."""

    # The normal experiment constructor writes heartbeat and reference files.
    def __init__(self, output: Path, runtime: dict, tick) -> None:  # pylint: disable=super-init-not-called
        self.output = output.resolve()
        self.config = SourceConfig(**read_json(self.output / "config.json"))
        self.manifest = read_json(self.output / "manifest.json")
        self.manifest_hash = sha256(self.output / "manifest.json")
        self.runtime = runtime
        root = Path(self.config.reference)
        expected = {
            "config": self.config.to_dict(),
            "runtime": runtime,
            "sources": old_sources(),
            "reference_manifest_sha256": sha256(root / "manifest.json"),
            "reference_result_lock_sha256": sha256(root / "result_lock.json"),
        }
        if any(self.manifest.get(k) != v for k, v in expected.items()):
            raise ValueError(
                "Source experiment configuration/runtime/provenance changed"
            )
        verify_files(self.output, self.manifest["artifacts"])
        locked = self.verify_result_lock()
        if locked is None:
            raise ValueError("Source continuation must be fully completed")
        self.reference = PilotReference(root, runtime, tick)
        self.config.validate(self.reference.config)
        self.data = self.reference.data
        if (
            read_json(self.output / "reference_lock.json")["artifacts"]
            != self.reference.pins
        ):
            raise ValueError("Source pilot reference lock changed")
        self.training = {}
        for arm, cell in JOBS:
            tick("verifying_source_training", source_cell=f"{arm}/{cell}")
            self.training[f"{arm}/{cell}"] = verify_job(Job(self, arm, cell))
        for arm in ("long_concept", "long_label"):
            verify_pair(self, arm, self.training)
        for arm in ("baseline", "long_concept", "long_label"):
            for role, conditions in (
                ("validation", CONDITIONS),
                ("train", TRAIN_CONDITIONS),
            ):
                for cell, name, mask in conditions:
                    tick("verifying_source_evaluation")
                    if self.verify_condition(arm, role, cell, name, mask) is None:
                        raise ValueError(
                            "Source continuation has an incomplete evaluation"
                        )
        self.pins = {
            **self.reference.pins,
            **{
                str(self.output / name): sha256(self.output / name)
                for name in (*locked["artifacts"], "result_lock.json")
            },
        }


class Reference:
    """Expose long_concept checkpoints to the existing Adam continuation route."""

    def __init__(self, output: Path, runtime: dict, tick) -> None:
        self.source = Source(output, runtime, tick)
        self.output = self.source.output / "long_concept"
        # The original label sample-order offset is 100, even though the actual
        # frontend has now completed 300 epochs. Never substitute 300 here.
        self.config = self.source.reference.config
        self.actual_concept_epochs = self.source.config.concept_epochs
        self.data = self.source.data
        self.data_hash = self.source.reference.data_hash
        self.pins = self.source.pins
        self.training = {
            cell: self.source.training[f"long_concept/{cell}"]
            for cell in ("concept", "independent", "no_feedback")
        }
        if self.training["concept"]["epochs"] != self.actual_concept_epochs:
            raise ValueError("Selected frontend has the wrong training budget")
        if (
            self.training["concept"]["frontend_sha256"]
            != self.training["independent"]["frontend_sha256"]
        ):
            raise ValueError(
                "Label source is not attached to the long-concept frontend"
            )

    def initial_for(self, cell: str) -> dict:
        return Job(self.source, "long_concept", cell).initial_for(cell)

    def verify_unchanged(self) -> None:
        verify_files(Path("/"), self.pins)
        verify_files(ROOT, self.source.manifest["sources"])
        self.source.reference.verify_unchanged()
