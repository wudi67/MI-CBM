"""Read the completed pilot without invoking its output-writing constructor."""

from __future__ import annotations

from pathlib import Path

from experiments.grouped_dynamic_vqc.runtime import ROOT, sha256
from experiments.grouped_robot_pilot.data import prepare, table_pins
from experiments.grouped_robot_pilot.protocol import (
    CELLS,
    CONDITIONS,
    read_json,
    source_hashes,
    verify_files,
)
from experiments.grouped_robot_pilot.protocol import (
    Config as PilotConfig,
)
from experiments.grouped_robot_pilot.runner import Experiment as PilotExperiment
from experiments.grouped_robot_pilot.training import load, verify_complete


class Reference(PilotExperiment):
    """Only inherited pure validation and model-initialization methods are used."""

    # The parent's constructor writes output files; reference loading must not.
    def __init__(self, output: Path, runtime: dict, tick) -> None:  # pylint: disable=super-init-not-called
        self.output = output.resolve()
        self.config = PilotConfig(**read_json(self.output / "config.json"))
        self.config.validate()
        self.manifest = read_json(self.output / "manifest.json")
        self.runtime = runtime
        if (
            self.manifest["config"] != self.config.to_dict()
            or self.manifest["sources"] != source_hashes()
            or self.manifest["runtime"] != runtime
            or self.manifest["tables"] != table_pins(Path(self.config.dataset))
        ):
            raise ValueError(
                "Reference configuration, sources, tables or CUDA runtime changed"
            )
        self.manifest_hash = sha256(self.output / "manifest.json")
        verify_files(self.output, self.manifest["artifacts"])
        # Require every cache file before prepare(): it must take its read-only path.
        self.data_hash = sha256(self.output / "data_lock.json")
        verify_files(
            self.output, read_json(self.output / "data_lock.json")["artifacts"]
        )
        locked = self.verify_result_lock()
        if locked is None:
            raise ValueError("Reference must be a fully completed and locked pilot")
        self.initial = load(self.output / "initialization.pt")
        self.data, self.audit = prepare(self.config, self.output, tick)
        self.training = {cell: verify_complete(self, cell) for cell in CELLS}
        a, b = self.training["independent"], self.training["no_feedback"]
        for key in ("initial_model_sha256", "frontend_sha256", "global_step"):
            if a[key] != b[key]:
                raise ValueError("Reference label routes are not paired")
        self.evaluations = []
        for cell, name, mask in CONDITIONS:
            value = self.verify_condition(cell, name, mask)
            if value is None:
                raise ValueError("Reference is missing a required validation condition")
            self.evaluations.append(value)
        names = {
            *locked["artifacts"],
            "result_lock.json",
            "manifest.json",
            "config.json",
            "initialization.pt",
            "data_lock.json",
            "data.pt",
            "preprocessing.json",
            "image_sha256.json",
        }
        for cell, name, _ in CONDITIONS:
            prefix = f"validation/{cell}/{name}"
            names.update(
                f"{prefix}/{part}"
                for part in (
                    "evaluation.json",
                    "predictions.pt",
                    "evaluation_lock.json",
                )
            )
        self.pins = {
            str(self.output / name): sha256(self.output / name)
            for name in sorted(names)
        }
        self.pins.update(self.manifest["tables"])

    def verify_unchanged(self) -> None:
        verify_files(Path("/"), self.pins)
        verify_files(ROOT, self.manifest["sources"])
