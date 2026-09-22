"""Read-only completed evaluation, frozen data and each mode's own checkpoint."""

from pathlib import Path

import torch

from experiments.grouped_dynamic_vqc.runtime import ROOT, sha256
from experiments.grouped_robot_pilot.protocol import read_json, verify_files
from experiments.grouped_robot_pilot.training import load
from experiments.grouped_robot_shots_final.data import (
    data_hashes,
    subset,
    verify_test_gate,
)
from experiments.grouped_robot_shots_final.evaluation import exact_metrics
from experiments.grouped_robot_shots_final.protocol import source_hashes as upstream
from experiments.grouped_robot_shots_final.reference import Sources as ModelSources

from .protocol import MODES, Config


class Sources:
    def __init__(self, config: Config, tick=lambda *_a, **_k: None):
        config.validate()
        self.config, self.output = config, Path(config.reference).resolve()
        self.stage = self.output / ("validation" if config.development else "test")
        manifest = read_json(self.output / "manifest.json")
        self.manifest_hash = sha256(self.output / "manifest.json")
        if (
            manifest["config"] != config.source().to_dict()
            or manifest["sources"] != upstream()
        ):
            raise ValueError("Completed evaluation config/source changed")
        lock = read_json(self.output / "result_lock.json")
        if lock["manifest_sha256"] != self.manifest_hash:
            raise ValueError("Reference manifest mismatch")
        verify_files(self.output, lock["artifacts"])
        summary = read_json(self.output / "summary.json")
        if summary["status"] != "complete" or (
            not config.development
            and (
                summary["development"]
                or not summary["test_evaluated"]
                or not summary["test_read"]
            )
        ):
            raise ValueError("A completed reference evaluation is required")
        self.models = ModelSources(config.source(), tick)
        self.runtime = self.models.runtime
        if manifest["runtime"] != self.runtime:
            raise ValueError("Original CUDA/software runtime is required")
        stage_lock = read_json(self.stage / "result_lock.json")
        if stage_lock["manifest_sha256"] != self.manifest_hash:
            raise ValueError("Stage reference identity mismatch")
        verify_files(self.stage, stage_lock["artifacts"])
        self.hashes = {
            **self.models.hashes,
            **{
                str(self.output / p): sha256(self.output / p)
                for p in {*lock["artifacts"], "result_lock.json"}
            },
            **{
                str(self.stage / p): sha256(self.stage / p)
                for p in {*stage_lock["artifacts"], "result_lock.json"}
            },
        }
        if not config.development:
            verify_test_gate(self.output, self.manifest_hash)
            inputs = read_json(self.stage / "test_inputs.json")["artifacts"]
            verify_files(Path("/"), inputs)
            self.hashes.update(inputs)
        data = load(self.stage / "data.pt")
        if data_hashes(data) != read_json(self.stage / "data_lock.json")["data_hashes"]:
            raise ValueError("Frozen data content changed")
        if not config.development and len(data["labels"]) != 6144:
            raise ValueError("Formal curves require all 6144 test images")
        self.data = subset(data, config.limit)
        self.checkpoints = {
            (seed, mode): self.models.model_path(seed, mode)
            for seed in config.seed_list()
            for mode in MODES
        }
        self.checkpoint_hashes = {key: sha256(p) for key, p in self.checkpoints.items()}

    def endpoint(self, seed: int, mode: str, mask: int) -> dict:
        if mask not in (0, 31):
            raise ValueError("Only zero/all correction endpoints can be reused")
        condition = "measured" if mask == 0 else "correct_all_five"
        root = self.stage / f"seed{seed}" / mode / condition
        record = read_json(root / "evaluation.json")
        if (
            record["checkpoint_sha256"] != self.checkpoint_hashes[seed, mode]
            or record["manifest_sha256"] != self.manifest_hash
            or record["data_lock_sha256"] != sha256(self.stage / "data_lock.json")
            or (record["seed"], record["training"], record["control_mode"])
            != (seed, mode, condition)
        ):
            raise ValueError("Endpoint does not match the frozen model/data")
        raw = load(root / "joint.pt")
        if exact_metrics(raw) != record["exact"]:
            raise ValueError("Historical endpoint metrics do not reproduce")
        lookup = {int(v): i for i, v in enumerate(raw["source_index"])}
        indices = [lookup[int(v)] for v in self.data["source_index"]]
        result = {k: v[indices] for k, v in raw.items()}
        for key in ("labels", "concepts", "source_index", "robot_ids"):
            if not torch.equal(result[key], self.data[key]):
                raise ValueError("Endpoint samples differ from frozen evaluation data")
        return result

    def verify_unchanged(self) -> None:
        verify_files(Path("/"), self.hashes)
        verify_files(ROOT, upstream())
        self.models.verify_unchanged()
