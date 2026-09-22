"""Use the unchanged Independent optimizer and full mid-epoch recovery protocol."""

from dataclasses import replace

from experiments.grouped_robot_label_depth.training import Job as DepthJob
from experiments.grouped_robot_label_depth.training import (
    Route,
    lock_training,
    verify_job,
)

from .protocol import DEPTH, cell_name

__all__ = ["Job", "Route", "lock_training", "verify_job"]


class Job(DepthJob):
    def __init__(self, parent, method: str, index: int) -> None:  # pylint: disable=super-init-not-called
        self.parent, self.method, self.index = parent, method, index
        self.depth = DEPTH
        self.name = cell_name(method, index)
        self.output = parent.output / self.name
        self.config = replace(
            parent.reference.pilot,
            head_epochs=parent.config.head_epochs,
            checkpoint_steps=parent.config.checkpoint_steps,
        )
        self.data, self.state_cache = parent.data, parent.state_cache
        self.data_hash, self.manifest_hash = parent.data_hash, parent.manifest_hash
        self.reused = method == "uniform" and parent.reuse_uniform

    @property
    def checkpoint_path(self):
        if self.reused:
            return self.parent.reference.checkpoint(self.index)
        return self.output / "training/independent/endpoint.pt"

    def identity(self) -> dict:
        return {
            **super().identity(),
            "initialization_method": self.method,
            "initialization_sigma": None
            if self.method == "uniform"
            else self.parent.config.sigma,
            "initialization_lock_sha256": self.parent.initial_hash,
        }
