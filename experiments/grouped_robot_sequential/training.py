"""Sequential uses the original measured-branch BCE step with a fresh label Adam."""

import math
from dataclasses import replace

import torch

from experiments.grouped_dynamic_vqc.runtime import atomic_json, restore_rng, sha256
from experiments.grouped_robot_independent.model import make_model, module_state
from experiments.grouped_robot_independent.training import Route as IndependentRoute
from experiments.grouped_robot_independent.training import verify_pair
from experiments.grouped_robot_pilot.protocol import read_json, verify_files
from experiments.grouped_robot_pilot.training import load
from experiments.grouped_vqc_training_modes.protocol import state_hash


class Job:
    def __init__(self, parent, seed: int):
        self.parent, self.seed, self.cell = parent, seed, "sequential"
        self.name = f"seed_{seed}/sequential"
        self.output = parent.output / f"seed_{seed}"
        self.config = replace(
            parent.reference.reference.config,
            seed=seed,
            concept_epochs=parent.reference.config.head_order_offset,
            head_epochs=parent.config.head_epochs,
            checkpoint_steps=parent.config.checkpoint_steps,
        )
        self.data, self.data_hash = parent.data, parent.data_hash
        self.manifest_hash = parent.manifest_hash

    @property
    def checkpoint_path(self):
        return self.output / "training/sequential/endpoint.pt"

    @property
    def stop_requested(self) -> bool:
        return self.parent.control["stop"]

    def initial_for(self, cell: str) -> dict:
        if cell != "sequential":
            raise ValueError("Only Sequential label training is allowed")
        return self.parent.initial[str(self.seed)]

    make_model = staticmethod(make_model)

    def prepare_states(self, model) -> None:
        self.parent.prepare_states(model)

    @property
    def state_cache(self) -> dict:
        return self.parent.state_cache

    def heartbeat(self, status: str, **details) -> None:
        self.parent.heartbeat(status, **details)


class Route(IndependentRoute):
    """Only initialization/identity differ; step/run remain the original methods.

    The original constructor accepts only concept/independent/no_feedback. This
    adapter initializes the same fields with cell='sequential'; the inherited
    step then uses zero=False, mask=0, i.e. the measured record for every branch.
    """

    def __init__(self, shared: Job):  # pylint: disable=super-init-not-called
        self.shared, self.config, self.cell = shared, shared.config, "sequential"
        self.output = shared.output / "training/sequential"
        self.output.mkdir(parents=True, exist_ok=True)
        self.epochs, self.epoch_offset = (
            self.config.head_epochs,
            self.config.concept_epochs,
        )
        initial = shared.initial_for(self.cell)
        self.model = make_model(initial["model"])
        restore_rng(initial["rng"])
        self.initial_hash = state_hash(initial["model"])
        self.frozen_module = "frontend"
        self.frozen_hash = state_hash(self.model.frontend.state_dict())
        self.model.frontend.requires_grad_(False)
        self.model.label_head.requires_grad_(True)
        self.parameters = list(self.model.label_head.parameters())
        self.optimizer = torch.optim.Adam(self.parameters, lr=self.config.learning_rate)
        self.progress = {
            "completed_epoch": 0,
            "global_step": 0,
            "order": None,
            "offset": 0,
            "loss_sums": [0.0],
            "history": [],
        }
        self.gradient_checks: dict = {}
        self.training_seconds = 0.0
        if (self.output / "resume.pt").exists():
            saved = load(self.output / "resume.pt")
            if any(saved.get(k) != v for k, v in self.identity().items()):
                raise ValueError("Resume Sequential identity/control mismatch")
            self.model.load_state_dict(saved["model"])
            self.optimizer.load_state_dict(saved["optimizer"])
            self.progress, self.gradient_checks = (
                saved["progress"],
                saved["gradient_checks"],
            )
            self.training_seconds = saved["training_seconds"]
            restore_rng(saved["rng"])
        else:
            atomic_json(
                self.output / "initialization.json",
                {
                    **self.identity(),
                    "fresh_adam": True,
                    "frontend_sha256": self.frozen_hash,
                    "head_sha256": state_hash(self.model.label_head.state_dict()),
                },
            )
            self.save()
        self.verify_frozen()
        self.verify_progress()
        shared.prepare_states(self.model)
        torch.cuda.reset_peak_memory_stats()

    def identity(self) -> dict:
        return {
            "manifest_sha256": self.shared.manifest_hash,
            "data_lock_sha256": self.shared.data_hash,
            "cell": "sequential",
            "seed": self.shared.seed,
            "head_layers": 5,
            "initial_model_sha256": self.initial_hash,
            "epochs": self.epochs,
            "epoch_offset": self.epoch_offset,
            "training_control": "measured",
        }


def verify_job(job: Job) -> dict:
    directory = job.checkpoint_path.parent
    result = read_json(directory / "result.json")
    verify_files(directory, result["artifacts"])
    saved, initial = load(job.checkpoint_path), job.initial_for("sequential")
    progress = saved["progress"]
    history = read_json(directory / "history.json")
    steps = (
        math.ceil(len(job.data["train"]["angles"]) / job.config.batch_size)
        * job.config.head_epochs
    )
    identity = {
        "manifest_sha256": job.manifest_hash,
        "data_lock_sha256": job.data_hash,
        "seed": job.seed,
        "cell": "sequential",
        "head_layers": 5,
        "initial_model_sha256": state_hash(initial["model"]),
        "epochs": job.config.head_epochs,
        "epoch_offset": job.config.concept_epochs,
        "training_control": "measured",
    }
    if (
        any(saved.get(k) != v or result.get(k) != v for k, v in identity.items())
        or result["status"] != "complete"
        or result["test_evaluated"]
        or result["global_step"] != steps
        or progress["global_step"] != steps
        or progress["completed_epoch"] != job.config.head_epochs
        or progress["order"] is not None
        or progress["offset"] != 0
        or progress["history"] != history
        or len(history) != job.config.head_epochs
        or progress["loss_sums"] != [0.0]
    ):
        raise ValueError("Sequential endpoint protocol/epoch boundary changed")
    if {int(v["step"]) for v in saved["optimizer"]["state"].values()} != {steps}:
        raise ValueError("Sequential Adam budget changed")
    frontend = state_hash(module_state(saved["model"], "frontend"))
    if (
        frontend != state_hash(module_state(initial["model"], "frontend"))
        or frontend != result["frontend_sha256"]
        or state_hash(saved["model"]) != result["model_sha256"]
        or state_hash(module_state(saved["model"], "label_head"))
        != result["head_sha256"]
    ):
        raise ValueError("Sequential weights or frozen frontend changed")
    gradients = saved["gradient_checks"]
    if (
        gradients["active_parameters"] != 112
        or gradients["gradient_l2"]["frontend"] is not None
    ):
        raise ValueError("Sequential active parameters/frozen gradients changed")
    record = {
        "seed": job.seed,
        "cell": "sequential",
        "reused": False,
        "checkpoint_path": str(job.checkpoint_path),
        "checkpoint_sha256": sha256(job.checkpoint_path),
        "initial_scope_sha256": identity["initial_model_sha256"],
        "frontend_sha256": frontend,
        "model_sha256": result["model_sha256"],
        "epochs": job.config.head_epochs,
        "epoch_offset": job.config.concept_epochs,
        "global_step": steps,
        "training_control": "measured",
        "history_order_sha256": [r["order_sha256"] for r in history],
    }
    for cell in ("independent", "no_feedback"):
        verify_pair(record, job.parent.reference.training[job.seed, cell])
    job.parent.save_once(
        f"seed_{job.seed}/training/sequential/training_reference.json", record
    )
    return record
