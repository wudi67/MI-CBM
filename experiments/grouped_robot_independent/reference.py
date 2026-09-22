"""Read-only admission of seed-zero endpoints; never relabel an old checkpoint."""

import math
from pathlib import Path

import torch

from experiments.grouped_dynamic_vqc.runtime import ROOT, array_hash, sha256
from experiments.grouped_robot_continuation.protocol import tree_hash
from experiments.grouped_robot_pilot.protocol import read_json, verify_files
from experiments.grouped_robot_pilot.training import load
from experiments.grouped_vqc_training_modes.protocol import epoch_order, state_hash

from .model import label_initial, make_model, module_state


def verify_endpoint(
    path: Path,
    initial: dict,
    *,
    epochs: int,
    offset: int,
    count: int,
    batch: int,
    seed: int,
    cell: str,
    learning_rate: float = 0.01,
) -> dict:
    result = read_json(path.parent / "result.json")
    verify_files(path.parent, result["artifacts"])
    value = load(path)
    history = read_json(path.parent / "history.json")
    p = value["progress"]
    steps = math.ceil(count / batch) * epochs
    expected = {
        "epochs": epochs,
        "epoch_offset": offset,
        "cell": cell,
        "initial_model_sha256": state_hash(initial["model"]),
        "training_control": "none" if cell == "concept" else "true",
    }
    if any(value.get(k) != v or result.get(k) != v for k, v in expected.items()):
        raise ValueError("Historical training initialization/budget/control differs")
    if (
        result["status"] != "complete"
        or result["test_evaluated"]
        or state_hash(value["model"]) != result["model_sha256"]
        or result["global_step"] != steps
        or p["global_step"] != steps
        or p["completed_epoch"] != epochs
        or p["offset"] != 0
        or p["order"] is not None
        or p["history"] != history
        or len(history) != epochs
    ):
        raise ValueError("Historical endpoint is not a completed fixed-budget run")
    for epoch, row in enumerate(history, 1):
        if (
            row["order_epoch"] != epoch + offset
            or row["global_step"] != epoch * math.ceil(count / batch)
            or row["order_sha256"]
            != array_hash(epoch_order(count, seed, epoch + offset).numpy())
        ):
            raise ValueError("Historical sample order differs")
    if {int(v["step"]) for v in value["optimizer"]["state"].values()} != {steps}:
        raise ValueError("Historical Adam state differs from budget")
    for group in value["optimizer"]["param_groups"]:
        if (
            group["lr"] != learning_rate
            or tuple(group["betas"]) != (0.9, 0.999)
            or group["eps"] != 1e-8
            or group["weight_decay"] != 0
        ):
            raise ValueError("Historical Adam hyperparameters differ")
    frozen = "label_head" if cell == "concept" else "frontend"
    if state_hash(module_state(value["model"], frozen)) != state_hash(
        module_state(initial["model"], frozen)
    ):
        raise ValueError("Historical frozen module changed")
    return value


def admit_root(root: Path, runtime: dict) -> dict:
    manifest = read_json(root / "manifest.json")
    lock = read_json(root / "result_lock.json")
    if (
        manifest["runtime"] != runtime
        or manifest["config"] != read_json(root / "config.json")
        or lock["manifest_sha256"] != sha256(root / "manifest.json")
        or manifest["test_read"]
        or lock["test_evaluated"]
    ):
        raise ValueError("Historical source identity/runtime changed")
    verify_files(ROOT, manifest["sources"])
    verify_files(root, manifest["artifacts"])
    verify_files(root, lock["artifacts"])
    upstream = read_json(root / "reference_lock.json")["artifacts"]
    verify_files(Path("/"), upstream)
    return {
        **upstream,
        **{
            str(root / name): sha256(root / name)
            for name in {*lock["artifacts"], "manifest.json", "result_lock.json"}
        },
    }


def reuse_sources(shared) -> tuple[dict, dict]:
    if not shared.config.reuse_seed0:
        return {}, {}
    cfg, pilot = shared.config, shared.reference.config
    concept_root, label_root = Path(cfg.concept_reference), Path(cfg.label_reference)
    pins = {}
    for root in (concept_root, label_root):
        shared.tick("verifying_reuse", source=str(root))
        pins.update(admit_root(root, shared.runtime))
    data_reference = read_json(concept_root / "data_reference.json")
    if (
        data_reference["data_hashes"] != shared.data_hashes
        or data_reference["reference_data_lock_sha256"] != shared.reference.data_hash
    ):
        raise ValueError(
            "Reused concept training used different samples or preprocessing"
        )
    old_initial = load(concept_root / "initialization.pt")["uniform/init_0"]
    fresh = shared.initial["0"]
    if state_hash(module_state(old_initial["model"], "frontend")) != state_hash(
        module_state(fresh["model"], "frontend")
    ) or tree_hash(old_initial["rng"]) != tree_hash(fresh["rng"]):
        raise ValueError("Seed-zero concept initialization/RNG differs")
    concept_path = concept_root / "uniform/init_0/training/concept/endpoint.pt"
    concept = verify_endpoint(
        concept_path,
        old_initial,
        epochs=cfg.concept_epochs,
        offset=0,
        count=len(shared.data["train"]["angles"]),
        batch=pilot.batch_size,
        seed=0,
        cell="concept",
        learning_rate=pilot.learning_rate,
    )
    expected = label_initial(fresh, module_state(concept["model"], "frontend"))
    old_label_initial = load(label_root / "initialization.pt")["eft_readout/init_0"]
    if state_hash(expected["model"]) != state_hash(
        old_label_initial["model"]
    ) or tree_hash(expected["rng"]) != tree_hash(old_label_initial["rng"]):
        raise ValueError("Seed-zero label initialization/RNG differs")
    label_path = label_root / "eft_readout/init_0/training/independent/endpoint.pt"
    verify_endpoint(
        label_path,
        expected,
        epochs=cfg.head_epochs,
        offset=cfg.head_order_offset,
        count=len(shared.data["train"]["angles"]),
        batch=pilot.batch_size,
        seed=0,
        cell="independent",
        learning_rate=pilot.learning_rate,
    )
    # Retained-state phases matter to the label circuit: verify full amplitudes,
    # in addition to targets and Born probabilities, against the original cache.
    label_data = read_json(label_root / "data_reference.json")
    states_path = Path(label_data["states_path"])
    if (
        str(states_path) not in pins
        or sha256(states_path.parent / "state_cache_lock.json")
        != label_data["reference_data_lock_sha256"]
    ):
        raise ValueError("Historical retained-state cache is not pinned")
    cached_states = load(states_path)
    if set(cached_states) != {"train", "validation"}:
        raise ValueError("Only train/validation state caches may be reused")
    model = make_model(expected["model"])
    with torch.no_grad():
        for role, data in shared.data.items():
            raw = load(
                label_root / f"eft_readout/init_0/{role}/measured/predictions.pt"
            )
            for key in ("concepts", "labels", "robot_ids", "source_index"):
                if not torch.equal(raw[key], data[key].cpu()):
                    raise ValueError("Historical label sample identities differ")
            pieces = []
            for start in range(0, len(data["angles"]), pilot.eval_batch_size):
                state = model.frontend(
                    data["angles"][start : start + pilot.eval_batch_size]
                )
                torch.testing.assert_close(
                    state.cpu(),
                    cached_states[role][start : start + pilot.eval_batch_size],
                    atol=2e-6,
                    rtol=2e-6,
                )
                pieces.append(state.reshape(-1, 32, 32).abs().square().sum(-1).cpu())
                shared.tick("verifying_reuse_born", role=role, offset=start)
            torch.testing.assert_close(
                torch.cat(pieces), raw["concept_probabilities"], atol=2e-6, rtol=2e-6
            )
    return {"0/concept": str(concept_path), "0/independent": str(label_path)}, pins
