"""Sample physical (five-bit concept record, label) outcomes on CUDA."""

import torch

from experiments.grouped_dynamic_vqc.evaluation import label_metrics
from experiments.grouped_dynamic_vqc.model import sample_shots
from experiments.grouped_robot_pilot.evaluation import evaluate, metrics
from experiments.grouped_robot_pilot.model import bits, codes
from experiments.grouped_robot_pilot.protocol import CONCEPTS
from experiments.grouped_shots_final.evaluation import draw_seed, validate_joint

from .protocol import Config


def exact_metrics(raw: dict) -> dict:
    validate_joint(raw)
    return metrics(raw)


@torch.no_grad()
def infer(model, data: dict, states: torch.Tensor, mode: str, batch: int, tick) -> dict:
    if mode not in ("measured", "correct_all_five", "zero"):
        raise ValueError("Unknown Robot control condition")
    evaluation_data = {**data, "concepts": data["concepts"].cuda()}
    _, raw = evaluate(
        model,
        evaluation_data,
        batch,
        states=states,
        zero=mode == "zero",
        mask=31 if mode == "correct_all_five" else 0,
        tick=tick,
    )
    validate_joint(raw)
    return raw


@torch.no_grad()
def sample_repeats(
    raw: dict, config: Config, identity: tuple[str, int, str, str], tick
) -> dict:
    validate_joint(raw)
    n, budgets = len(raw["labels"]), config.shot_list()
    shape = (config.repeats, len(budgets), n)
    values = {
        k: torch.empty(shape, dtype=torch.int32)
        for k in ("label_ones", "joint_prediction", "true_record_count")
    }
    values["bit_ones"] = torch.empty((*shape, 5), dtype=torch.int32)
    targets = codes(raw["concepts"]).cuda()
    table = bits(torch.device("cuda"))
    p, mass = raw["concept_probabilities"].cuda(), raw["branch_label_mass"].cuda()
    seeds = []
    for repeat in range(config.repeats):
        seed = draw_seed(config.sampling_seed, *identity, repeat)
        seeds.append(seed)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        for start in range(0, n, config.eval_batch_size):
            stop = min(start + config.eval_batch_size, n)
            measured, labels = sample_shots(
                {"concept_probs": p[start:stop], "branch_label_mass": mass[start:stop]},
                budgets[-1],
                generator,
            )
            for j, shots in enumerate(budgets):
                records = measured[:, :shots]
                counts = torch.zeros(
                    (stop - start, 32), device="cuda", dtype=torch.int64
                )
                counts.scatter_add_(1, records, torch.ones_like(records))
                stats = {
                    "label_ones": labels[:, :shots].sum(1),
                    "bit_ones": (counts[:, :, None] * table[None]).sum(1),
                    "joint_prediction": counts.argmax(1),
                    "true_record_count": counts.gather(
                        1, targets[start:stop, None]
                    ).squeeze(1),
                }
                for key, value in stats.items():
                    values[key][repeat, j, start:stop] = value.cpu().to(torch.int32)
            tick("sampling", repetition=repeat + 1, offset=stop)
    return {
        "budgets": budgets,
        "seeds": seeds,
        "values": values,
        "device": str(p.device),
        "distribution": "joint physical (m,y)",
    }


def sampled_metrics(raw: dict, samples: dict) -> list[dict]:
    truth, labels = raw["concepts"].bool(), raw["labels"]
    targets = codes(raw["concepts"])
    rows = []
    for repeat, seed in enumerate(samples["seeds"]):
        for j, shots in enumerate(samples["budgets"]):
            v = {k: x[repeat, j] for k, x in samples["values"].items()}
            for key, value in v.items():
                expected = truth.shape if key == "bit_ones" else labels.shape
                if (
                    value.shape != expected
                    or value.min() < 0
                    or value.max() > (31 if key == "joint_prediction" else shots)
                ):
                    raise ValueError("Invalid finite-shot counts/predictions")
            marginals = v["bit_ones"].float() / shots
            matches = (marginals >= 0.5) == truth
            true_probability = v["true_record_count"].float() / shots
            concept = {
                "mean_bit_accuracy": float(matches.float().mean()),
                "all_concepts_accuracy": float(matches.all(1).float().mean()),
                "joint_map_accuracy": float(
                    (v["joint_prediction"] == targets).float().mean()
                ),
                "joint_single_shot_probability": float(true_probability.mean()),
                "joint_nll": float(-true_probability.clamp_min(1e-7).log().mean()),
                "per_concept": {
                    name: label_metrics(marginals[:, i], raw["concepts"][:, i])
                    for i, name in enumerate(CONCEPTS)
                },
            }
            rows.append(
                {
                    "shots": shots,
                    "repeat": repeat,
                    "sampling_seed": seed,
                    "label": label_metrics(v["label_ones"].float() / shots, labels),
                    "concept": concept,
                }
            )
    return rows
