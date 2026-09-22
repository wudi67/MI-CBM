"""Reuse Adam continuation; add true-control validation without changing updates."""

from experiments.grouped_robot_continuation.training import Route as PreviousRoute
from experiments.grouped_robot_pilot.evaluation import evaluate


class Route(PreviousRoute):
    def save(self) -> None:
        progress = self.progress
        epoch = progress["completed_epoch"]
        interval = self.shared.parent.config.diagnostic_every
        if (
            self.cell == "independent"
            and epoch > self.start_epoch
            and (epoch % interval == 0 or epoch == self.epochs)
            and progress["order"] is None
            and progress["offset"] == 0
            and "validation_true" not in progress["history"][-1]
        ):
            # Insert before the parent's atomic checkpoint/history writes. An
            # interrupted epoch therefore cannot silently skip this diagnostic.
            self.heartbeat("validation_true_controls")
            value, _ = evaluate(
                self.model,
                self.shared.data["validation"],
                self.config.eval_batch_size,
                mask=31,
                states=self.shared.state_cache["validation"],
            )
            progress["history"][-1]["validation_true"] = value
        super().save()
