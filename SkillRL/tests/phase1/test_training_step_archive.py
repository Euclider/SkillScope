import json
from types import SimpleNamespace

import numpy as np
import torch

from phase1.archive import archive_training_step


class TensorBatch(dict):
    @property
    def batch_size(self):
        return (next(iter(self.values())).shape[0],)


def test_training_step_archive_links_advantage_to_trajectory(tmp_path):
    batch = SimpleNamespace(
        batch=TensorBatch({
            "response_mask": torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
            "token_level_rewards": torch.tensor([[0.0, 1.0], [2.0, 9.0]]),
            "advantages": torch.tensor([[0.5, 0.5], [-1.0, 9.0]]),
        }),
        non_tensor_batch={
            "traj_uid": np.array(["t1", "t2"]),
            "uid": np.array(["g1", "g1"]),
        },
    )
    path = archive_training_step(
        output_dir=tmp_path,
        run_id="run",
        global_step=5,
        batch=batch,
        metrics={"actor/grad_norm": 0.25},
        update_type="real_rl",
    )
    payload = json.loads(path.read_text())
    assert payload["trajectory_update_summaries"][0]["advantages_sum"] == 1.0
    assert payload["trajectory_update_summaries"][1]["token_level_rewards_sum"] == 2.0
    assert payload["metrics"]["actor/grad_norm"] == 0.25
