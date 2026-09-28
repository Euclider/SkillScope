import pytest
import torch

from phase3.common import ProtocolError
from phase3.run import verify_same_training_batch


def test_editor_source_episodes_must_match_readout_batch_trajectory_ids(tmp_path):
    episodes = [{'trajectory_id': f't{i}'} for i in range(128)]
    path = tmp_path / 'direction.pt'
    torch.save({'schema_version': 'skillrl.phase3.direction_batch.v1',
                'branch_id': 'readout_d', 'bank_sha256': 'a' * 64,
                'global_update': 6,
                'metadata': [{'trajectory_id': f't{i}'} for i in range(128)]}, path)
    kwargs = dict(branch='readout_d', bank_sha256='a' * 64, update=6)
    verify_same_training_batch(path, episodes, **kwargs)
    episodes[-1] = {'trajectory_id': 'foreign'}
    with pytest.raises(ProtocolError, match='do not match'):
        verify_same_training_batch(path, episodes, **kwargs)
