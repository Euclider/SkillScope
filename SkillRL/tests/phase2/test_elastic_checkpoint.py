import pytest
import torch

from phase2.elastic_checkpoint import redistribute_slice, tensor_hash


@pytest.mark.parametrize("source_world", [1,2,4,8])
@pytest.mark.parametrize("target_world", [1,2,3,4,5,6,7,8])
def test_redistribution_preserves_unpadded_values_and_zeroes_padding(source_world,target_world):
    import math
    flat=torch.arange(37,dtype=torch.float32)
    old_width=math.ceil(len(flat)/source_world)
    padded=torch.cat([flat,torch.full((old_width*source_world-len(flat),),999.)])
    shards=list(padded.split(old_width))
    width=math.ceil(len(flat)/target_world)
    rebuilt=torch.cat([redistribute_slice(shards,r*width,width,len(flat)) for r in range(target_world)])
    torch.testing.assert_close(rebuilt[:len(flat)],flat,rtol=0,atol=0)
    assert torch.equal(rebuilt[len(flat):],torch.zeros_like(rebuilt[len(flat):]))
    assert tensor_hash(rebuilt[:len(flat)])==tensor_hash(flat)
