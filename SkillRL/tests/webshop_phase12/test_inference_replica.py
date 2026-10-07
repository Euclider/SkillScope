import torch
from webshop_phase12.inference_replica import synchronize_replica


class TinyPolicy(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        self.layer=torch.nn.Linear(config,config)


def test_replica_uses_current_parameters_and_preserves_sampling_rng():
    policy=TinyPolicy(3)
    rng=torch.get_rng_state().clone()
    replica=synchronize_replica(policy,None,policy.state_dict(),factory=TinyPolicy)
    assert torch.equal(torch.get_rng_state(),rng)
    assert not replica.training and not any(p.requires_grad for p in replica.parameters())
    for name,value in policy.state_dict().items():
        assert torch.equal(value,replica.state_dict()[name])
    with torch.no_grad():policy.layer.weight.add_(2)
    same=synchronize_replica(policy,replica,policy.state_dict(),factory=TinyPolicy)
    assert same is replica
    assert torch.equal(policy.layer.weight,replica.layer.weight)
