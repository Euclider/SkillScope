import pytest
import torch

from phase2.elastic_training import minibatch_indices,cpu_adam_step,validate_elastic_dispatch


def elastic_config():
    from omegaconf import OmegaConf
    return OmegaConf.create({"phase2":{"enabled":True},"algorithm":{"adv_estimator":"grpo"},
        "data":{"train_batch_size":8},"env":{"rollout":{"n":4}},
        "actor_rollout_ref":{"model":{"load_text_only":True},"rollout":{"name":"hf","n":1},
        "actor":{"ppo_mini_batch_size":32,"ppo_micro_batch_size_per_gpu":1,"use_dynamic_bsz":False}}})


def test_elastic_dispatch_uses_real_environment_rollout_count():
    config=elastic_config()
    validate_elastic_dispatch(config)


@pytest.mark.parametrize("path,value",[("phase2.enabled",False),("actor_rollout_ref.rollout.name","vllm"),
    ("algorithm.adv_estimator","gae"),("data.train_batch_size",7),
    ("actor_rollout_ref.actor.ppo_mini_batch_size",31),("actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu",2),
    ("actor_rollout_ref.actor.use_dynamic_bsz",True),("actor_rollout_ref.model.load_text_only",False)])
def test_elastic_dispatch_refuses_unaudited_paths(path,value):
    from omegaconf import OmegaConf
    config=elastic_config();OmegaConf.update(config,path,value)
    with pytest.raises(ValueError,match="Unsupported elastic training"):
        validate_elastic_dispatch(config)


@pytest.mark.parametrize("world",range(1,9))
@pytest.mark.parametrize("total",[1,8,32,33,740,744])
def test_global_optimizer_batch_has_exact_weights(world,total):
    ranks=[minibatch_indices(total,world,r) for r in range(world)]
    observed=[]
    for batch in range(len(ranks[0])):
        current=[]
        weightsum=0.
        for rank in ranks:
            indices,weights=rank[batch]
            for index,weight in zip(indices,weights):
                if weight:current.append(index)
                # FSDP mean-reduces gradients over world ranks.
                weightsum+=weight*(world/32)/world
        assert len(current)<=32
        assert weightsum==pytest.approx(len(current)/32)
        observed.extend(current)
    assert sorted(observed)==list(range(total))


def test_cpu_adam_preserves_parameter_keys_moments_and_updates():
    a=torch.nn.Parameter(torch.tensor([1.,2.,3.]))
    b=torch.nn.Parameter(a.detach().clone())
    oa=torch.optim.AdamW([a],lr=1e-6)
    ob=torch.optim.AdamW([b],lr=1e-6)
    for _ in range(3):
        a.grad=torch.tensor([.4,-.2,.1]);b.grad=a.grad.clone()
        oa.step();cpu_adam_step(ob)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        assert ob.param_groups[0]["params"][0] is b
        for key in oa.state[a]:torch.testing.assert_close(oa.state[a][key],ob.state[b][key],rtol=0,atol=0)
