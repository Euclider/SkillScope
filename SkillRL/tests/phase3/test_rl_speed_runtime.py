from omegaconf import OmegaConf
import json
import sys
import pytest


def test_runtime_changes_only_compute_options_at_u5_boundary(tmp_path):
    from logicbench_rl_speed.runtime import apply_options, OPTIONS
    cfg = OmegaConf.create({'actor_rollout_ref': {'actor': {'ppo_mini_batch_size':128,
        'ppo_micro_batch_size_per_gpu':1,'loss_agg_mode':'token-mean'},
        'rollout':{'micro_batch_size':2},'ref':{'log_prob_micro_batch_size_per_gpu':1}},
        'trainer':{'total_training_steps':150}})
    before=OmegaConf.to_container(cfg)
    apply_options(cfg, start=0)
    assert OmegaConf.to_container(cfg)==before
    apply_options(cfg, start=5)
    for key,value in OPTIONS.items():
        assert OmegaConf.select(cfg,key)==value
    assert cfg.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu==1
    assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size==128
    assert cfg.actor_rollout_ref.actor.loss_agg_mode=='token-mean'
    assert cfg.actor_rollout_ref.rollout.micro_batch_size==2
    assert cfg.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu==1
    assert cfg.trainer.total_training_steps==150


def test_child_commands_keep_fast_runtime_for_train_readout_and_gate():
    from logicbench_rl_speed.runtime import child_args
    args=['phase3.logicbench_loop','--stage','train']
    assert child_args(args)==['logicbench_rl_speed.launch','--stage','train']
    assert args[0]=='phase3.logicbench_loop'
    assert child_args(['phase2.export_model'])==['phase2.export_model']


@pytest.mark.parametrize('sealed', [False, True])
def test_handoff_never_restarts_editor_and_only_replaces_stopped_controller(tmp_path, monkeypatch, sealed):
    from logicbench_rl_speed import handoff
    root=tmp_path.resolve()
    (root/'performance').mkdir()
    (root/'launch.json').write_text(json.dumps({'gpu_ids':[2,3,4,5]}))
    seal=root/'windows/u0000-u0005/complete.json'
    if sealed:
        seal.parent.mkdir(parents=True);seal.write_text(json.dumps({'end':5}))
    monkeypatch.setattr(sys,'argv',['handoff','--root',str(root),'--controller','101',
        '--controller-start','1000','--gate','102','--gate-start','1001'])
    monkeypatch.setattr(handoff,'validate',lambda root: None)
    from phase3 import logicbench_loop
    cleaned=[]
    monkeypatch.setattr(logicbench_loop,'cleanup',lambda root,end:cleaned.append(end))
    parent={'state':'T','start':'1000','argv':[b'phase3.logicbench_loop',str(root).encode()]}
    monkeypatch.setattr(handoff,'process',lambda pid: parent if pid==101 else {'state':'Z','start':'1001'})
    signals=[]
    def kill(pid, sig):
        signals.append((pid,sig))
        if sig==handoff.signal.SIGKILL: parent['state']='Z'
    monkeypatch.setattr(handoff.os,'kill',kill)
    class ExecReached(Exception): pass
    def execv(path,args):
        assert args[-1]=='--execute' and 'run_logicbench_phase3_fast.sh' in path
        assert args[args.index('--gpus')+1]=='2,3,4,5'
        assert args[args.index('--setting')+1]==str(root/'setting.json')
        raise ExecReached
    monkeypatch.setattr(handoff.os,'execv',execv)
    with pytest.raises(ExecReached if sealed else RuntimeError): handoff.main()
    assert signals==[(101,handoff.signal.SIGKILL if sealed else handoff.signal.SIGCONT)]
    assert (root/'performance/handoff-complete.json').exists()==sealed
    assert cleaned==([5] if sealed else [])
