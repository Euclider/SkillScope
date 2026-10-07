import pytest


@pytest.mark.parametrize('failure',[None,'init','fit'])
def test_training_always_closes_its_own_environment(failure):
    from webshop_phase12.lifecycle import train_and_close_environment
    events=[]
    class Trainer:
        def init_workers(self):
            events.append('init')
            if failure=='init':raise ValueError('Initialization failed')
        def fit(self):
            events.append('fit')
            if failure=='fit':raise ValueError('Training failed')
            return 'trained'
    class Environment:
        def close(self):events.append('closed')
    if failure:
        with pytest.raises(ValueError):train_and_close_environment(Trainer(),Environment())
    else:
        assert train_and_close_environment(Trainer(),Environment())=='trained'
    assert events[-1]=='closed'
    assert events.count('closed')==1
