import pandas as pd

from phase1.metrics import clustered_bootstrap_interval


def test_game_clusters_not_eval_seeds_are_independent_units():
    data = pd.DataFrame({
        "game_id": ["g1", "g1", "g2", "g2"],
        "margin": [1.0, 1.0, -1.0, -1.0],
    })
    interval = clustered_bootstrap_interval(data, value_column="margin", resamples=100, seed=7)
    assert interval.n_clusters == 2
    assert interval.estimate == 0.0

