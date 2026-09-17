import json

from phase2.run_fast import Pipeline


def test_completed_shards_do_not_wait_for_gpus(tmp_path):
    (tmp_path/"protocol.json").write_text("{}")
    directory=tmp_path/"evaluations/u0030"
    directory.mkdir(parents=True)
    for shard in range(8):
        (directory/f"shard-{shard}-complete.json").write_text(json.dumps({"max_jobs":None,"jobs":1}))
        (directory/f"shard-{shard}.jsonl").write_text(json.dumps({"trajectory_id":str(shard)})+"\n")
    pipeline=Pipeline(tmp_path)
    pipeline.parallel("phase2.evaluate",30)
    assert not (tmp_path/"status.json").exists()


def test_smoke_shard_is_not_a_completed_evaluation(tmp_path):
    (tmp_path/"protocol.json").write_text("{}")
    directory=tmp_path/"evaluations/u0031"
    directory.mkdir(parents=True)
    (directory/"shard-0-complete.json").write_text(json.dumps({"max_jobs":1,"jobs":1}))
    assert not Pipeline(tmp_path).shard_complete("phase2.evaluate",31,0)
