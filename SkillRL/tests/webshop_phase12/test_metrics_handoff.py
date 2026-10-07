import json
import pandas as pd


def test_portable_metrics_support_empty_utility_changes(tmp_path):
    from webshop_phase12.metrics import run,METHODS
    readout=tmp_path/'readout';readout.mkdir()
    paired=tmp_path/'paired_eval';paired.mkdir()
    records=[{'skill_id':sid,**{m:float(i) for m in ('D_sign_balance',)+METHODS}}
             for i,sid in enumerate(['gen_001','gen_002','gen_003'])]
    pd.DataFrame(records).to_csv(readout/'skill_scores.csv',index=False)
    (paired/'skill_utility.json').write_text(json.dumps([{'skill_id':r['skill_id'],'delta_m':0.} for r in records]))
    run(tmp_path)
    report=json.loads((tmp_path/'metrics/metrics.json').read_text())
    assert report['evaluated_invoked_skills']==3
    assert report['thresholds']['0.0']['D_sign_balance']['average_precision'] is None
