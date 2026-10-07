import argparse
import json
from pathlib import Path
import pandas as pd
from logicbench_phase12.metrics import METHODS,measure_method


def run(seed_dir):
    frame=pd.read_csv(seed_dir/'readout/skill_scores.csv')
    utility=pd.DataFrame(json.loads((seed_dir/'paired_eval/skill_utility.json').read_text()))
    merged=frame.merge(utility,on='skill_id',how='left',validate='one_to_one')
    included=merged[merged.delta_m.notna()]
    out=seed_dir/'metrics';out.mkdir(exist_ok=False)
    report={'schema_version':'skillscope.webshop_phase12_metrics.v1','invoked_skills':len(frame),'evaluated_invoked_skills':len(included),
        'no_eval_anchor':merged[merged.delta_m.isna()].skill_id.tolist(),'primary_score':'D_sign_balance','thresholds':{}}
    for threshold in (0.,.05):
        report['thresholds'][str(threshold)]={method:measure_method(included[method],included.delta_m,threshold=threshold)
            for method in ('D_sign_balance',)+METHODS} if len(included) else {}
    merged.to_csv(out/'skill_scores_with_utility.csv',index=False)
    (out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed-dir',type=Path,required=True);a=p.parse_args();run(a.seed_dir)
