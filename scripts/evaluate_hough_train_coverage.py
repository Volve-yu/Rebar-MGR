"""Evaluate the train-maximum-coverage configuration, without val selection."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

from tune_mask_hough import candidates, fit_centers
from run_traditional_measurement import (
    AnnotationStore, RulerConfig, dump, load_depth_labels, load_scaled_gt_xs,
    read_ids, score, summarize,
)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--pred-root',type=Path,required=True)
    p.add_argument('--annotations',type=Path,required=True)
    a=p.parse_args()
    store=AnnotationStore(a.annotations)
    cv2.setNumThreads(4)
    for site in ('site-1','site-2'):
        root=a.results/site
        output=root/'train_max_coverage_metrics.json'
        if output.exists():
            raise FileExistsError(output)
        calibration=json.loads((root/'calibration.json').read_text())
        protocol=json.loads((root/'protocol.json').read_text())
        if protocol['annotation_sha256']!=store.sha256:
            raise ValueError('References changed')
        choice=calibration['maximum_train_coverage_candidate']
        cfg=choice['config']
        rows=[]
        distances=load_depth_labels(str(a.data_root/site/'depth_labels.txt'))
        for sid in read_ids(a.data_root,site,'val'):
            path=a.pred_root/(site+'_full_seed42')/'predictions/within/model'/site/'seg'/(sid+'.png')
            mask=cv2.imread(str(path),cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(path)
            h,w=mask.shape
            mpp=RulerConfig(hfov_deg=87,image_width_px=w,image_height_px=h).mm_per_pixel(distances[sid])
            xs=load_scaled_gt_xs(a.data_root,site,sid,w)
            truth=store.spacings(site,sid)
            if not xs or len(truth)!=len(xs)-1:
                raise ValueError('Reference mismatch')
            start=time.perf_counter()
            pred=fit_centers(candidates(mask),mask.shape,cfg)
            rows.append(score(site,sid,pred,xs,truth,mpp,time.perf_counter()-start))
        report=summarize(rows)
        rng=np.random.default_rng(42)
        boots=[summarize([rows[j] for j in rng.integers(0,len(rows),len(rows))]) for _ in range(1000)]
        keys=('spacing_mae_mm','spacing_rmse_mm','interval_coverage','count_mae','failure_rate')
        report['image_bootstrap_95ci']={k:np.percentile([b[k] for b in boots if b[k] is not None],[2.5,97.5]).tolist() if any(b[k] is not None for b in boots) else None for k in keys}
        report['selection']=dict(objective='maximum training interval coverage, not validation',config=cfg,train_metrics=choice['metrics'])
        dump(output,report)
        (root/'train_max_coverage_per_image.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
        print(site,json.dumps(report),flush=True)
    path=Path(__file__).resolve()
    dump(a.results/'coverage_evaluator_hash.json',{str(path):hashlib.sha256(path.read_bytes()).hexdigest()})


if __name__=='__main__':
    main()
