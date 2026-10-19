"""Regression tests for design rules, physical truth and report evidence boundaries."""
import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'src_report'),str(ROOT/'scripts')]
from inspection_data import AnnotationStore, design_compliance, measurement_record, matched_intervals
from design_evidence import template_report, risk_level
from generate import _build_messages
from experiment_snapshot import snapshot
from qaep_utils import measurement_features, target_risk_from_measurement
spec=importlib.util.spec_from_file_location('report_eval',ROOT/'src_report/eval.py')
report_eval=importlib.util.module_from_spec(spec);spec.loader.exec_module(report_eval)

RULE=[dict(region_id='A',design_spacing_mm=100,lower_tolerance_mm=-5,upper_tolerance_mm=10)]


class RulesTest(unittest.TestCase):
    def test_uniform_wrong_spacing_and_asymmetric_bounds(self):
        self.assertEqual(design_compliance([150,150,150],RULE)['status'],'FAIL')
        self.assertEqual(design_compliance([95,100,110],RULE)['status'],'PASS')
        comp=design_compliance([94.99,110.01],RULE)
        self.assertTrue(all(not p['within_design_tol'] for p in comp['per_spacing']))
        self.assertEqual(design_compliance([100],[])['status'],'N/A')
        self.assertEqual(design_compliance([100],RULE+RULE)['status'],'N/A')

    def test_matching_does_not_bridge_missing_bar(self):
        pairs,gaps=matched_intervals([0,100,200,300],[0,100,300],1,10)
        self.assertEqual(gaps,[(0,0)])
        self.assertEqual(len(pairs),3)

    def test_report_qaep_eval_agree(self):
        m=measurement_record([94,106,111],4,RULE)
        self.assertEqual(report_eval.measurement_facts(m)['fail_indices'],{0,2})
        self.assertEqual(target_risk_from_measurement(m),risk_level(m))
        self.assertAlmostEqual(measurement_features(m)['fail_ratio'],2/3)
        score=report_eval.evaluate_record(dict(prediction=template_report(m),reference_report=template_report(m),measurement=m))
        self.assertEqual(score['fail_index_f1'],1)
        messages=_build_messages(dict(measurement=m,reference_report='SECRET_PHYSICAL_TRUTH'), 'measure_rule')
        serialized=json.dumps(messages)
        self.assertNotIn('SECRET_PHYSICAL_TRUTH',serialized)
        self.assertNotIn('GB 50204',serialized)
        self.assertIn('[-5, 10]',serialized)

    def test_pipeline_calibration_and_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            temp=Path(temp);data=temp/'data';rows=[]
            for site in ['site-1','site-2']:
                base=data/site
                for directory in ['images','segmentation','edge','rebar_labels','gt_sets']:(base/directory).mkdir(parents=True)
                for split,ids in [('train','1\n'),('val','2\n')]:
                    (base/'gt_sets'/(split+'.txt')).write_text(ids)
                (base/'depth_labels.txt').write_text('1\t0.5\n2\t0.5\n')
                for sid in ['1','2']:
                    mask=np.zeros((120,1280),dtype=np.uint8)
                    for x in [100,400,700,1000]:mask[:,x-20:x+21]=255
                    for directory in ['images','segmentation','edge']:Image.fromarray(mask).save(base/directory/(sid+'.png'))
                    (base/'rebar_labels'/(sid+'.txt')).write_text('100\n400\n700\n1000\n')
                    for i in range(1,4):
                        rows.append(dict(sample_key=site+'/'+sid,interval_id='d'+str(i),left_bar=i,right_bar=i+1,
                            region_id='A',design_spacing_mm=100,lower_tolerance_mm=-5,upper_tolerance_mm=10,
                            submitted_spacing_mm=201,working_spacing_cm='20.1',value_source='user_reported_measurement',review_status='user_verified'))
            annotations=temp/'annotations.csv'
            with annotations.open('w',newline='') as stream:
                w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            store=AnnotationStore(annotations)
            self.assertEqual(store.spacings('site-1','1'),[201]*3)
            self.assertNotIn('submitted_spacing_mm',store.rules('site-1','1')[0])
            for stage,site,split in [('train','site-1','train'),('cross','site-2','val')]:
                pred=temp/stage/'full';(pred/site/'seg').mkdir(parents=True)
                sid='1' if stage=='train' else '2'
                Image.open(data/site/'segmentation'/(sid+'.png')).save(pred/site/'seg'/(sid+'.png'))
                manifest=dict(checkpoint_sha256='test_fixture_not_a_real_model',training_snapshot=snapshot(data,['site-1'],['train','val']),
                              prediction_snapshot=snapshot(data,[site],[split]))
                (pred/'prediction_manifest.json').write_text(json.dumps(manifest))
                cmd=[sys.executable,str(ROOT/'scripts/run_physical_inspection.py'),'--mode','calibrate' if stage=='train' else 'evaluate',
                     '--annotations',str(annotations),'--data_root',str(data),'--pred_root',str(temp/stage),'--model','full',
                     '--subsets',site,'--split',split,'--out_dir',str(temp/(stage+'_out')),'--bootstrap','10']
                if stage=='cross':cmd+=['--profile',str(temp/'train_out/peak_profile.json'),'--protocol','cross_site']
                result=subprocess.run(cmd,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
            scores=json.loads((temp/'cross_out/summary.json').read_text())
            self.assertGreater(scores['matched_spacing_mae_mm'],10) # Physical GT differs from image geometry.
            out=json.loads((temp/'cross_out/report_inputs.jsonl').read_text())
            self.assertNotEqual(out['measurement']['spacings_mm'],[201]*3)
            out['reference_report']='NEVER_SEND_PHYSICAL_TRUTH'
            self.assertNotIn('NEVER_SEND_PHYSICAL_TRUTH',json.dumps(_build_messages(out,'measure_rule')))
            qaep_data=temp/'qaep_train.jsonl';qaep_model=temp/'qaep_model.json';qaep_applied=temp/'qaep_applied.jsonl'
            for command in [
                [sys.executable,str(ROOT/'src_report/build_qaep_dataset.py'),'--input_jsonl',str(temp/'train_out/report_inputs.jsonl'),'--data_root',str(data),'--out_jsonl',str(qaep_data)],
                [sys.executable,str(ROOT/'src_report/train_qaep.py'),'--train_jsonl',str(qaep_data),'--model_out',str(qaep_model),'--epochs','2'],
                [sys.executable,str(ROOT/'src_report/apply_qaep.py'),'--input_jsonl',str(temp/'cross_out/report_inputs.jsonl'),'--data_root',str(data),'--model_path',str(qaep_model),'--out_jsonl',str(qaep_applied)],
                [sys.executable,str(ROOT/'src_report/eval.py'),'--gen_files',str(temp/'cross_out/template_reports.jsonl'),'--out_dir',str(temp/'report_metrics')],
            ]:
                result=subprocess.run(command,capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
            applied=json.loads(qaep_applied.read_text())
            self.assertIsNone(applied['qaep']['quality']['seg_iou'])
            self.assertIn('plan_text',applied['qaep'])
            bad=dict(manifest);bad['training_snapshot']=snapshot(data,['site-1','site-2'],['train','val'])
            (temp/'cross/full/prediction_manifest.json').write_text(json.dumps(bad))
            cmd[cmd.index('--out_dir')+1]=str(temp/'blocked_out')
            result=subprocess.run(cmd,capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Target site participated',result.stderr)


if __name__=='__main__':unittest.main()
