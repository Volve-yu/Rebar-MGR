import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/(name+'.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class SupplementsTest(unittest.TestCase):
    def test_common_mask_metrics_and_missing_image(self):
        mod = module('run_revision_supplements')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for site in ('site-1', 'site-2'):
                split = root/'data'/site/'gt_sets/val.txt'
                split.parent.mkdir(parents=True)
                split.write_text('one\ntwo\n')
                for sid in ('one','two'):
                    gt = np.zeros((480,640),dtype=np.uint8)
                    gt[:,:320] = 255
                    pred = gt.copy() if sid=='one' else 255-gt
                    for directory, pixels in [(root/'data'/site/'segmentation',gt),
                        (root/'runs'/f'experiment1_mobile_seed/{site}/predictions/val/mobile_seed/{site}/seg',pred)]:
                        directory.mkdir(parents=True,exist_ok=True)
                        Image.fromarray(pixels).save(directory/(sid+'.png'))
            a = argparse.Namespace(data_root=root/'data',runs_root=root/'runs',out_root=root/'out',methods=['mobile_seed'],plan_only=False)
            mod.mask_eval(a)
            result = json.loads((root/'out/mask_seg/site-1_mobile_seed.json').read_text())
            self.assertEqual(result['n_images'],2)
            self.assertAlmostEqual(result['metrics']['seg_f1'],.5,places=6)
            self.assertAlmostEqual(result['metrics']['seg_iou'],.5,places=6)
            (root/'data/site-2/segmentation/two.png').unlink()
            a.out_root = root/'new_out'
            with self.assertRaises(FileNotFoundError):
                mod.mask_eval(a)
            self.assertFalse(a.out_root.exists())

    def test_design_rule_prompt_and_no_rgb_fact_leak(self):
        mod = module('run_external_reports_v2')
        record = dict(measurement=dict(schema_version='design_physical_v2',rebar_count=2,
            spacings_mm=[150.0],mean_spacing_mm=150.0,std_spacing_mm=0,
            compliance=dict(status='FAIL',schema_version='design_physical_v2',
                rule=dict(region_id='A',design_spacing_mm=100,lower_tolerance_mm=-5,upper_tolerance_mm=10),
                per_spacing=[dict(index=0,spacing_mm=150,deviation_from_design_mm=50,within_design_tol=False,
                    design_spacing_mm=100,lower_tolerance_mm=-5,upper_tolerance_mm=10)])))
        text = mod.prompt(record,True)
        self.assertIn('design spacing 100',text)
        self.assertIn('[-5, 10]',text)
        self.assertIn('spacing[0]=150.00',text)
        self.assertNotIn('GB 50204',text)
        self.assertNotIn('150',mod.prompt(record,False))
        self.assertNotIn('FAIL',mod.prompt(record,False))


if __name__ == '__main__':
    unittest.main()
