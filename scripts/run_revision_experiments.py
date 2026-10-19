"""Print or execute reproducible within-site and bidirectional cross-site runs."""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run_root',type=Path,required=True)
    p.add_argument('--annotations',default='annotation_all_images/all_images_annotations.csv')
    p.add_argument('--seeds',nargs='+',type=int,default=[42])
    p.add_argument('--variants',nargs='+',choices=['full','seg_only','edge_aux','measure_only'],default=['full'])
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if args.run_root.exists():raise FileExistsError('Choose a fresh run root.')
    commands=[]
    for seed in args.seeds:
        for variant in args.variants:
            for source,target in [('site-1','site-2'),('site-2','site-1')]:
                name=f'{source}_{variant}_seed{seed}';base=args.run_root/name
                checkpoint=base/'checkpoints';model='model'
                edge=.1 if variant in ('full','edge_aux') else 0
                proj=.15 if variant in ('full','measure_only') else 0
                commands.append(['python','src_measure_guided/train.py','--subsets',source,'--seed',str(seed),
                    '--save_dir',str(checkpoint),'--log_dir',str(base/'logs'),'--alpha_edge',str(edge),'--beta_proj',str(proj),
                    '--lambda_c','0','--max_edge_to_seg','0.04','--max_proj_to_seg','0.05',
                    '--aux_schedule','warmup_cosine_decay','--aux_warmup_epochs','5','--aux_min_scale','0.2',
                    '--seg_refine_epochs','5','--device','cuda'])
                for label,site,split in [('train',source,'train'),('within',source,'val'),('cross',target,'val')]:
                    commands.append(['python','src_measure_guided/predict.py','--checkpoint',str(checkpoint/'best.pth'),
                        '--subsets',site,'--split',split,'--pred_dir',str(base/'predictions'/label/model),'--device','cuda'])
                    cmd=['python','scripts/run_physical_inspection.py','--mode','calibrate' if label=='train' else 'evaluate',
                         '--annotations',args.annotations,'--pred_root',str(base/'predictions'/label),'--model',model,
                         '--subsets',site,'--split',split,'--out_dir',str(base/label)]
                    if label!='train':cmd+=['--profile',str(base/'train/peak_profile.json'),'--protocol','cross_site' if label=='cross' else 'within_site']
                    commands.append(cmd)
    commands=[[part.replace('\\','/') for part in command] for command in commands]
    args.run_root.mkdir(parents=True)
    (args.run_root/'commands.json').write_text(json.dumps(commands,indent=2),encoding='utf-8')
    (args.run_root/'run.sh').write_text('set -euo pipefail\nexport OMP_NUM_THREADS=4\n'+'\n'.join(shlex.join(c) for c in commands)+'\n',encoding='utf-8')
    print(f'{len(commands)} commands prepared in {args.run_root}; execute={args.execute}')
    if args.execute:
        for cmd in commands:
            cmd[0]=sys.executable
            subprocess.run(cmd,cwd=ROOT,check=True)


if __name__=='__main__':main()
