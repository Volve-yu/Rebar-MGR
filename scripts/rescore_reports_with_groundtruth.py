"""Re-evaluate archived reports with shared, explicitly reviewed groundtruth."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE/'scripts'))
from compare_template_with_groundtruth import load


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--groundtruth',type=Path,required=True)
    p.add_argument('--gen-files',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    truth=load(args.groundtruth)
    if not truth or any(r.get('reviewed') is not True or not r.get('reference_report') or not r.get('measurement') for r in truth.values()):
        raise ValueError('Use the reviewed groundtruth export with report text and physical-reference facts.')
    if args.out.exists(): raise FileExistsError('Use a new output directory to preserve previous evaluations.')
    batches=[]
    for file in args.gen_files:
        rows=[]
        for key,r in load(file).items():
            if key not in truth: raise ValueError(f'Missing reviewed groundtruth for {key}')
            if r.get('error') or not str(r.get('prediction') or '').strip():
                raise ValueError(f'Incomplete generation in {file}: {key}')
            row=copy.deepcopy(r)
            row['input_measurement']=row.get('input_measurement',row.get('measurement'))
            row['measurement']=truth[key]['measurement']
            row['reference_report']=truth[key]['reference_report']
            row['evaluation_protocol']='shared_reviewed_groundtruth'
            rows.append(row)
        if not rows: raise ValueError(f'Empty generation file {file}')
        batches.append((file,rows))
    args.out.mkdir(parents=True)
    files=[]
    for index,(source,rows) in enumerate(batches):
        file=args.out/f'{index:03d}_{source.stem}.jsonl'
        file.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
        files.append(file)
    (args.out/'sources.json').write_text(json.dumps({str(p):str(source) for p,(source,_) in zip(files,batches)},indent=2),encoding='utf-8')
    subprocess.run([sys.executable,str(CODE/'src_report/eval.py'),'--gen_files',*map(str,files),
                    '--out_dir',str(args.out/'metrics')],check=True)


if __name__=='__main__':
    main()
