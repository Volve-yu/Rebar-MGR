"""Official Mobile-Seed training/export followed by the shared measurement protocol."""
import argparse, json, math, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from experiment_snapshot import snapshot, file_sha
REPO=ROOT/'baselines_clean/official/Mobile-Seed/repo'

def main():
 p=argparse.ArgumentParser(); p.add_argument('--sites',nargs='+',default=['site-1','site-2'],choices=['site-1','site-2'])
 p.add_argument('--epochs',type=int,default=50); p.add_argument('--batch-size',type=int,default=4)
 p.add_argument('--resume',action='store_true'); p.add_argument('--run-root',type=Path,default=ROOT/'runs/experiment1_mobile_seed'); a=p.parse_args()
 os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
 weight=REPO/'ckpt/AFFormer_tiny_ImageNet1k.pth'
 if not weight.exists(): raise FileNotFoundError('Download the official ImageNet weight first: '+str(weight))
 def call(cmd,cwd,run,name):
  marker=run/(name+'.done.json')
  if a.resume and marker.exists(): return
  with (run/(name+'.log')).open('a',encoding='utf8') as f:
   q=subprocess.Popen([sys.executable,'-u',*map(str,cmd)],cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace')
   for line in q.stdout: print(line,end='',flush=True); f.write(line); f.flush()
   if q.wait(): raise RuntimeError(name+' failed; fix the issue and use --resume')
  marker.write_text(json.dumps(list(map(str,cmd))),encoding='utf8')
 for site in a.sites:
  run=a.run_root.resolve()/site
  if run.exists() and not a.resume: raise FileExistsError(run)
  run.mkdir(parents=True,exist_ok=True); work=run/'work'; data=run/'data_ms'
  snap=snapshot(ROOT/'data',[site],['train','val']); sp=run/'training_snapshot.json'
  if sp.exists() and json.loads(sp.read_text())!=snap: raise ValueError('Training dataset changed')
  sp.write_text(json.dumps(snap),encoding='utf8')
  call([REPO/'tools/prepare_rebar_data.py','--project-root',ROOT,'--subsets',site,'--out-root',data],ROOT,run,'prepare')
  steps=math.ceil(len(snap['members'][site+'/train'])/a.batch_size)
  cfg=REPO/'configs/rebar/MS_tiny_rebar.py'
  cmd=[REPO/'tools/train.py',cfg,'--work-dir',work,'--seed','42']
  if a.resume and (work/'latest.pth').exists(): cmd+=['--resume-from',work/'latest.pth']
  cmd+=['--cfg-options',f'data.samples_per_gpu={a.batch_size}',f'runner.max_iters={steps*a.epochs}',
        f'checkpoint_config.interval={steps}',f'evaluation.interval={steps}','evaluation.save_best=mIoU',
        'evaluation.rule=greater',f'model.pretrained={weight}',*[f'data.{x}.data_root={data}' for x in ('train','val','test')]]
  call(cmd,REPO,run,'train')
  best=list(work.glob('best_*.pth'))
  if len(best)!=1: raise ValueError('Expected one validation-selected best checkpoint: '+str(best))
  ck=best[0]
  for split in ('train','val'):
   pred=run/'predictions'/split; name='predict_'+split
   if a.resume and pred.exists() and not (run/(name+'.done.json')).exists(): pred.rename(pred.with_name(split+'.incomplete.'+str(time.time_ns())))
   call([REPO/'tools/export_rebar.py',cfg,ck,'--data-root',data,'--subset',site,'--split',split,'--out-dir',pred],REPO,run,name)
   for sid in snap['members'][site+'/'+split]:
    if not (pred/'mobile_seed'/site/'seg'/(sid+'.png')).exists(): raise FileNotFoundError(sid)
   (pred/'mobile_seed/prediction_manifest.json').write_text(json.dumps(dict(checkpoint=str(ck),checkpoint_sha256=file_sha(ck),training_snapshot=snap,prediction_snapshot=snapshot(ROOT/'data',[site],[split]))),encoding='utf8')
   name='calibrate' if split=='train' else 'evaluate'; out=run/name
   if a.resume and out.exists() and not (run/(name+'.done.json')).exists(): out.rename(out.with_name(name+'.incomplete.'+str(time.time_ns())))
   cmd=[ROOT/'scripts/run_physical_inspection.py','--mode',name,'--pred_root',pred,'--model','mobile_seed','--subsets',site,'--split',split,'--out_dir',out]
   if split=='val': cmd+=['--profile',run/'calibrate/peak_profile.json']
   call(cmd,ROOT,run,name)
if __name__=='__main__': main()
