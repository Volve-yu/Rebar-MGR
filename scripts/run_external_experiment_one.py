"""Run unified external baselines on both sites with resumable stages."""
import argparse, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
METHODS=('deeplab','gscnn','sctransnet','afenet'); SITES=('site-1','site-2')
def main():
 p=argparse.ArgumentParser(); p.add_argument('--run-root',type=Path,default=Path('runs/experiment1_external'))
 p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS)); p.add_argument('--sites',nargs='+',choices=SITES,default=list(SITES))
 p.add_argument('--epochs',type=int,default=50); p.add_argument('--batch-size',type=int,default=2); p.add_argument('--workers',type=int,default=4); p.add_argument('--resume',action='store_true'); p.add_argument('--plan-only',action='store_true'); a=p.parse_args(); os.chdir(ROOT)
 env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',TORCH_HOME=str(ROOT/'vendor/torch')); stages=[]
 for m in a.methods:
  for s in a.sites:
   r=a.run_root/m/s; ck=r/'checkpoints'; pr=r/'predictions'; common=['--data_root','data','--subsets',s,'--num_workers',str(a.workers),'--device','cuda']
   stages += [(r,'train',['baselines_clean/train.py','--mode',m,'--epochs',str(a.epochs),'--batch_size',str(a.batch_size),'--seed','42','--save_dir',str(ck),'--log_dir',str(r/'logs'),*common]),
    (r,'eval',['baselines_clean/eval.py','--mode',m,'--checkpoint',str(ck/m/'best.pth'),'--split','val',*common]),
    (r,'predict_train',['baselines_clean/predict.py','--mode',m,'--checkpoint',str(ck/m/'best.pth'),'--split','train','--pred_dir',str(pr/'train'),'--batch_size',str(a.batch_size),*common]),
    (r,'calibrate',['scripts/run_physical_inspection.py','--mode','calibrate','--pred_root',str(pr/'train'),'--model',m,'--subsets',s,'--split','train','--out_dir',str(r/'calibration')]),
    (r,'predict_val',['baselines_clean/predict.py','--mode',m,'--checkpoint',str(ck/m/'best.pth'),'--split','val','--pred_dir',str(pr/'val'),'--batch_size',str(a.batch_size),*common]),
    (r,'evaluate_physical',['scripts/run_physical_inspection.py','--mode','evaluate','--pred_root',str(pr/'val'),'--model',m,'--subsets',s,'--split','val','--profile',str(r/'calibration/peak_profile.json'),'--out_dir',str(r/'measurement')])]
 if a.plan_only:
  for r,n,c in stages: print(n,r.as_posix(),' '.join(map(str,c)))
  return
 if not a.resume:
  for r in {item[0] for item in stages}:
   if r.exists() and any(r.iterdir()): raise FileExistsError(f'{r} exists; use --resume or a new --run-root')
 for r,n,c in stages:
  marker=r/(n+'.done.json'); r.mkdir(parents=True,exist_ok=True)
  if a.resume and marker.exists(): print('skip',marker); continue
  if a.resume and n=='train':
   latest=Path(c[c.index('--save_dir')+1])/c[c.index('--mode')+1]/'latest.pth'
   if latest.exists(): c += ['--resume',str(latest)]
  if a.resume and n in ('predict_train','predict_val','calibrate','evaluate_physical'):
   key='--pred_dir' if n.startswith('predict') else '--out_dir'
   directory=Path(c[c.index(key)+1])
   if directory.exists(): directory.rename(directory.with_name(directory.name+'.incomplete.'+str(time.time_ns())))
  print('START',n,r,flush=True); t=time.time()
  with (r/(n+'.console.log')).open('a',encoding='utf8') as log:
   q=subprocess.Popen([sys.executable,'-u',*map(str,c)],cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace')
   for line in q.stdout: print(line,end='',flush=True); log.write(line); log.flush()
   if q.wait(): raise SystemExit(f'{n} failed; fix the error and use --resume')
  marker.write_text(json.dumps({'command':list(map(str,c)),'seconds':time.time()-t},indent=2),encoding='utf8')
 print('External experiment complete:',a.run_root)
if __name__=='__main__': main()
