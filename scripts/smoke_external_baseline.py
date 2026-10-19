"""One model per process: check shapes, finite loss, and backward."""
import argparse, os, sys, json, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'baselines_clean'))
os.environ.setdefault('TORCH_HOME',str(ROOT/'vendor/torch'))
import torch
from train import build_model_clean, forward_and_loss
def main():
 p=argparse.ArgumentParser(); p.add_argument('--method',required=True,choices=['deeplab','gscnn','sctransnet','afenet']); p.add_argument('--device',default='cpu'); p.add_argument('--height',type=int,default=64); p.add_argument('--width',type=int,default=96); p.add_argument('--pretrained',action='store_true'); p.add_argument('--eval-only',action='store_true'); a=p.parse_args()
 torch.set_num_threads(4); torch.manual_seed(42); t=time.time(); device=torch.device(a.device)
 model=build_model_clean(a.method,(a.height,a.width),pretrained=a.pretrained).to(device)
 batch=dict(image=torch.randn(2,3,a.height,a.width,device=device),seg=torch.randint(0,2,(2,1,a.height,a.width),device=device).float(),edge=torch.randint(0,2,(2,1,a.height,a.width),device=device).float())
 if not a.eval_only:
  model.train(); losses,_,out=forward_and_loss(model,batch,device,a.method,10.,0.)
  assert out.shape==batch['seg'].shape; assert torch.isfinite(losses['total']); losses['total'].backward()
  assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
 model.eval()
 with torch.no_grad():
  out=model(batch['image']); out=out[-1] if isinstance(out,tuple) else out
 assert out.shape==batch['seg'].shape and torch.isfinite(out).all()
 print(json.dumps(dict(method=a.method,shape=list(out.shape),backward=not a.eval_only,pretrained=a.pretrained,seconds=time.time()-t,torch=torch.__version__)))
if __name__=='__main__':main()
