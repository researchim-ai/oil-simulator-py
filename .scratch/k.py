import sys;sys.path.insert(0,'src')
import torch
from simulator.eclipse_deck import EclipseCase
_,a,_,_=EclipseCase('tests/data/spe1/SPE1CASE1.DATA').build()
_,b,_,_=EclipseCase('.scratch/e.DATA').build()
n=a.s_w.numel();torch.manual_seed(0)
sw=torch.rand(n,dtype=torch.float64)*0.8+0.1;sg=torch.rand(n,dtype=torch.float64)*(1-sw)
sw.requires_grad_();sg.requires_grad_()
for i,(x,y) in enumerate(zip(a.blackoil_relperm(sw,sg),b.blackoil_relperm(sw,sg))):
  gx=torch.autograd.grad(x.sum(),(sw,sg),allow_unused=True,retain_graph=True);gy=torch.autograd.grad(y.sum(),(sw,sg),allow_unused=True,retain_graph=True)
  print(i,float((x-y).abs().max()),[None if u is None or v is None else float((u-v).abs().max()) for u,v in zip(gx,gy)])
pa=a.blackoil_relperm.pc(sw,sg);pb=b.blackoil_relperm.pc(sw,sg)
print([float((u-v).abs().max()) for u,v in zip(pa,pb)])
print((a.s_w-b.s_w).abs().max(),(a.s_g-b.s_g).abs().max())
