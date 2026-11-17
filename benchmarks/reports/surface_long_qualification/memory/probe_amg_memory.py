import gc,json,torch
from solver.classical_amg import ClassicalAMG
n=1024;side=32;i=torch.arange(n)
rows=[i];cols=[i];vals=[torch.full((n,),4.,dtype=torch.float64)]
for offset in [1,side]:
 a=i[:-offset];b=i[offset:]
 if offset==1:
  mask=a//side==b//side;a,b=a[mask],b[mask]
 rows.extend([a,b]);cols.extend([b,a]);vals.extend([-torch.ones(len(a),dtype=torch.float64)]*2)
A=torch.sparse_coo_tensor(torch.stack([torch.cat(rows),torch.cat(cols)]),torch.cat(vals),(n,n)).coalesce().to_sparse_csr()
def rss():return int(open('/proc/self/statm').read().split()[1])*4096
for k in range(101):
 if __import__('sys').argv[1]=='product':x=A@A
 else:
  mg=ClassicalAMG(A,max_levels=15,theta=.25,mixed_precision=True,verbose=False);x=mg.solve(torch.ones(n,dtype=torch.float64));del mg
 del x
 if k%20==0:
  gc.collect();print(json.dumps({'iteration':k,'rss':rss()}),flush=True)
