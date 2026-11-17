"""Sparse RAP equivalence/cost probe. Shared host; never creates a dense matrix."""
import gc,json,os,time,statistics
import torch
from solver.sparse_product import sparse_product

def rss():
    return int(open('/proc/self/statm').read().split()[1])*os.sysconf('SC_PAGE_SIZE')

rows=[]
for side in (32,128,256):
    n=side*side
    i=torch.arange(n)
    rr=[i];cc=[i];vv=[torch.full((n,),4.,dtype=torch.float64)]
    for offset in (1,side):
        a=i[:-offset];b=i[offset:]
        if offset==1:
            m=a//side==b//side;a,b=a[m],b[m]
        rr.extend([a,b]);cc.extend([b,a]);vv.extend([-torch.ones(len(a),dtype=torch.float64)]*2)
    A=torch.sparse_coo_tensor(torch.stack([torch.cat(rr),torch.cat(cc)]),torch.cat(vv),(n,n)).coalesce().to_sparse_csr()
    coarse=(i//side//2)*(side//2)+(i%side//2)
    P=torch.sparse_coo_tensor(torch.stack([i,coarse]),torch.ones(n,dtype=torch.float64),(n,n//4)).coalesce().to_sparse_csr()
    PT=P.transpose(0,1).to_sparse_csr()
    def native():return PT@(A@P)
    def safe():return sparse_product(PT,sparse_product(A,P))
    expected=native().to_sparse_coo().coalesce();actual=safe().to_sparse_coo().coalesce()
    # Canonical COO sorts explicit zeros and values without densification.
    expected=expected.to_sparse_csr().to_sparse_coo().coalesce()
    torch.testing.assert_close(expected.indices(),actual.indices(),rtol=0,atol=0)
    torch.testing.assert_close(expected.values(),actual.values(),rtol=1e-13,atol=1e-13)
    timing={}
    for label,fn in [('native_csr',native),('safe_coo',safe)]:
        for _ in range(3):fn()
        gc.collect();before=rss();samples=[]
        for _ in range(10):
            t=time.perf_counter();fn();samples.append(time.perf_counter()-t)
        timing[label]={'median_s':statistics.median(samples),'rss_growth_bytes':rss()-before}
    row={'cells':n,'input_nnz':A._nnz(),'coarse_nnz':actual._nnz(),'algebra_equal':True,'timing':timing}
    rows.append(row);print(json.dumps(row),flush=True)
json.dump({'scope':__doc__,'torch':torch.__version__,'threads':torch.get_num_threads(),'rows':rows},open('.scratch/sparse_rap_scaling.json','w'),indent=2)
