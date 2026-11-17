from pathlib import Path
# Packaged helper path; numerical probe unchanged.
exec(Path(__file__).with_name('probe_amg_memory.py').read_text().split('for k in range(101):')[0])
mode=__import__('sys').argv[1]
for k in range(501):
 if mode=='coo':x=torch.sparse.mm(A.to_sparse_coo().coalesce(),A.to_sparse_coo().coalesce()).to_sparse_csr()
 elif mode=='csrvec':x=torch.sparse.mm(A,torch.ones((n,1),dtype=torch.float64))
 else:x=torch.sparse.mm(A.to_sparse_coo().coalesce(),torch.ones((n,1),dtype=torch.float64))
 del x
 if k%100==0:
  gc.collect();print(json.dumps({'iteration':k,'rss':rss()}),flush=True)
