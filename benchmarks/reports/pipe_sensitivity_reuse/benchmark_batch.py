import json,time,statistics,torch
from surface.blackoil_pipe import BlackOilPipe
from surface.blackoil_pipe_batch import PipeBatch
from surface.blackoil_properties import BlackOilMixture
from surface.process import StockDensities
torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.12)
m=BlackOilMixture(json.load(open('.scratch/pipe_port_reuse/pvt.json')),StockDensities(850,1000,1.2)).to('cuda')
pipes=[BlackOilPipe(f'p{i}',300+10*(i%3),.08,30,1e-5) for i in range(64)]
batch=PipeBatch(pipes,m);boundaries=[2.72e6]*64;rates=[[.85,2.,.06]]*64
checked=batch.checked;calls=[0]
def counted(*a,**kw):calls[0]+=1;return checked(*a,**kw)
batch.checked=counted
for _ in range(3):batch.solve(boundaries,rates)
times=[];peaks=[];counts=[]
for _ in range(4):
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();base=torch.cuda.memory_allocated();calls[0]=0;start=time.perf_counter()
 for _ in range(5):rows=batch.solve(boundaries,rates)
 torch.cuda.synchronize();times.append((time.perf_counter()-start)/5);peaks.append(torch.cuda.max_memory_allocated()-base);counts.append(calls[0]/5)
print(json.dumps({'seconds_per_batch':times,'median':statistics.median(times),'peak_extra_bytes':peaks,'checked_integrals_per_solve':counts,'rows':rows},indent=2))
