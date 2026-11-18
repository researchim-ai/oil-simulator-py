import json,sys
from pathlib import Path
sys.path[:0]=['src','scripts']
from check_facility_refinement_pair import completed
from compare_fim_facility_forecast import compare as fields
from check_long_facility_trajectory import compare as wells
from check_facility_capacity_search import FIELD_BUDGETS,WELL_BUDGETS
w=Path('.scratch/capacity_warm')
a,b=[completed(w/f'startup_{name}',.00005) for name in ['reference','candidate']]
assert a[1]['config']==b[1]['config']
f=fields(a[2],b[2],FIELD_BUDGETS);q=wells(a[2],b[2]);q['budgets']=WELL_BUDGETS;q['accepted']=q['mode_mismatches']==0 and all(q['errors'][k]<=v for k,v in WELL_BUDGETS.items())
r={'accepted':f['accepted'] and q['accepted'],'fields':f,'wells':q,'reference_work':a[1]['result']['temporal_work'],'candidate_work':b[1]['result']['temporal_work'],'reference_wall_s':a[0]['solve_wall_s'],'candidate_wall_s':b[0]['solve_wall_s']}
(w/'comparison.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
