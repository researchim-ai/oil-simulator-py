import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=Path('benchmarks/reports/surface_forecast_profiles')
r=json.loads((p/'qualification.json').read_text())
fig,axs=plt.subplots(1,2,figsize=(12,5.8),layout='constrained')
for ax,case,title in zip(axs,['depletion','large'],['Сильное истощение · 1024 ячейки · CPU','Слабое истощение · 65 536 ячеек · GPU']):
 runs={x['level']:x for x in r['runs'] if x['case']==case}
 checks=r['checks_against_finest'][case]
 for check in checks:
  score=max([v/check['final_fields']['budgets'][k] for k,v in check['final_fields']['errors'].items()]+[v/check['well_trajectories']['budgets'][k] for k,v in check['well_trajectories']['errors'].items()])
  assert check['well_trajectories']['mode_mismatches']==0
  row=runs[check['candidate_level']];accepted=check['accepted']
  ax.scatter(row['solve_wall_s'],score,s=85,c='#087f73' if accepted else '#b33f36',marker='o' if accepted else 'x',linewidths=2,zorder=3)
  ax.annotate(f"{row['max_step_days']:g} сут.",(row['solve_wall_s'],score),xytext=(7, -18 if case == 'large' and row['level'] == 1 else 7),textcoords='offset points',fontsize=10)
 ref=runs[max(runs)]
 ax.axvline(ref['solve_wall_s'],color='#9199a0',ls=':',lw=1.2)
 ax.text(ref['solve_wall_s'],.97,f"Контроль\n{ref['solve_wall_s']:.0f} с",ha='right',va='top',transform=ax.get_xaxis_transform(),fontsize=9,color='#606a73')
 ax.axhline(1,color='#465563',lw=1.3,ls='--')
 ax.set_yscale('log');ax.set_ylim(.1,20);ax.set_xlim(0,ref['solve_wall_s']*1.08)
 ax.set_title(title,fontsize=12,loc='left',pad=16)
 ax.set_xlabel('Время интегрирования 30 суток, с',fontsize=10)
 ax.set_ylabel('Максимальное отношение ошибки к цели',fontsize=10)
 ax.grid(axis='y',which='major',alpha=.18)
 ax.spines[['top','right']].set_visible(False)
 ax.tick_params(labelsize=9)
fig.suptitle('Быстрый профиль принимается только после уточнения шага',fontsize=15,x=.02,ha='left')
fig.supxlabel('Ниже пунктирной линии — все цели выполнены. Подписи точек — максимальный шаг.\nВремя на общей машине; модели синтетические, площадка квазистационарная.',fontsize=10)
fig.savefig(p/'accuracy_cost.png',dpi=180)
fig.savefig(p/'accuracy_cost.svg')
