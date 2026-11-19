export function validate(d){
 if(d?.schema!=='oil-engineering-viewer/v1')throw Error('Нужен экспорт oil-engineering-viewer/v1. Используйте scripts/export_engineering_viewer.py.');
 if(!Array.isArray(d.frames)||!d.frames.length||d.frames.length>10000)throw Error('Некорректное число кадров.');
 if(d.provenance?.status!=='complete'||d.provenance?.source_guard!==true||d.provenance?.audit?.accepted!==true)throw Error('Нужен завершённый расчёт с пройденным аудитом.');
 if(d.field?.layout!=='C-order [i,j,k]; flat=(i*ny+j)*nz+k')throw Error('Неизвестный порядок ячеек.');
 const dims=d.field.dimensions;if(!Array.isArray(dims)||dims.length!==3||!dims.every(x=>Number.isInteger(x)&&x>0)||dims.reduce((a,b)=>a*b,1)>2000000)throw Error('Неподдерживаемые размеры сетки.');
 for(const key of ['pressure_pa','s_w','s_g','rs'])if(!Array.isArray(d.field[key])||d.field[key].length!==dims.reduce((a,b)=>a*b,1)||!d.field[key].every(Number.isFinite))throw Error('Некорректное поле '+key);
 if(!d.topology?.gathering?.pipes||!Array.isArray(d.topology.wells)||!d.topology.facility?.water_network)throw Error('Не поддержана топология этого экспорта.');
 const top=d.topology, plant=top.facility;
 if(top.gathering.model!=='homogeneous_blackoil_tree'||plant.separator_stages?.length!==2||plant.water_network.pumps?.length!==1||!plant.gas_export||top.wells.length<1||top.wells.length>16)throw Error('Просмотр v1 поддерживает до 16 добывающих скважин, две ступени сепарации, один насос воды и экспорт газа.');
 const expected=new Set([...Object.values(top.well_nodes).map(n=>JSON.stringify([n,'header'])),JSON.stringify(['header','receiver'])]);
 if(top.gathering.pipes.length!==expected.size||top.gathering.pipes.some(p=>!expected.has(JSON.stringify([p.start,p.end]))))throw Error('Для этой топологии нужен отдельный адаптер схемы.');
 let last=-Infinity;
 for(const f of d.frames){
  if(!Number.isFinite(f.time_days)||f.time_days<=last||!Number.isFinite(f.dt_s)||f.dt_s<=0)throw Error('Нарушен порядок времени.');last=f.time_days;
  if(!f.wells||!f.pipes||!f.control||!f.facility?.limits||!f.facility?.measurements||!f.facility?.gas_export)throw Error('Неполный кадр сети.');
  if(top.wells.some(w=>!Object.hasOwn(f.wells,w.name))||top.gathering.pipes.some(p=>!Object.hasOwn(f.pipes,p.name)))throw Error('Отсутствуют объекты в кадре.');
  for(const w of Object.values(f.wells))if(!Number.isFinite(w.bhp_pa)||!Array.isArray(w.component_mass_rates_kg_s)||w.component_mass_rates_kg_s.length!==3||!w.component_mass_rates_kg_s.every(Number.isFinite))throw Error('Некорректная история скважины.');
 }
 for(const f of d.frames){for(const w of top.wells){const p=[w.i,w.j,w.k];if(!p.every((v,i)=>Number.isInteger(v)&&v>=0&&v<dims[i]))throw Error('Скважина вне сетки.');}for(const p of top.gathering.pipes){const v=f.pipes[p.name];if(!Number.isFinite(v.inlet_pressure_pa)||!Number.isFinite(v.outlet_pressure_pa)||!v.outlet_state?.phase_volume_rates_m3_s||!v.component_mass_rates_kg_s)throw Error('Неполные данные трубы.');}if(!Array.isArray(f.facility.gas_export.routes)||!f.facility.gas_export.valve?.nozzle||!f.facility.network.pump_npsh_margin_m||!Array.isArray(f.control.trace)||top.facility.separator_stages.some(s=>!f.facility.stages.some(r=>r.name===s.name)))throw Error('Неполные данные оборудования.');}
 if(!['initial_component_mass_kg','final_component_mass_kg','withdrawn_component_mass_kg','mass_balance_error_kg'].every(k=>Array.isArray(d.totals?.[k])&&d.totals[k].length===3&&d.totals[k].every(Number.isFinite)))throw Error('Неполный материальный баланс.');
 if(d.field.time_days!==d.frames.at(-1).time_days)throw Error('Дата поля не совпадает с концом архива.');
 return d;
}
