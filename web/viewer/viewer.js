const $ = s => document.querySelector(s);
const esc = v => String(v ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = (v, n=3) => Number.isFinite(v) ? (v !== 0 && Math.abs(v) < .001 ? v.toExponential(3) : v.toLocaleString('ru-RU',{maximumFractionDigits:n})) : '—';
const colors={oil:'#e9a864',water:'#61b2cf',gas:'#a4b99b',mixed:'#9fb7ad'};
let data, index=0, selected='facility', view='network', timer=null, entities=[], fieldBounds;
const frame=()=>data.frames[index];
const measure=(f,k)=>f.facility.measurements[k]?.value;
const sum=a=>a.reduce((x,y)=>x+y,0);
const vec=o=>['oil','water','gas'].map(k=>o[k]);
function validate(d){
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
function load(d){
 validate(d); stop();data=d;index=d.frames.length-1;selected='facility';
 entities=[{id:'facility',name:'Вся площадка',kind:'facility',group:'Система',icon:'◇'},
 ...d.topology.wells.map(w=>({id:'well:'+w.name,name:w.name,kind:'well',key:w.name,group:'Скважины',icon:'⊙'})),
 ...d.topology.gathering.pipes.map(p=>({id:'pipe:'+p.name,name:p.name,kind:'pipe',key:p.name,group:'Трубопроводы',icon:'⌁'})),
 ...d.topology.facility.separator_stages.map(s=>({id:'stage:'+s.name,name:s.name,kind:'stage',key:s.name,group:'Оборудование',icon:'▱'})),
 {id:'gas',name:'Компримирование и экспорт',kind:'gas',group:'Оборудование',icon:'▷'},
 ...d.topology.facility.water_network.pumps.map(p=>({id:'pump:'+p.name,name:p.name,kind:'pump',key:p.name,group:'Оборудование',icon:'◉'})),
 {id:'valve',name:'Экспортный газовый клапан',kind:'valve',group:'Оборудование',icon:'⋈'}];
 $('#case-title').textContent=d.title;$('#dataset-info').textContent=`${d.field.dimensions.join(' × ')} ячеек · ${d.topology.wells.length} скважин\n${fmt(d.frames.at(-1).time_days)} суток · синтетическая модель`;
 $('#audit-status').textContent='Архив · аудит пройден';$('#timeline').max=d.frames.length-1;$('#timeline').disabled=false;$('#play').disabled=false;$('#export').disabled=false;
 $('#time-start').textContent=fmt(d.frames[0].time_days,6)+' сут';$('#time-end').textContent=fmt(d.frames.at(-1).time_days)+' сут';
 $('#layer').innerHTML=Array.from({length:d.field.dimensions[2]},(_,i)=>`<option value="${i}">${i+1}</option>`).join('');
 $('#object-count').textContent=entities.length;$('#error').hidden=true;renderObjects();render();
}
function renderObjects(){
 if(!data)return;const query=$('#search').value.toLowerCase();let group='';
 $('#objects').innerHTML=entities.filter(e=>e.name.toLowerCase().includes(query)||e.group.toLowerCase().includes(query)).map(e=>{
 const heading=e.group!==group?`<div class="object-group">${esc(e.group)}</div>`:'';group=e.group;
 return heading+`<button class="object-button ${selected===e.id?'active':''}" data-object="${esc(e.id)}"><span class="icon">${esc(e.icon)}</span>${esc(e.name)}<small>${e.kind==='well'?esc(frame().wells[e.key]?.mode):''}</small></button>`;
 }).join('');
}
function select(id){selected=id;renderObjects();render();}
function entity(){if(selected.startsWith('cell:'))return {id:selected,kind:'cell',name:'Ячейка '+selected.slice(5).replaceAll(':',', '),group:'Пласт'};return entities.find(e=>e.id===selected)||entities[0];}
function kv(label,value,unit=''){return `<div class="kv"><span>${esc(label)}</span><strong>${esc(value)} <small>${esc(unit)}</small></strong></div>`;}
function section(title,content){return `<section class="detail-section"><h3>${esc(title)}</h3>${content}</section>`;}
function composition(q){const total=sum(q.map(Math.abs));return `<div class="composition">${q.map((v,i)=>`<span style="width:${total?Math.abs(v)/total*100:0}%;background:${Object.values(colors)[i]}"></span>`).join('')}</div>`+q.map((v,i)=>kv(['Нефть','Вода','Газ'][i],fmt(v,6),'кг/с')).join('');}
function render(){
 if(!data)return;const f=frame();$('#timeline').value=index;$('#time').textContent=fmt(f.time_days,6);$('#frame-label').textContent=`КАДР ${index+1} / ${data.frames.length}`;
 $('#metrics').innerHTML=[['Отбор нефти',measure(f,'product:oil:mass_kg_s'),'кг/с','Масса нефтяного компонента'],['Экспорт газа',measure(f,'gas_export_mass_kg_s'),'кг/с','Газовый компонент'],['Мощность приводов',measure(f,'facility_drive_power_total_w')/1000,'кВт','Насосы + компрессоры'],['Использование клапана',measure(f,'gas_valve_utilization')*100,'%','Экспортное ограничение']].map((m,i)=>`<div class="metric ${i===3?'accent':''}"><label>${m[0]}</label><strong>${fmt(m[1],3)}<small>${m[2]}</small></strong><p>${m[3]}</p></div>`).join('');
 if(view==='network')renderNetwork();if(view==='reservoir')renderField();if(view==='balances')renderBalances();if(view==='diagnostics')renderDiagnostics();renderInspector();renderCharts();
}
function node(id,x,y,w,h,title,value,kind='rect'){
 const shape=kind==='vessel'?`<rect class="body" x="${-w/2}" y="${-h/2}" width="${w}" height="${h}" rx="${h/2}"/>`:kind==='pump'?`<circle class="body" r="${h/2}"/><path d="M-9,-12 L12,0 L-9,12 Z" fill="none" stroke="#94b6a5"/>`:`<rect class="body" x="${-w/2}" y="${-h/2}" width="${w}" height="${h}" rx="5"/>`;
 return `<g class="schematic-node ${selected===id?'selected':''}" data-object="${esc(id)}" tabindex="0" role="button" aria-label="${esc(title)}" transform="translate(${x},${y})">${shape}<text text-anchor="middle" class="node-title" y="${kind==='pump'?h/2+18:-2}">${esc(title)}</text><text text-anchor="middle" class="node-value" y="${kind==='pump'?h/2+33:15}">${esc(value)}</text></g>`;
}
function flow(path,color,id,moving=true){return `<path d="${path}" stroke="${color}" class="flow ${moving?'moving':''}"/>${id?`<path d="${path}" class="pipe-hit" data-object="${esc(id)}" tabindex="0" role="button" aria-label="${esc(id)}"/>`:''}`;}
function renderNetwork(){
 const f=frame(), wells=data.topology.wells,n=wells.length, rows=Math.ceil(n/4), gap=Math.min(86,340/Math.max(rows-1,1));let paths='',nodes='';
 // Exact gathering connections, with a schematic layout independent of lengths/elevations.
 const positions={receiver:[540,270],header:[435,270]};
 wells.forEach((w,i)=>{positions[data.topology.well_nodes[w.name]]=[65+(i%4)*95,105+Math.floor(i/4)*gap];});
 const net=data.topology.gathering;
 // Intermediate junctions are explicit objects; layout is only a drawing aid.
 net.nodes.filter(n=>!positions[n.name]).forEach((n,i)=>{positions[n.name]=[460,100+i*45];});
 for(const p of net.pipes){const a=positions[p.start],b=positions[p.end];if(!a||!b)continue;const r=f.pipes[p.name];const moving=r&&sum(Object.values(r.component_mass_rates_kg_s).map(Math.abs))>1e-14;paths+=flow(`M${a[0]+25},${a[1]} H${b[0]-24} V${b[1]} H${b[0]}`,colors.mixed,'pipe:'+p.name,moving);}
 for(const w of wells){const pos=positions[data.topology.well_nodes[w.name]],r=f.wells[w.name];nodes+=node('well:'+w.name,...pos,70,44,w.name,r?`${fmt(r.bhp_pa/1e6,3)} МПа`:'нет данных');}
 paths+=flow('M435,80 V435',colors.mixed,'pipe:collector');nodes+=`<circle cx="435" cy="270" r="5" fill="#a9beb4"/><text x="423" y="461" class="diagram-label" text-anchor="end">КОЛЛЕКТОР</text>`;
 const stages=f.facility.stages;
 const first=stages[0], second=stages[1];
 if(first)nodes+=node('stage:'+first.name,580,270,125,72,first.name,`${fmt(first.pressure_pa/1e6)} МПа`,'vessel');
 if(second){paths+=flow('M642,280 H690 V335 H727',colors.oil);nodes+=node('stage:'+second.name,790,335,125,66,second.name,`${fmt(second.pressure_pa/1e6)} МПа`,'vessel');paths+=flow('M852,335 H970',colors.oil);}
 paths+=flow('M580,234 V120 H707',colors.gas);paths+=flow('M790,302 V170 H760 V150',colors.gas);paths+=flow('M813,120 H910',colors.gas,'valve');
 nodes+=node('gas',760,120,108,59,'Компрессоры',`${fmt(measure(f,'gas_drive_power_w'))} Вт`);
 nodes+=node('valve',945,120,100,54,'Газ · экспорт',`${fmt(measure(f,'gas_export_pressure_pa')/1e6)} МПа`);
 const pump=data.topology.facility.water_network.pumps[0];
 if(pump){paths+=flow('M580,306 V430 H730',colors.water);paths+=flow('M790,430 H970',colors.water);nodes+=node('pump:'+pump.name,760,430,60,60,'Насос воды',`${fmt(f.facility.network.pump_head_gains_m[pump.name])} м`,'pump');}
 nodes+=`<text x="970" y="321" text-anchor="end" class="diagram-label">НЕФТЬ · ПРОДУКТ</text><text x="970" y="416" text-anchor="end" class="diagram-label">ВОДА · ОТВОД</text><text x="40" y="58" class="diagram-label">СКВАЖИНЫ / ЗАБОЙНОЕ ДАВЛЕНИЕ</text><text x="560" y="58" class="diagram-label">ПОДГОТОВКА И ЭКСПОРТ</text>`;
 $('#network').innerHTML=paths+nodes;
}
function renderInspector(){
 const f=frame(),e=entity();let content='',badge='';
 if(e.kind==='cell'){
  const [i,j,k]=selected.slice(5).split(':').map(Number),[,ny,nz]=data.field.dimensions,n=(i*ny+j)*nz+k,p=data.field;badge=`Конечный снимок · ${fmt(p.time_days)} сут`;
  content=section('Состояние ячейки',kv('Давление',fmt(p.pressure_pa[n]/1e6,7),'МПа')+kv('Sw',fmt(p.s_w[n],7),'доля')+kv('Sg',fmt(p.s_g[n],7),'доля')+kv('So',fmt(1-p.s_w[n]-p.s_g[n],7),'доля')+kv('Rs',fmt(p.rs[n],7),'м³/м³'))+`<p class="detail-note">Индексы i, j, k начинаются с нуля. История полей в этом архиве отсутствует.</p>`;
 }else if(e.kind==='well'){
  const w=f.wells[e.key], config=data.topology.wells.find(w=>w.name===e.key);
  badge=w.mode==='rate'?'Управление дебитом':w.mode==='surface'?'Давление сети':w.mode;
  content=section('Рабочая точка',kv('Забойное давление',fmt(w.bhp_pa/1e6,6),'МПа')+kv('Давление сети',fmt(w.network_reference_pressure_pa/1e6,6),'МПа')+kv('Управляющий дебит',fmt(w.standard_control_rate_m3_s*86400,6),'м³/сут'))+section('Компонентные расходы',composition(w.component_mass_rates_kg_s))+section('Привязка',kv('Узел сети',w.node)+kv('Ячейка i, j, k',[config.i,config.j,config.k].join(', ')));
 }else if(e.kind==='pipe'){
  const p=f.pipes[e.key], config=data.topology.gathering.pipes.find(p=>p.name===e.key);badge='Установившийся поток';
  content=section('Давление на концах',kv('Вход',fmt(p.inlet_pressure_pa/1e6,6),'МПа')+kv('Выход',fmt(p.outlet_pressure_pa/1e6,6),'МПа')+kv('Δp (вход − выход)',fmt(p.inlet_pressure_pa-p.outlet_pressure_pa,6),'Па'))+section('Геометрия',kv('Длина',fmt(config.length_m),'м')+kv('Диаметр',fmt(config.diameter_m*1000),'мм'))+section('Компонентные расходы',composition(vec(p.component_mass_rates_kg_s)))+section('Фазы на выходе',Object.entries(p.outlet_state.phase_volume_rates_m3_s).map(([i,v])=>kv(['Нефть','Вода','Газ'][i],fmt(v,7),'м³/с')).join(''))+`<p class="detail-note">Профиль между концами не сохранён. Пространственная интерполяция не выполняется.</p>`;
 }else if(e.kind==='stage'){
  const s=f.facility.stages.find(s=>s.name===e.key);badge='Квазистационарная ступень';
  content=section('Сепарация',kv('Давление',fmt(s.pressure_pa/1e6),'МПа')+kv('Растворённый газ Rs',fmt(s.dissolved_gas_ratio_m3_m3),'м³/м³'))+section('Фазовые объёмные расходы',Object.entries(s.phase_volume_rates_m3_s).map(([k,v])=>kv({oil:'Нефть',water:'Вода',gas:'Газ'}[k],fmt(v,7),'м³/с')).join(''))+section('Плотности фаз',Object.entries(s.phase_density_kg_m3).map(([k,v])=>kv({oil:'Нефть',water:'Вода',gas:'Газ'}[k],fmt(v),'кг/м³')).join(''))+`<p class="detail-note">Уровни и геометрия сосуда в этом расчёте не моделируются. Их значений в архиве нет.</p>`;
 }else if(e.kind==='pump'){
  const n=f.facility.network,k=e.key;badge='Рабочая характеристика';
  content=section('Рабочая точка',kv('Расход',fmt(n.pump_volume_rates_m3_s[k]*3600,6),'м³/ч')+kv('Напор',fmt(n.pump_head_gains_m[k]),'м')+kv('Мощность на валу',fmt(n.pump_shaft_power_w[k]),'Вт')+kv('КПД',fmt(n.pump_efficiencies[k]*100),'%'))+section('Кавитационный запас',kv('NPSH доступный',fmt(n.pump_npsh_available_m[k]),'м')+kv('NPSH требуемый',fmt(n.pump_npsh_required_m[k]),'м')+kv('Запас',fmt(n.pump_npsh_margin_m[k]),'м'))+pumpCurve(k,n);
 }else if(e.kind==='valve'){
  const v=f.facility.gas_export.valve;badge=v.nozzle?.regime==='choked'?'Критическое истечение':v.nozzle?.regime||'Газовый клапан';
  content=section('Экспортное ограничение',kv('Открытие',fmt(v.opening*100,5),'%')+kv('Загрузка',fmt(v.utilization*100,5),'%')+kv('Пропускная способность',fmt(v.capacity_kg_s,8),'кг/с')+kv('Фактический расход',fmt(v.inlet.mass_rate_kg_s,8),'кг/с'))+section('Газовое сопло',kv('Давление перед клапаном',fmt(v.inlet.pressure_pa/1e6),'МПа')+kv('Давление после',fmt(v.outlet.pressure_pa/1e6),'МПа')+kv('Температура в горле',fmt(v.nozzle?.throat_temperature_k),'K')+kv('Число Маха в горле',fmt(v.nozzle?.throat_mach)));
 }else if(e.kind==='gas'){
  const g=f.facility.gas_export;badge='Сухой идеальный газ';
  content=section('Общая нагрузка',kv('Массовый расход',fmt(g.mass_rate_kg_s,7),'кг/с')+kv('Мощность приводов',fmt(g.drive_power_w),'Вт')+kv('Теплоотвод',fmt(g.rejected_heat_w),'Вт'));
  for(const train of g.routes||[])content+=section('Ветка '+train.source,(train.equipment||[]).map(x=>`<details><summary>${esc(x.name)}</summary>${kv('Давление на выходе',fmt(x.outlet.pressure_pa/1e6),'МПа')+kv('Температура на выходе',fmt(x.outlet.temperature_k),'K')+kv('Мощность на валу',fmt(x.shaft_power_w),'Вт')+kv('Теплоотвод',fmt(x.rejected_heat_w),'Вт')}</details>`).join(''));
 }else{
  badge=f.facility.feasible?'Ограничения соблюдены':'Есть нарушения';
  content=section('Связанное решение',kv('Множитель дебита',fmt(f.control.fraction,8))+kv('Шаг интегрирования',fmt(f.dt_s),'с')+kv('Итерации Ньютона',f.newton_iterations)+kv('Полных FIM в поиске',f.control.reservoir_evaluations))+section('Ограничения площадки',limitsHTML(f))+`<p class="detail-note">Загрузка газового клапана близка к 100%. Подробности — в инспекторе клапана.</p>`;
 }
 $('#inspector').innerHTML=`<span class="object-type">${esc(e.group)}</span><h2>${esc(e.name)}</h2><div class="object-badge">${esc(badge)}</div>${content}`;
}
function limitsHTML(f){return f.facility.limits.map(l=>{
 const maximum=l.maximum, minimum=l.minimum;const ratio=maximum?l.value/maximum:(minimum?minimum/l.value:0);
 return `<div class="limit"><div><span>${esc(l.name)}</span><b>${fmt(l.value,6)} ${esc(l.unit)}</b></div><div class="limit-track"><i style="width:${Math.min(100,Math.max(0,ratio*100))}%;background:${!l.passed?'#cd6f50':ratio>.95?'#c89b55':'#82a283'}"></i></div><small>${maximum!==null?'≤ '+fmt(maximum):'≥ '+fmt(minimum)} ${esc(l.unit)} · ${l.passed?'соблюдено':'нарушено'}</small></div>`;
 }).join('');}
function pumpCurve(k,n){
 const c=data.topology.facility.water_network.pumps.find(p=>p.name===k)?.head_curve;if(!c)return '';
 const maxQ=Math.max(...c.flow_m3_s),maxH=Math.max(...c.values),x=v=>15+v/maxQ*200,y=v=>95-v/maxH*80;
 return section('Заданная кривая Q–H',`<svg class="pump-curve" viewBox="0 0 230 120"><path d="M15,10 V95 H220" stroke="#d7ded7" fill="none"/><polyline points="${c.flow_m3_s.map((q,i)=>`${x(q)},${y(c.values[i])}`).join(' ')}" fill="none" stroke="#67916d" stroke-width="2"/><circle cx="${x(n.pump_volume_rates_m3_s[k])}" cy="${y(n.pump_head_gains_m[k])}" r="4" fill="#d49b53"/><text x="15" y="112">0</text><text x="148" y="112">${fmt(maxQ*3600)} м³/ч</text><text x="20" y="14">${fmt(maxH)} м</text></svg><p class="detail-note">Ломаная соединяет заданные узлы карты. Точка — рассчитанный режим, не повторный расчёт характеристики.</p>`);
}
function chartSVG(values,color){
 const finite=values.filter(Number.isFinite);if(!finite.length)return '<p>Нет истории</p>';
 let lo=Math.min(...finite),hi=Math.max(...finite);const span=hi-lo||Math.max(Math.abs(hi)*.02,1e-6);lo-=span*.1;hi+=span*.1;
 const times=data.frames.map(f=>f.time_days),t0=times[0],tr=times.at(-1)-t0||1;const x=i=>8+(times[i]-t0)/tr*274,y=v=>83-(v-lo)/(hi-lo)*70;
 const path=values.map((v,i)=>`${i===0?'M':'L'}${x(i).toFixed(2)},${y(v).toFixed(2)}`).join(' ');
 return `<svg viewBox="0 0 290 96" preserveAspectRatio="none" aria-label="История по сохранённым кадрам"><path d="M8,83 H282 M8,48 H282 M8,13 H282" stroke="#e7ece7" stroke-width=".7"/><path d="${path} L282,90 L8,90 Z" fill="${color}" opacity=".07"/><path d="${path}" fill="none" stroke="${color}" stroke-width="1.8"/><path d="M${x(index)},5 V89" stroke="#82968a" stroke-dasharray="2 3"/><circle cx="${x(index)}" cy="${y(values[index])}" r="3.2" fill="${color}" stroke="white" stroke-width="1.5"/></svg>`;
}
function chartSeries(){const e=entity();
 if(e.kind==='well')return [['BHP','МПа',f=>f.wells[e.key].bhp_pa/1e6,colors.mixed],['Нефть · компонент','кг/с',f=>f.wells[e.key].component_mass_rates_kg_s[0],colors.oil],['Газ · компонент','кг/с',f=>f.wells[e.key].component_mass_rates_kg_s[2],colors.gas]];
 if(e.kind==='pipe')return [['Входное давление','МПа',f=>f.pipes[e.key].inlet_pressure_pa/1e6,colors.mixed],['Перепад давления','Па',f=>f.pipes[e.key].inlet_pressure_pa-f.pipes[e.key].outlet_pressure_pa,colors.water],['Массовый расход','кг/с',f=>sum(vec(f.pipes[e.key].component_mass_rates_kg_s)),colors.oil]];
 if(e.kind==='gas')return [['Мощность приводов газа','Вт',f=>measure(f,'gas_drive_power_w'),colors.gas],['Теплоотвод газа','Вт',f=>measure(f,'gas_cooling_duty_w'),colors.water],['Экспорт газа','кг/с',f=>measure(f,'gas_export_mass_kg_s'),colors.oil]];
 if(e.kind==='pump')return [['Напор','м',f=>f.facility.network.pump_head_gains_m[e.key],colors.water],['Мощность на валу','Вт',f=>f.facility.network.pump_shaft_power_w[e.key],colors.oil],['Запас NPSH','м',f=>f.facility.network.pump_npsh_margin_m[e.key],colors.gas]];
 if(e.kind==='stage')return ['oil','water','gas'].map(k=>[{'oil':'Нефтяная фаза','water':'Водная фаза','gas':'Газовая фаза'}[k],'м³/с',f=>f.facility.stages.find(s=>s.name===e.key).phase_volume_rates_m3_s[k],colors[k]]);
 if(e.kind==='valve')return [['Загрузка клапана','%',f=>measure(f,'gas_valve_utilization')*100,colors.gas],['Расход газа','кг/с',f=>measure(f,'gas_export_mass_kg_s'),colors.oil],['Открытие клапана','%',f=>measure(f,'gas_valve_opening')*100,colors.water]];
 return [['Нефть · компонент','кг/с',f=>measure(f,'product:oil:mass_kg_s'),colors.oil],['Мощность приводов','кВт',f=>measure(f,'facility_drive_power_total_w')/1000,colors.water],['Множитель дебита','доля',f=>f.control.fraction,colors.gas]];
}
function renderCharts(){
 if(entity().kind==='cell'){$('#trend-title').textContent=entity().name;$('#charts').innerHTML='<p class="no-history">Для ячеек сохранён только конечный снимок. Временная история доступна для скважин, труб и оборудования.</p>';return;}
 $('#trend-title').textContent=entity().name;$('#charts').innerHTML=chartSeries().map(([title,unit,get,color])=>{const values=data.frames.map(get);return `<div class="chart-card"><label>${esc(title)}</label><div class="chart-value">${fmt(values[index],6)} <small>${esc(unit)}</small></div>${chartSVG(values,color)}<div class="chart-axis"><span>${fmt(data.frames[0].time_days,6)} сут</span><span>${fmt(data.frames.at(-1).time_days)} сут</span></div></div>`;}).join('');
}
function fieldValues(){const key=$('#field').value;return key==='s_o'?data.field.s_w.map((w,i)=>1-w-data.field.s_g[i]):data.field[key].map(v=>key==='pressure_pa'?v/1e6:v);}
function fieldColor(t){const stops=[[32,60,90],[66,141,157],[142,191,157],[237,215,140]],p=Math.max(0,Math.min(1,t))*3,i=Math.min(2,Math.floor(p));return `rgb(${stops[i].map((v,k)=>Math.round(v+(stops[i+1][k]-v)*(p-i))).join(',')})`;}
function renderField(){
 const [nx,ny,nz]=data.field.dimensions,k=Number($('#layer').value),values=fieldValues(),slice=[];for(let i=0;i<nx;i++)for(let j=0;j<ny;j++)slice.push(values[(i*ny+j)*nz+k]);const min=slice.reduce((a,b)=>Math.min(a,b),Infinity),max=slice.reduce((a,b)=>Math.max(a,b),-Infinity);
 const canvas=$('#field-canvas'),ctx=canvas.getContext('2d');const W=900,H=640;canvas.width=W;canvas.height=H;ctx.clearRect(0,0,W,H);const size=Math.min((W-100)/nx,(H-60)/ny),x0=(W-size*nx)/2,y0=(H-size*ny)/2;
 for(let i=0;i<nx;i++)for(let j=0;j<ny;j++){ctx.fillStyle=fieldColor((values[(i*ny+j)*nz+k]-min)/(max-min||1));ctx.fillRect(x0+i*size,y0+(ny-1-j)*size,size+.2,size+.2);}
 if(selected.startsWith('cell:')){const [i,j,sk]=selected.slice(5).split(':').map(Number);if(sk===k){ctx.strokeStyle='#fff7da';ctx.lineWidth=3;ctx.strokeRect(x0+i*size,y0+(ny-1-j)*size,size,size);}}
 ctx.font='18px Plex';ctx.fillStyle='#a6b9b3';ctx.fillText('J ↑',x0-45,y0+20);ctx.fillText('I →',x0+nx*size-35,y0+ny*size+28);
 for(const w of data.topology.wells){const p=w.location||w.position||[w.i,w.j,w.k];if(!p||p[2]!==k)continue;const x=x0+(p[0]+.5)*size,y=y0+(ny-p[1]-.5)*size;ctx.beginPath();ctx.arc(x,y,selected==='well:'+w.name?8:5,0,2*Math.PI);ctx.fillStyle='#172b32';ctx.fill();ctx.strokeStyle='#f0e4b5';ctx.lineWidth=2;ctx.stroke();ctx.fillStyle='#f8f2da';ctx.font='14px Plex';ctx.fillText(w.name,x+8,y-6);}
 fieldBounds={x0,y0,size,W,H,nx,ny,nz,k,values};$('#field-min').textContent=fmt(min,6);$('#field-max').textContent=fmt(max,6);$('#field-time').textContent=`Снимок: ${fmt(data.field.time_days)} сут · конечное поле`;
}
function renderBalances(){
 const t=data.totals;$('#balance-content').innerHTML=`<div class="balance-grid">${['Нефть','Вода','Газ'].map((name,i)=>`<div class="balance-card"><h3>${name} · компонент</h3><p>Начальный запас<br><b>${fmt(t.initial_component_mass_kg[i])}</b> кг</p><p>Конечный запас<br><b>${fmt(t.final_component_mass_kg[i])}</b> кг</p><p>Накопленный отбор<br><b>${fmt(t.withdrawn_component_mass_kg[i])}</b> кг</p><hr><p>Невязка баланса<br><b>${t.mass_balance_error_kg[i].toExponential(3)}</b> кг</p></div>`).join('')}</div><div class="balance-note"><strong>Интегральный баланс за весь расчёт · ${fmt(data.field.time_days)} суток.</strong><br>Начальный запас − конечный запас − накопленный отбор. Этот итог не перематывается вместе с историей сети.<br>Материальный аудит пройден. Это проверка сохранения массы, а не независимая экспериментальная валидация физических замыканий.</div>`;
}
function switchView(v){view=v;for(const name of ['network','reservoir','balances','diagnostics'])$('#'+name+'-view').hidden=name!==v;document.querySelectorAll('[data-view]').forEach(b=>{b.classList.toggle('active',b.dataset.view===v);b.setAttribute('aria-selected',b.dataset.view===v);});$('#stage-title').textContent={network:'От скважины до экспорта',reservoir:'Поле пласта · ячейка за ячейкой',balances:'Проверка сохранения компонентов',diagnostics:'Как найдено принятое решение'}[v];$('#stage-eyebrow').textContent={network:'СВЯЗАННАЯ СИСТЕМА',reservoir:'КОНЕЧНЫЙ СНИМОК',balances:'АУДИТ ВСЕГО ГОРИЗОНТА',diagnostics:'ДИАГНОСТИКА ОГРАНИЧИТЕЛЯ'}[v];$('#stage-note').textContent=v==='network'?'Схема связей · не географическая карта':v==='reservoir'?'Значения в ячейках · без сглаживания':'Из архивного расчёта';render();}
function stop(){clearInterval(timer);timer=null;$('#play').textContent='▶';$('#play').setAttribute('aria-label','Воспроизвести историю');}
function play(){if(!data)return;if(timer){stop();return;}if(index===data.frames.length-1)index=0;$('#play').textContent='Ⅱ';$('#play').setAttribute('aria-label','Остановить воспроизведение');render();timer=setInterval(()=>{if(index>=data.frames.length-1){stop();return;}index++;render();},Number($('#speed').value));}
function showError(e){$('#error').hidden=false;$('#error').textContent=e.message+' Можно открыть подготовленный JSON кнопкой «Открыть расчёт».';}
function sourceInfo(){if(!data)return;const p=data.provenance;$('#source-info').innerHTML=`<p><strong>Синтетическая связанная модель, не Norne.</strong> Изотермический black-oil пласт, однородная равновесная модель сборных труб, квазистационарная подготовка и экспорт.</p><p>Архив: <code>${esc(p.run)}</code><br>Устройство расчёта: ${esc(p.device)} · время: ${fmt(p.solve_wall_s)} с<br>Принятых стадий: ${p.accepted_steps}; кадров в просмотре: ${data.frames.length}.</p><p>Каждый кадр — реальное принятое состояние. Промежуточные значения не вычисляются. Соединяющие линии графиков служат навигацией по сохранённым кадрам; скорость воспроизведения не показывает стоимость симуляции.</p><p>Время кадра — конец интеграционного шага. Дебиты — компонентные или фазовые, как подписано; это разные величины. Уровни сепараторов, температурное поле пласта и подробная динамика труб в этом архиве отсутствуют.</p><p>Поля пласта и интегральные балансы относятся только к концу расчёта. Схема показывает связи и не задаёт географическое положение оборудования.</p><p>Экспортёр проверил завершение, неизменность исходников, SHA-256 траектории и материальный аудит. Импортированный JSON отображается локально и не пересчитывает физику. Браузер не подтверждает подлинность произвольного импортированного файла.</p><p>SHA-256 исходной траектории:<br><code>${esc(p.trajectory_sha256)}</code></p>`;$('#provenance').showModal();}
function downloadCSV(){
 if(!data)return;const f=frame(),rows=[['time_days','object','kind','metric','value','unit']];
 const add=(object,kind,metric,value,unit)=>{if(value!==null&&value!==undefined)rows.push([f.time_days,object,kind,metric,value,unit]);};
 for(const [name,w]of Object.entries(f.wells)){
  add(name,'well','bhp',w.bhp_pa,'Pa');add(name,'well','network_pressure',w.network_reference_pressure_pa,'Pa');
  w.component_mass_rates_kg_s.forEach((q,i)=>add(name,'well',['oil_component','water_component','gas_component'][i],q,'kg/s'));
 }
 for(const [name,p]of Object.entries(f.pipes)){
  add(name,'pipe','inlet_pressure',p.inlet_pressure_pa,'Pa');add(name,'pipe','outlet_pressure',p.outlet_pressure_pa,'Pa');
  Object.entries(p.component_mass_rates_kg_s).forEach(([k,q])=>add(name,'pipe',k+'_component',q,'kg/s'));
 }
 for(const [metric,m]of Object.entries(f.facility.measurements))add('facility','measurement',metric,m.value,m.unit);
 for(const l of f.facility.limits){add(l.name,'limit','value',l.value,l.unit);add(l.name,'limit','minimum',l.minimum,l.unit);add(l.name,'limit','maximum',l.maximum,l.unit);}
 const quote=v=>'"'+String(typeof v==='string'&&/^[=+@\-\t\r]/.test(v)?"'"+v:v).replaceAll('"','""')+'"';
 const blob=new Blob(['\uFEFF'+rows.map(r=>r.map(quote).join(',')).join('\r\n')],{type:'text/csv;charset=utf-8'});
 const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=`fieldroom-${f.time_days.toFixed(6)}d.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}

$('#objects').addEventListener('click',e=>{const b=e.target.closest('[data-object]');if(b)select(b.dataset.object);});
$('#network').addEventListener('click',e=>{const b=e.target.closest('[data-object]');if(b)select(b.dataset.object);});
$('#network').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){const b=e.target.closest('[data-object]');if(b){e.preventDefault();select(b.dataset.object);}}});
$('#search').addEventListener('input',renderObjects);document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>switchView(b.dataset.view)));
$('#timeline').addEventListener('input',e=>{stop();index=Number(e.target.value);render();});$('#play').addEventListener('click',play);$('#speed').addEventListener('change',()=>{if(timer){stop();play();}});$('#previous').addEventListener('click',()=>{if(data){stop();index=Math.max(0,index-1);render();}});$('#next').addEventListener('click',()=>{if(data){stop();index=Math.min(data.frames.length-1,index+1);render();}});$('#field').addEventListener('change',()=>data&&renderField());$('#layer').addEventListener('change',()=>data&&renderField());$('#source-button').addEventListener('click',sourceInfo);$('#close-source').addEventListener('click',()=>$('#provenance').close());$('#export').addEventListener('click',downloadCSV);
$('#field-canvas').addEventListener('mousemove',e=>{if(!fieldBounds)return;const b=fieldBounds,{i,j}=fieldPoint(e);const tip=$('#cell-tooltip');if(i<0||i>=b.nx||j<0||j>=b.ny){tip.hidden=true;return;}tip.hidden=false;tip.textContent=`i=${i} · j=${j} · k=${b.k} → ${fmt(b.values[(i*b.ny+j)*b.nz+b.k],7)} · ${$('#field').selectedOptions[0].textContent.split('·')[1]}`;});$('#field-canvas').addEventListener('mouseleave',()=>$('#cell-tooltip').hidden=true);
$('#import').addEventListener('change',async e=>{const file=e.target.files[0];if(!file)return;try{if(file.size>80*1024**2)throw Error('Файл больше 80 МиБ. Уменьшите число экспортируемых кадров.');load(JSON.parse(await file.text()));}catch(err){showError(err);}finally{e.target.value='';}});
fetch('../data/engineering-case.json').then(r=>{if(!r.ok)throw Error('Не удалось загрузить архив: HTTP '+r.status);return r.json();}).then(load).catch(showError);

function renderDiagnostics(){
 const f=frame();$('#diagnostics-content').innerHTML=`<div class="diagnostic-summary">${kv('Конец принятого шага',fmt(f.time_days,7),'сут')+kv('Длительность шага',fmt(f.dt_s,7),'с')+kv('Итерации Ньютона',f.newton_iterations)+kv('Множитель дебита',fmt(f.control.fraction,9))+kv('Физических FIM-проб',f.control.reservoir_evaluations)}</div><p class="data-note">Пробы ограничителя в выбранной принятой стадии. Недопустимые точки поиска не являются принятыми состояниями пласта.</p><div class="trace-wrap"><table class="trace-table"><thead><tr><th>Проба</th><th>Множитель</th><th>Оборудование</th><th>Расчёт</th><th>Наибольшая невязка ограничения</th></tr></thead><tbody>${f.control.trace.map((r,i)=>{const a=Object.entries(r.constraint_margins||{}).filter(([,v])=>Number.isFinite(v)).sort((a,b)=>b[1]-a[1])[0];return `<tr><td>${i+1}</td><td>${fmt(r.fraction,9)}</td><td class="${r.feasible?'ok':'warn'}">${r.feasible?'Допустимо':'Недопустимо'}</td><td>${r.certified_from_trace_index!==undefined?'Сертификат ← '+(r.certified_from_trace_index+1):'FIM'}</td><td>${a?esc(a[0])+' · '+fmt(a[1],7):'Нет значения'}</td></tr>`;}).join('')}</tbody></table></div><p class="data-note">Невязки нормированы: положительное значение означает превышение. Полные журналы итераций и локальные CNV по ячейкам в этот компактный экспорт не включены.</p>`;
}
let zoomBox=[0,0,1050,530],dragState=null;
function zoom(factor){const [x,y,w,h]=zoomBox,nw=Math.max(260,Math.min(1400,w*factor)),nh=nw*530/1050;zoomBox=[x+(w-nw)/2,y+(h-nh)/2,nw,nh];$('#network').setAttribute('viewBox',zoomBox.join(' '));}
$('#zoom-in').addEventListener('click',()=>zoom(.8));$('#zoom-out').addEventListener('click',()=>zoom(1.25));$('#zoom-reset').addEventListener('click',()=>{zoomBox=[0,0,1050,530];$('#network').setAttribute('viewBox',zoomBox.join(' '));});
$('#network').addEventListener('pointerdown',e=>{if(e.target.closest('[data-object]'))return;dragState={x:e.clientX,y:e.clientY,box:[...zoomBox]};e.target.setPointerCapture(e.pointerId);});
$('#network').addEventListener('pointermove',e=>{if(!dragState)return;const r=$('#network').getBoundingClientRect(),b=dragState.box;zoomBox=[b[0]-(e.clientX-dragState.x)*b[2]/r.width,b[1]-(e.clientY-dragState.y)*b[3]/r.height,b[2],b[3]];$('#network').setAttribute('viewBox',zoomBox.join(' '));});
$('#network').addEventListener('pointerup',()=>dragState=null);$('#network').addEventListener('pointercancel',()=>dragState=null);
$('#field-canvas').addEventListener('click',e=>{if(!fieldBounds)return;const b=fieldBounds,{i,j}=fieldPoint(e);if(i>=0&&i<b.nx&&j>=0&&j<b.ny)select(`cell:${i}:${j}:${b.k}`);});
$('#charts').addEventListener('click',e=>{const svg=e.target.closest('svg');if(!svg||!data)return;const r=svg.getBoundingClientRect(),frac=Math.max(0,Math.min(1,(e.clientX-r.left)/r.width)),t=data.frames[0].time_days+frac*(data.frames.at(-1).time_days-data.frames[0].time_days);let best=0;for(let i=1;i<data.frames.length;i++)if(Math.abs(data.frames[i].time_days-t)<Math.abs(data.frames[best].time_days-t))best=i;stop();index=best;render();});

function fieldPoint(e){
 const b=fieldBounds,r=e.target.getBoundingClientRect(),scale=Math.min(r.width/b.W,r.height/b.H);
 const ox=(r.width-b.W*scale)/2,oy=(r.height-b.H*scale)/2;
 const x=(e.clientX-r.left-ox)/scale,y=(e.clientY-r.top-oy)/scale;
 return {i:Math.floor((x-b.x0)/b.size),j:b.ny-1-Math.floor((y-b.y0)/b.size)};
}
