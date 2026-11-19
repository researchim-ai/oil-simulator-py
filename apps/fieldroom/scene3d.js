// Dependency-free perspective 3D overview. Schematic coordinates are NOT survey geometry.
export class FieldScene3D {
 constructor(canvas, select) {
  this.canvas=canvas; this.ctx=canvas.getContext('2d'); this.select=select;
  this.yaw=-.65;this.pitch=.55;this.zoom=1;this.pan=[0,0];this.hits=[];
  this.field='pressure_pa';this.layer=0;this.showField=true;this.showLabels=true;
  this.drag=null;this.pending=false;
  canvas.addEventListener('pointerdown',e=>{if(e.button!==0&&e.button!==2)return;canvas.focus({preventScroll:true});this.drag={x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY,pan:e.shiftKey||e.button===2};canvas.setPointerCapture(e.pointerId);});
  canvas.addEventListener('pointermove',e=>{if(!this.drag)return;const d=this.drag,dx=e.clientX-d.x,dy=e.clientY-d.y;d.x=e.clientX;d.y=e.clientY;if(d.pan){this.pan[0]+=dx;this.pan[1]+=dy;}else{this.yaw+=dx*.006;this.pitch=Math.max(.12,Math.min(1.45,this.pitch+dy*.005));}this.request();});
  canvas.addEventListener('pointerup',e=>{const d=this.drag;this.drag=null;if(d&&Math.hypot(e.clientX-d.startX,e.clientY-d.startY)<5){const r=canvas.getBoundingClientRect();this.pick(e.clientX-r.left,e.clientY-r.top);} });
  canvas.addEventListener('pointercancel',()=>{this.drag=null;});
  canvas.addEventListener('lostpointercapture',()=>{this.drag=null;});
  canvas.addEventListener('contextmenu',e=>e.preventDefault());
  canvas.addEventListener('wheel',e=>{e.preventDefault();this.zoom=Math.max(.45,Math.min(3,this.zoom*Math.exp(-e.deltaY*.001)));this.request();},{passive:false});
  canvas.addEventListener('keydown',e=>{let used=true;if(e.key==='ArrowLeft')this.yaw-=.12;else if(e.key==='ArrowRight')this.yaw+=.12;else if(e.key==='ArrowUp')this.pitch=Math.min(1.45,this.pitch+.1);else if(e.key==='ArrowDown')this.pitch=Math.max(.12,this.pitch-.1);else if(e.key==='+'||e.key==='=')this.zoom=Math.min(3,this.zoom*1.1);else if(e.key==='-')this.zoom=Math.max(.45,this.zoom/1.1);else if(e.key.toLowerCase()==='r')this.reset();else used=false;if(used){e.preventDefault();this.request();}});
  this.resize=new ResizeObserver(()=>this.request());this.resize.observe(canvas);
 }
 reset(top=false){this.yaw=top?0:-.65;this.pitch=top?1.4:.55;this.zoom=1;this.pan=[0,0];this.request();}
 update(data,frame,selected){if(this.data!==data){this.data=data;this.layer=0;this.reset();}this.frame=frame;this.selected=selected;this.request();}
 request(){if(this.pending)return;this.pending=true;requestAnimationFrame(()=>{this.pending=false;this.draw();});}
 project([x,y,z]){const a=this.yaw,b=this.pitch;const xx=x*Math.cos(a)-z*Math.sin(a),zz=x*Math.sin(a)+z*Math.cos(a);const yy=y*Math.cos(b)-zz*Math.sin(b),depth=y*Math.sin(b)+zz*Math.cos(b);const scale=this.scale*950/(950-depth);return [this.width/2+xx*scale+this.pan[0],this.height*.39-yy*scale+this.pan[1],depth];}
 polygon(points,fill,id,stroke='#142d39',width=.6){this.items.push({points:points.map(p=>this.project(p)),fill,id,stroke,width});}
 line(points,color,id,width=2){this.items.push({points:points.map(p=>this.project(p)),id,stroke:color,width,line:true});}
 box(x,y,z,w,h,d,color,id){const p=[[x,y,z],[x+w,y,z],[x+w,y+h,z],[x,y+h,z],[x,y,z+d],[x+w,y,z+d],[x+w,y+h,z+d],[x,y+h,z+d]];for(const f of [[0,1,2,3],[4,7,6,5],[3,2,6,7],[0,4,5,1],[1,5,6,2],[0,3,7,4]])this.polygon(f.map(i=>p[i]),color,id);}
 vessel(x,y,z,length,radius,color,id){const rings=[-length/2,length/2].map(dx=>Array.from({length:14},(_,i)=>[x+dx,y+radius*Math.cos(i*Math.PI/7),z+radius*Math.sin(i*Math.PI/7)]));this.polygon(rings[0],color,id);this.polygon(rings[1],color,id);for(let i=0;i<14;i++)this.polygon([rings[0][i],rings[0][(i+1)%14],rings[1][(i+1)%14],rings[1][i]],color,id);}
 label(pos,text,id,color='#e3ece5'){this.labels.push({p:this.project(pos),text,id,color});}
 draw(){
  if(!this.data||!this.frame||!this.canvas.clientWidth||this.canvas.closest('[hidden]'))return;
  const c=this.ctx,r=this.canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2);this.width=r.width;this.height=r.height;
  this.canvas.width=Math.round(r.width*dpr);this.canvas.height=Math.round(r.height*dpr);c.setTransform(dpr,0,0,dpr,0,0);
  c.fillStyle='#142832';c.fillRect(0,0,r.width,r.height);this.scale=Math.min(r.width/720,r.height/530)*this.zoom;this.items=[];this.labels=[];this.hits=[];
  const data=this.data, f=this.frame, [nx,ny,nz]=data.field.dimensions,k=Math.min(this.layer,nz-1);
  if(this.fieldCache?.data!==data||this.fieldCache?.key!==this.field){const values=this.field==='s_o'?data.field.s_w.map((v,i)=>1-v-data.field.s_g[i]):data.field[this.field];let lo=Infinity,hi=-Infinity;for(const v of values){lo=Math.min(lo,v);hi=Math.max(hi,v);}this.fieldCache={data,key:this.field,values,lo,hi};}
  const {values,lo,hi}=this.fieldCache;
  const color=v=>{const q=hi===lo?.5:(v-lo)/(hi-lo);return `hsl(${205-165*q} 44% ${35+22*q}%)`;};
  // i,j coordinates are faithful grid indices; vertical scale and all equipment positions are illustrative.
  const point=(i,j,kk)=>[-270+(i+.5)*280/nx,-100-(kk+.5)*95/nz,-110+(j+.5)*220/ny];
  if(this.showField){const strideI=Math.max(1,Math.ceil(nx/40)),strideJ=Math.max(1,Math.ceil(ny/40));for(let i=0;i<nx;i+=strideI)for(let j=0;j<ny;j+=strideJ){const n=(i*ny+j)*nz+k,p=point(i,j,k);this.box(p[0]-140/nx,p[1]-47.5/nz,p[2]-110/ny,280/nx,95/nz,220/ny,color(values[n]),`cell:${i}:${j}:${k}`);}this.sampleStrides=[strideI,strideJ];}
  for(const y of [-100,-195])this.line([[-270,y,-110],[10,y,-110],[10,y,110],[-270,y,110],[-270,y,-110]],'#6c8b91',null,1);
  for(const x of [-270,10])for(const z of [-110,110])this.line([[x,-100,z],[x,-195,z]],'#6c8b91',null,1);
  this.label([-140,-215,120],`ПЛАСТ · K=${k+1}/${nz} · конечное поле`,null,'#a5c3c6');
  this.polygon([[45,0,-140],[300,0,-140],[300,0,160],[45,0,160]],'#243b40',null,'#46605e');
  for(let x=-280;x<=300;x+=40)this.line([[x,.2,-140],[x,.2,160]],'#304a4e',null,.5);
  for(let z=-140;z<=160;z+=40)this.line([[-290,.2,z],[300,.2,z]],'#304a4e',null,.5);
  const pos={header:[35,12,0],receiver:[100,12,0]};
  for(const w of data.topology.wells){const p=point(w.i,w.j,w.k),head=[p[0],12,p[2]],id='well:'+w.name;pos[data.topology.well_nodes[w.name]]=head;
   this.line([p,[p[0],-15,p[2]],head],'#dcbb79',id,4);this.box(p[0]-4,2,p[2]-4,8,14,8,'#c0d2b3',id);this.line([[p[0]-8,12,p[2]],[p[0]+8,12,p[2]]],'#cfdfbd',id,3);
   this.label([p[0],25,p[2]],w.name+(this.selected===id?' · '+(f.wells[w.name].bhp_pa/1e6).toFixed(3)+' МПа':''),id);
  }
  for(const p of data.topology.gathering.pipes){const a=pos[p.start],b=pos[p.end];if(!a||!b)continue;this.line([a,[b[0],a[1],a[2]],b],'#90b9ad','pipe:'+p.name,3);}
  const stages=data.topology.facility.separator_stages;
  stages.forEach((s,i)=>{const x=110+i*110,z=i*65,id='stage:'+s.name;this.vessel(x,20,z,65,15,'#bdaf82',id);this.box(x-22,1,z-4,5,10,8,'#879183',id);this.box(x+17,1,z-4,5,10,8,'#879183',id);this.label([x,48,z],s.name+' · '+(f.facility.stages[i].pressure_pa/1e6).toFixed(2)+' МПа',id);});
  this.line([[140,14,0],[165,14,0],[165,14,65],[185,14,65]],'#e9a864',null,3);
  this.line([[250,14,65],[285,14,65]],'#e9a864',null,3);this.label([285,24,65],'Нефть',null,'#e9a864');
  this.line([[110,35,0],[110,42,-75],[205,42,-75]],'#a4b99b',null,3);
  this.line([[220,35,65],[250,42,65],[250,42,-75],[205,42,-75]],'#a4b99b',null,2);
  this.box(180,3,-100,50,45,40,'#8baba2','gas');this.vessel(205,25,-75,48,13,'#759c94','gas');this.label([205,63,-80],'Компрессоры','gas');
  this.line([[230,25,-75],[285,25,-75]],'#a4b99b',null,4);this.box(260,20,-80,12,12,12,'#b1bd94','valve');this.label([278,45,-75],'Клапан','valve');
  for(const p of data.topology.facility.water_network.pumps){const id='pump:'+p.name;this.line([[110,8,0],[110,8,125],[205,8,125]],'#61b2cf',null,3);this.vessel(205,12,125,30,10,'#61b2cf',id);this.label([205,35,125],p.name,id);this.line([[220,8,125],[290,8,125]],'#61b2cf',null,3);this.label([280,20,140],'Отвод воды',null,'#61b2cf');}
  this.items.sort((a,b)=>a.points.reduce((s,p)=>s+p[2],0)/a.points.length-b.points.reduce((s,p)=>s+p[2],0)/b.points.length);
  for(const item of this.items){const pts=item.points;c.beginPath();pts.forEach((p,i)=>i?c.lineTo(p[0],p[1]):c.moveTo(p[0],p[1]));if(!item.line){c.closePath();c.fillStyle=item.fill;c.fill();}c.strokeStyle=item.id===this.selected?'#fff2b3':item.stroke;c.lineWidth=item.id===this.selected?Math.max(item.width,3):item.width;c.stroke();if(item.id)this.hits.push(item);}
  if(this.showLabels){c.font='11px monospace';c.textAlign='center';const occupied=[];for(const l of this.labels){let [x,y]=l.p;if(x<0||x>r.width||y<0||y>r.height)continue;const width=c.measureText(l.text).width;for(let tries=0;tries<12&&occupied.some(b=>x+width/2+4>b[0]&&x-width/2-4<b[2]&&y+5>b[1]&&y-12<b[3]);tries++)y-=19;occupied.push([x-width/2-4,y-12,x+width/2+4,y+5]);if(y!==l.p[1]){c.beginPath();c.moveTo(x,y+5);c.lineTo(l.p[0],l.p[1]);c.strokeStyle='#718b8a';c.lineWidth=.7;c.stroke();}c.fillStyle='#10222cd9';c.fillRect(x-width/2-4,y-12,width+8,17);c.fillStyle=l.id===this.selected?'#fff2b3':l.color;c.fillText(l.text,x,y);if(l.id)this.hits.push({id:l.id,label:[x-width/2-4,y-12,width+8,17]});}}
  const legend=document.querySelector('#scene-scale');if(legend){const unit=this.field==='pressure_pa'?'МПа':this.field==='rs'?'м³/м³':'доля';const scale=this.field==='pressure_pa'?1e6:1;legend.textContent=`Поле: ${(lo/scale).toPrecision(4)} … ${(hi/scale).toPrecision(4)} ${unit} · снимок ${data.field.time_days} сут${this.showField&&this.sampleStrides?.some(s=>s>1)?' · выборка I/J с шагом '+this.sampleStrides.join('/')+' (не усреднение)':''}`;}
  const note=document.querySelector('#scene-time');if(note)note.textContent=`Сеть: ${f.time_days.toLocaleString('ru-RU',{maximumFractionDigits:5})} сут · выбранный объект подсвечен`;
 }
 pick(x,y){for(const h of [...this.hits].reverse()){let hit=false;if(h.label){const [a,b,w,t]=h.label;hit=x>=a&&x<=a+w&&y>=b&&y<=b+t;}else if(h.line){for(let i=1;i<h.points.length;i++){const a=h.points[i-1],b=h.points[i],dx=b[0]-a[0],dy=b[1]-a[1],t=Math.max(0,Math.min(1,((x-a[0])*dx+(y-a[1])*dy)/(dx*dx+dy*dy||1)));if(Math.hypot(x-a[0]-t*dx,y-a[1]-t*dy)<6)hit=true;}}else{for(let i=0,j=h.points.length-1;i<h.points.length;j=i++){const a=h.points[i],b=h.points[j];if((a[1]>y)!==(b[1]>y)&&x<(b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0])hit=!hit;}}if(hit){this.select(h.id);return;}}}
}
