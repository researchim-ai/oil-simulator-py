/* Exercise the actual import contract without a browser or simulator run. */
const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const root=path.resolve(__dirname,'..');
const fixture=JSON.parse(fs.readFileSync(path.join(root,'apps/fieldroom/data/engineering-case.json'),'utf8'));
const element={addEventListener(){}};
const context=vm.createContext({document:{querySelector(){return element},querySelectorAll(){return []}},FieldScene3D:class{},location:{search:''},URLSearchParams,fetch(){return new Promise(()=>{})}});
vm.runInContext(fs.readFileSync(path.join(root,'apps/fieldroom/archive.js'),'utf8').replace('export function','function')+'\n'+fs.readFileSync(path.join(root,'apps/fieldroom/viewer.js'),'utf8').replace(/^import .*;$/gm,''),context);
context.fixture=fixture;
test('actual exported archive satisfies the browser contract',()=>{
 assert.equal(vm.runInContext('validate(fixture).schema',context),'oil-engineering-viewer/v1');
});
for(const [label,mutate] of [
 ['field size',d=>d.field.pressure_pa.pop()],
 ['time order',d=>d.frames[1].time_days=d.frames[0].time_days],
 ['missing well',d=>delete d.frames[0].wells.P1],
 ['nonfinite component',d=>d.frames[0].wells.P1.component_mass_rates_kg_s[0]=Infinity],
 ['unsupported equipment',d=>d.topology.facility.separator_stages.pop()],
 ['unsupported connection',d=>d.topology.gathering.pipes[0].start='unknown'],
 ['missing phase data',d=>delete d.frames[0].pipes.collector.outlet_state],
 ['unqualified source',d=>d.provenance.source_guard=false],
])test('reject '+label,()=>{
 const invalid=structuredClone(fixture);mutate(invalid);context.invalid=invalid;
 assert.throws(()=>vm.runInContext('validate(invalid)',context));
});
test('rendered untrusted object labels are escaped',()=>{
 assert.equal(vm.runInContext('esc(\'<img src=x onerror="alert(1)">\')',context),'&lt;img src=x onerror=&quot;alert(1)&quot;&gt;');
});

test('cell picking accounts for object-fit letterboxing on narrow screens',()=>{
 vm.runInContext('fieldBounds={W:900,H:640,x0:162,y0:32,size:18,nx:32,ny:32}',context);
 context.pointer={clientX:150,clientY:150,target:{getBoundingClientRect(){return {left:0,top:0,width:300,height:300}}}};
 const actual=vm.runInContext('fieldPoint(pointer)',context);
 assert.equal(actual.i,16);assert.equal(actual.j,15);
 context.pointer.clientY=20;
 assert.ok(vm.runInContext('fieldPoint(pointer).j',context)>31);
});
