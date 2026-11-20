const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const path=require('node:path');
const root=path.resolve(__dirname,'../apps/fieldroom');
const source=fs.readFileSync(path.join(root,'archive.js'),'utf8').replace('export function','function')+'\n'+fs.readFileSync(path.join(root,'workspace.js'),'utf8').replace(/^import .*;$/gm,'');
const archive=JSON.parse(fs.readFileSync(path.join(root,'data/engineering-case.json')));
async function setup(){
 const elements=new Map(),records=new Map();let seq=0,fail=false;
 function element(){return {value:'',disabled:false,children:[],append(...v){this.children.push(...v)},replaceChildren(){this.children=[]},click(){}};}
 const $=s=>{if(!elements.has(s))elements.set(s,element());return elements.get(s)};
 const context=vm.createContext({document:{querySelector:$,querySelectorAll:()=>[...elements.values()],createElement:element},crypto:{randomUUID:()=>`id-${++seq}`},Date,Blob,URL:{createObjectURL:()=>'',revokeObjectURL(){}},setTimeout:fn=>fn(),fetch:async()=>({ok:true,json:async()=>structuredClone(archive)}),listProjects:async()=>[...records.values()].map(structuredClone),readProject:async id=>structuredClone(records.get(id)),saveProject:async p=>{if(fail)throw Error('Storage full');records.set(p.id,structuredClone(p));}});
 vm.runInContext(source,context);await new Promise(setImmediate);
 return {$,records,context,fail:value=>{fail=value},run:s=>vm.runInContext(s,context)};
}
function file(value){return {target:{value:'selected',files:[{size:10,text:async()=>JSON.stringify(value)}]}};}
test('backup restore creates an independent copy and preserves draft when replacing archive',async()=>{
 const h=await setup();await h.$('#demo').onclick();
 h.$('#name').value='Changed name';h.$('#notes').value='Engineering notes';
 const event=file(archive);await h.$('#archive').onchange(event);
 assert.equal(event.target.value,'');assert.equal(h.records.get('id-1').name,'Changed name');assert.equal(h.records.get('id-1').notes,'Engineering notes');
 const backup=JSON.parse(JSON.stringify(h.records.get('id-1')));
 await h.$('#restore').onchange(file(backup));
 assert.equal(h.records.size,2);assert.notEqual(h.records.get('id-2').id,backup.id);
 assert.deepEqual(h.records.get('id-2').archive,backup.archive);
});
test('invalid restore cannot create a project and can be selected again',async()=>{
 const h=await setup();for(const value of [null,{}, {schema:'fieldroom-project/v1',name:'X',notes:'',archive:false}]){
 const event=file(value);await h.$('#restore').onchange(event);assert.equal(h.records.size,0);assert.equal(event.target.value,'');assert.match(h.$('#message').textContent,/Ошибка/);
 }
});
test('failed import preserves existing archive and unsaved draft; retry succeeds',async()=>{
 const h=await setup();await h.$('#demo').onclick();h.$('#name').value='Unsaved';h.fail(true);
 const event=file(archive);await h.$('#archive').onchange(event);
 assert.equal(h.records.get('id-1').name,'Демонстрация · 30 суток');assert.equal(h.$('#name').value,'Unsaved');assert.equal(event.target.value,'');
 h.fail(false);await h.$('#archive').onchange(file(archive));assert.equal(h.records.get('id-1').name,'Unsaved');
});
test('overlapping actions are blocked until the current operation completes',async()=>{
 const h=await setup();let release;h.context.pending=new Promise(r=>{release=r});const work=h.run('guard(()=>pending)');
 await h.$('#create').onclick();assert.equal(h.records.size,0);release();await work;
 await h.$('#create').onclick();assert.equal(h.records.size,1);
});
test('switching projects saves draft before opening another',async()=>{
 const h=await setup();await h.$('#create').onclick();h.$('#name').value='First';await h.$('#create').onclick();
 assert.equal(h.records.get('id-1').name,'First');assert.equal(h.records.size,2);
});
