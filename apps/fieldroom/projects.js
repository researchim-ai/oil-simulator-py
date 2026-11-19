// Browser-local storage. Nothing is sent to a server or an AI provider.
function open(){return new Promise((resolve,reject)=>{const r=indexedDB.open('fieldroom-projects',1);r.onupgradeneeded=()=>r.result.createObjectStore('projects',{keyPath:'id'});r.onerror=()=>reject(r.error);r.onsuccess=()=>resolve(r.result);});}
async function transact(mode,fn){const db=await open();return new Promise((resolve,reject)=>{const tx=db.transaction('projects',mode);let value;try{const r=fn(tx.objectStore('projects'));r.onsuccess=()=>{value=r.result;};}catch(e){db.close();reject(e);return;}tx.oncomplete=()=>{db.close();resolve(value);};tx.onabort=tx.onerror=()=>{db.close();reject(tx.error||Error('Не удалось сохранить проект'));};});}
export const listProjects=()=>transact('readonly',s=>s.getAll());
export const readProject=id=>transact('readonly',s=>s.get(id));
export const saveProject=p=>transact('readwrite',s=>s.put(p));
