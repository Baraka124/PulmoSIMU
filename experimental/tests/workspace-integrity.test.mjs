import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {spawnSync} from 'node:child_process';
const root=new URL('../../',import.meta.url);
function checkScripts(filename){
 const html=readFileSync(new URL(filename,root),'utf8');
 const found=[...html.matchAll(/<script(\s[^>]*)?>([\s\S]*?)<\/script>/g)];
 assert.ok(found.length>=1,filename+' has scripts');
 assert.equal((html.match(/<script(?:\s|>)/g)||[]).length,found.length,'All script tags must close');
 for(const m of found){
  const isModule=(m[1]||'').includes('type="module"');
  const args=isModule?['--input-type=module','--check']:['--check'];
  const run=spawnSync(process.execPath,args,{input:m[2],encoding:'utf8'});
  assert.equal(run.status,0,filename+': '+run.stderr);
 }
}
test('Studio entry and procedural wrapper scripts parse separately',()=>{
 checkScripts('studio.html');
 checkScripts('procedural-workspace.html');
});
