import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {spawnSync} from 'node:child_process';
test('experimental inline module is syntactically valid JavaScript',()=>{
 const html=readFileSync(new URL('../lab.html',import.meta.url),'utf8');
 const match=html.match(/<script type="module">([\s\S]*?)<\/script>/);
 assert.ok(match,'Expected module script');
 const result=spawnSync(process.execPath,['--input-type=module','--check'],{input:match[1],encoding:'utf8'});
 assert.equal(result.status,0,result.stderr||'Module syntax check failed');
});
test('source selector controls and explicit nonclinical modes are wired',()=>{
 const html=readFileSync(new URL('../lab.html',import.meta.url),'utf8');
 for(const id of ['surfaceManifest','surfaceMesh','loadSurfaceAtlas','surfaceLobes','surfaceVessels','surfaceInspection'])
 assert.match(html,new RegExp('id="'+id+'"'));
 assert.match(html,/NOT computed from displayed geometry/);
});
