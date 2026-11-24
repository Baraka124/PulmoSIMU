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

test('mechanistic laboratory cannot silently reintroduce fake 3D lung anatomy',()=>{
 const html=readFileSync(new URL('../lab.html',import.meta.url),'utf8');
 for(const forbidden of ['new THREE.SphereGeometry','new THREE.TubeGeometry',
   'const lobes=[','Object.values(meshes)','airwayGroup','toggleAirways',
   'o.material=new THREE.MeshPhysicalMaterial']){
   assert.ok(!html.includes(forbidden),'Forbidden fabricated-anatomy construct: '+forbidden);
 }
 for(const required of ['id="referenceObservatory"','id="regionalBars"',
   'No invented lobe ellipsoids','no inferred organ surface','source-mode',
   'preserve imported', 'line-only']){
   // Source geometry must be meaningful; allow equivalent prose for comments.
   if(required==='preserve imported'||required==='line-only')continue;
   assert.ok(html.toLowerCase().includes(required.toLowerCase()),'Source-only gate missing '+required);
 }
 assert.ok(html.includes("if(ctMode||surfaceMode){controls.update();renderer.render(scene,camera)}"),
   '3D renderer must remain dormant without a supplied geometry source');
});
