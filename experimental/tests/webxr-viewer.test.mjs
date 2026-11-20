import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {spawnSync} from 'node:child_process';
const html=readFileSync(new URL('../vr.html',import.meta.url),'utf8');
test('WebXR prototype has a valid standalone ES module and import map',()=>{
 const scripts=[...html.matchAll(/<script([^>]*)>([\s\S]*?)<\/script>/g)];
 const module=scripts.find(m=>m[1].includes('type="module"'));
 assert.ok(module,'Standalone module required');
 const check=spawnSync(process.execPath,['--input-type=module','--check'],{input:module[2],encoding:'utf8'});
 assert.equal(check.status,0,check.stderr);
 const importmap=scripts.find(m=>m[1].includes('type="importmap"'));
 assert.ok(importmap);
 const imports=JSON.parse(importmap[2]).imports;
 assert.ok(imports.three.startsWith('https://'));
});
test('Quest mode includes VR lifecycle, controller picks, desktop fallback and source provenance',()=>{
 for(const token of ['VRButton.createButton(renderer)','renderer.xr.enabled=true','getController(i)','selectstart',
  'setAnimationLoop','isSessionSupported','resize.observe(canvas)','fetchReferenceAirwayGraph',
  'REFERENCE MODEL','Not clinically validated','Back to Studio']){
   assert.ok(html.includes(token),token);
 }
});
test('VR viewer never uses patient datasets or calibrated physiology as implicit source',()=>{
 assert.ok(!html.includes('patientData'));
 assert.ok(!html.includes('withTherapy'));
 assert.ok(!html.includes('upload'));
 assert.match(html,/not a measured patient lung/i);
});
