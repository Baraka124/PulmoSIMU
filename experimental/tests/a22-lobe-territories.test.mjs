import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
const source=readFileSync(new URL('../anatomy/lobe-territories.js',import.meta.url),'utf8');
const main=readFileSync(new URL('../../index.html',import.meta.url),'utf8');
const page={};runInNewContext(source,{window:page});
const atlas=page.PulmoCTLobeTerritories;
test('A2.2 RAS-side aware atlas seeds five pulmonary study territories',()=>{
  const examples=[
    [[90,60,-310],'RUL'],
    [[90,120,-360],'RML'],
    [[90,80,-470],'RLL'],
    [[-90,60,-330],'LUL'],
    [[-90,60,-460],'LLL']
  ];
  for(const [ras,name]of examples)
    assert.equal(atlas.classifyRAS(ras),name,ras+' classified as '+name);
  assert.deepEqual(Object.keys(atlas.names).sort(),['LLL','LUL','RLL','RML','RUL']);
  assert.throws(()=>atlas.classifyRAS([NaN,0,0]),/Non-finite/);
});
test('A2.2 provisional fissure offsets are constrained and visibly change territories',()=>{
  const extreme=atlas.parameters({rightMajor:999,rightMinor:-999,leftMajor:999,selected:'random'});
  assert.equal(extreme.rightMajor,35);
  assert.equal(extreme.rightMinor,-30);
  assert.equal(extreme.leftMajor,35);
  assert.equal(extreme.selected,'all');
  const testRAS=[90,120,-352];
  assert.notEqual(atlas.classifyRAS(testRAS,{rightMinor:-30}),
    atlas.classifyRAS(testRAS,{rightMinor:30}),
    'Editable horizontal boundary must reassign atlas territory');
});
test('A2.2 source geometry remains untouched with reversible display materials',()=>{
 for(const key of [
  'const working=base.clone()',
  'rec.mesh.geometry=rec.working;rec.mesh.material=rec.atlasMaterial',
  'rec.mesh.geometry=rec.base;rec.mesh.material=rec.originalMaterial',
  'rec.working.dispose();rec.atlasMaterial.dispose()',
  "mode:active?'editable-atlas-territories':'original-source'",
  "label:'ATLAS-SEEDED LOBE TERRITORIES / NOT FISSURE SEGMENTATION'",
  'modelToRAS.clone().multiply(rel)'
 ]) assert.ok(source.includes(key),'Missing non-destructive atlas invariant: '+key);
 assert.ok(!source.includes('GeometryUtils.mergeVertices('));
 assert.ok(!source.includes('morphTargets'));
 assert.ok(!source.includes('new THREE.SphereGeometry('));
});
test('A2.2 existing CT workspace and procedural simulator remain independently reachable',()=>{
 for(const key of [
  "lobe-territories.js?v=A22-FISSURE-ATLAS",
  'id="ct-lobe-panel"',
  'id="ct-lobe-focus"',
  'id="ct-lobe-toggle"',
  'window.toggleCTTerritoryAtlas=()=>',
  'window.updateCTTerritoryAtlas=()=>',
  "if(ctLobeAtlas)closeCTTerritoryAtlas();",
  "_setResearchNeutralMaterial(false)",
  "setViewMode('reference')",
  'scene.background=new THREE.Color(0x121c22)',
  'gnd.visible=false;vignPlane.visible=false',
  'sssL.intensity=0;heartL.intensity=0',
  'scene.background=old.background;scene.fog=old.fog'
 ]) assert.ok(main.includes(key),'Missing research environment/atlas invariant: '+key);
 assert.ok(main.includes('bronchoscopyAllowed')===false,
   'No new blanket clinical permission should be introduced by the atlas');
});
