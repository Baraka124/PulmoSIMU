import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {spawnSync} from 'node:child_process';
const stage=readFileSync(new URL('../xr/stage-presence.js',import.meta.url),'utf8');
const xr=readFileSync(new URL('../xr/procedural-immersive.js',import.meta.url),'utf8');
const main=readFileSync(new URL('../../index.html',import.meta.url),'utf8');
const launcher=readFileSync(new URL('../../quest-short/index.html',import.meta.url),'utf8');

test('XR13 stage remains one low-draw-call environment not synthetic pulmonary anatomy',()=>{
  const syntax=spawnSync(process.execPath,['--check'],{input:stage,encoding:'utf8'});
  assert.equal(syntax.status,0,syntax.stderr);
  assert.match(stage,/const columns=12, radius=3\.45/);
  assert.match(stage,/new THREE\.LineSegments\(/);
  assert.match(stage,/new THREE\.HemisphereLight\(/);
  assert.ok(!stage.includes('fetch('),'No network or image requests in spatial scene');
  assert.ok(!stage.includes('getAnatomyMeshes'),'Scene dressing must not modify anatomy');
  assert.match(stage,/boundaries\.visible=next==='studio'/);
  assert.match(stage,/if\(grid\)grid\.visible=next==='studio'/);
  assert.match(stage,/if\(plinth\)plinth\.visible=next==='studio'/);
});
test('XR13 expert menu preserves original actions while primary menu stays compact',()=>{
  for(const name of ["'anatomy'","'procedure'","'reset'","'view'","'more'","'exit'",
    "'tool'","'scale'","'back'"]){
    assert.ok(xr.includes('id:'+name),'Required spatial menu action '+name);
  }
  for(const token of ['actionHitbox=panelLayouts.primary',
    "menuPage=action==='more'?'advanced':'primary'",
    "stage.setMode(modeNext)", "menuPage==='advanced'?'Research instruments'",
    "panelPage:menuPage"]){
    assert.ok(xr.includes(token),'Missing contextual scene/UI contract '+token);
  }
  assert.ok(launcher.includes('XR-14 · Immersive Procedure Laboratory'));
  assert.ok(launcher.includes('EXPERT TOOLS'));
  assert.ok(launcher.includes('FOCUS VIEW'));
});
test('XR13 foveation is optional and reported as a request, not validated frames',()=>{
  assert.match(xr,/if\(layer&&'fixedFoveation' in layer\)/);
  assert.match(xr,/layer\.fixedFoveation=\.35/);
  assert.match(xr,/typeof rBro\.xr\.setFoveation==='function'/);
  assert.match(xr,/headset performance unverified/);
  assert.match(xr,/qualityStatus=configureComfortRendering\(\)/);
  assert.match(xr,/stagePresentation:stage\.snapshot\(\)/);
  assert.ok(main.includes('stage-presence.js?v=XR13-ARCHITECTURE'));
  assert.ok(main.includes('procedural-immersive.js?v=XR14-PROCEDURE-MR'));
});
