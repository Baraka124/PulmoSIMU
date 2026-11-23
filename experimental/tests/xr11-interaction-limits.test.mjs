import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
const source=readFileSync(new URL('../xr/interaction-math.js',import.meta.url),'utf8');
function math(){
  const w={};runInNewContext(source,{window:w});return w.PulmoXRInteractionMath;
}
test('XR-11 caps anatomical world-space span regardless of original model units',()=>{
 const m=math();
 assert.equal(m.clampScale(100,8),m.CONFIG.maximumSpanM/8);
 assert.equal(m.clampScale(.0001,8),m.CONFIG.minimumSpanM/8);
 assert.equal(m.clampScale(1.0,.5),m.CONFIG.maximumSpanM/.5);
 for(const extent of [.08,.5,1.0,3,12,150]){
   for(const raw of [.0001,.4,1,1.65,9,100]){
     const span=extent*m.clampScale(raw,extent);
     assert.ok(span<=m.CONFIG.maximumSpanM+1e-8,'Cannot fill a VR room');
     assert.ok(span>=m.CONFIG.minimumSpanM-1e-8,'Model remains findable');
   }
 }
});
test('XR-11 limits two-hand pinch ratios as well as final world size',()=>{
 const m=math();
 const initial=.4,extent=4;
 const normal=m.scaleFromHands(initial,.35,.42,extent);
 assert.equal(normal.valid,true);
 assert.ok(normal.scale*extent<=.92);
 const expansive=m.scaleFromHands(initial,.20,8,extent);
 assert.equal(expansive.valid,true);
 assert.equal(expansive.scale,m.CONFIG.maximumSpanM/extent);
 const folded=m.scaleFromHands(initial,.3,.01,extent);
 assert.equal(folded.valid,false);
 assert.equal(folded.scale,m.CONFIG.maximumSpanM/extent);
});
test('XR-11 limits the actual organ focus, not the world-origin pivot',()=>{
 const m=math();
 const far=m.clampFocus([0,0,-15],[0,0,0]);
 assert.equal(far.clamped,true);
 assert.ok(Math.abs(far.center[2]+m.CONFIG.maximumViewerDistanceM)<1e-9);
 const near=m.clampFocus([0,0,-.1],[0,0,0]);
 assert.equal(near.clamped,true);
 assert.ok(Math.abs(near.center[2]+m.CONFIG.minimumViewerDistanceM)<1e-9);
 const ordinary=m.clampFocus([.3,.2,-1],[0,0,0]);
 assert.equal(ordinary.clamped,false);
});
test('XR-11 preserves original procedural engine, source geometry and independent bronchoscopy',()=>{
 const xr=readFileSync(new URL('../xr/procedural-immersive.js',import.meta.url),'utf8');
 const main=readFileSync(new URL('../../index.html',import.meta.url),'utf8');
 assert.match(main,/interaction-math\.js\?v=XR11-METRIC/);
 for(const token of ['constrainSizeAndFocus()', 'limits.clampFocus(', 'limits.scaleFromHands(',
  'applyAnalogAnatomy(', 'limits.clampScale(', "mode==='anatomy'",
  'worldMenu.updateWorldMatrix(true,false)', 'panelIntersection(rayFrom(i))',
  'hoverPanels.includes(a.id)', 'aimTargets[i]', 'hoverPanels[i]',
  'menuActions', 'right', 'left', 'updatePointerAim(time)', 'rBro.render(scene,vrCam)']){
  assert.ok(xr.includes(token),'Required real interaction invariant: '+token)
 }
 assert.ok(!xr.includes('const scaling=safe(.18,1.65'),'No unbounded two-hand scene-unit resizing');
 assert.ok(!xr.includes('const distance=mGrp.position.distanceTo(head)'),
   'No pivot-based artificial move block');
 assert.ok(!xr.includes('new THREE.CatmullRomCurve3('),'No second/competing lung model');
});
