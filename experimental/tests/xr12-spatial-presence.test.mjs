import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
const routerSource=readFileSync(new URL('../xr/spatial-target.js',import.meta.url),'utf8');
const xr=readFileSync(new URL('../xr/procedural-immersive.js',import.meta.url),'utf8');
const main=readFileSync(new URL('../../index.html',import.meta.url),'utf8');
function choose(args){
  const window={};runInNewContext(routerSource,{window});
  return window.PulmoXRSpatialTarget.choose(args);
}
test('XR12 pointer and menu use the same nearest intersection, blocking click-through',()=>{
 const panel={distance:.55,action:'reset'};
 const anatomy={hit:true,distanceM:.92};
 const v=choose({panel,anatomy,mode:'anatomy'});
 assert.equal(v.kind,'panel');assert.equal(v.action,'reset');
 const reversed=choose({panel:{distance:1.15,action:'exit'},
   anatomy:{hit:true,distanceM:.65},mode:'anatomy'});
 assert.equal(reversed.kind,'anatomy');
 assert.equal(reversed.action,null);
 assert.equal(choose({panel:{distance:.7,action:null},anatomy:{hit:true,distanceM:.9}}).kind,'panel');
 assert.equal(choose({anatomy:{hit:true,distanceM:.5},mode:'procedure'}).kind,'none');
 assert.equal(choose({panel:{distance:.8,action:'tool'},mode:'procedure'}).kind,'panel');
 assert.equal(choose({panel:{distance:NaN,action:'tool'},mode:'procedure'}).kind,'none');
});
test('XR12 controller release never teleports source anatomy; explicit recenter exists',()=>{
 const release=xr.slice(xr.indexOf('const release='),xr.indexOf('function setMode('));
 assert.ok(!release.includes('summon();'),
   'A trigger tap may not teleport the model on release');
 assert.ok(release.includes('POSITION PRESERVED'));
 assert.ok(xr.includes("if(action==='reset')frameGroup()"));
 assert.ok(xr.includes("if(side==='left'&&edge(5))toggleSpatialPanel(i)"));
});
test('XR12 provides visual, tactile and local-only interaction feedback',()=>{
 for(const token of ['global.PulmoXRSpatialTarget.choose({panel,anatomy:organ,mode})',
  'choosePointer(i)',"choice.kind==='panel'",'hoverPanels[i]',
  'function toggleSpatialPanel(i)',"if(nextHover)haptic(i,.09,16)",
  "if(source==='controller')haptic(i,.31,48)",
  'hapticActuators','vibrationActuator',
  "new THREE.SphereGeometry(8,32,16)",
  "XR-12 controller routing on original procedural scene",
  "pulmosimu-xr12-session-report.json"]){
  assert.ok(xr.includes(token),'Missing XR12 interaction invariant: '+token);
 }
 assert.ok(main.includes('spatial-target.js?v=XR12-ROUTER'));
 assert.ok(!routerSource.includes('fetch('));
 assert.ok(!xr.includes('new THREE.CatmullRomCurve3('),
  'No second lung, fake patient model or bronchoscopy engine');
});
