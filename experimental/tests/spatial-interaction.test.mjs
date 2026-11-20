import test from 'node:test';import assert from 'node:assert/strict';
import {bimanualPose,clampDisplayScale,rotationBetweenVectors,rotateVector,dominantThumbstick,normalizeProgress} from '../xr/spatial-interaction.js';
const close=(a,b)=>{assert.equal(a.length,b.length);a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<1e-5,`coordinate ${i}: ${v} vs ${b[i]}`))};
const start={left:[-.2,1,-1],right:[.2,1,-1],position:[0,1.2,-1.6],quaternion:[0,0,0,1],scale:1};
test('two-handed unchanged grips preserve exact anatomical pose',()=>{
 const p=bimanualPose(start,start.left,start.right);
 close(p.position,start.position);close(p.quaternion,start.quaternion);assert.equal(p.scale,1);
});
test('moving both grips translates the held model, without camera locomotion',()=>{
 const p=bimanualPose(start,[.3,1.1,-1],[.7,1.1,-1]);
 close(p.position,[.5,1.3,-1.6]);assert.ok(Math.abs(p.scale-1)<1e-9);
});
test('two-handed pinch changes size about midpoint with bounded scale',()=>{
 const p=bimanualPose(start,[-.4,1,-1],[.4,1,-1]);
 assert.equal(p.scale,1.9);
 const q=bimanualPose(start,[-.04,1,-1],[.04,1,-1]);
 assert.equal(q.scale,.55);
});
test('rotating hand axis turns anatomy while keeping grip midpoint fixed',()=>{
 const p=bimanualPose(start,[0,.8,-1],[0,1.2,-1]);
 close(rotateVector(p.quaternion,[1,0,0]),[0,1,0]);
 close(p.position,[-.2,1,-1.6]);
});
test('coincident grips fail safe and invalid data rejects',()=>{
 const p=bimanualPose(start,[0,1,-1],[0,1,-1]);
 close(p.position,start.position);assert.equal(p.scale,1);
 assert.throws(()=>bimanualPose({...start,position:[NaN,0,0]},start.left,start.right));
 assert.throws(()=>clampDisplayScale(Infinity));
 close(rotateVector(rotationBetweenVectors([1,0,0],[-1,0,0]),[1,0,0]),[-1,0,0]);
});
test('controller sticks and route progress have safe dead zones',()=>{
 assert.deepEqual(dominantThumbstick([0,0,.05,-.8]),[0,-.8]);
 assert.deepEqual(dominantThumbstick([0.35,-.05]),[.35,0]);
 assert.deepEqual(dominantThumbstick(null),[0,0]);
 assert.equal(normalizeProgress(-2),0);assert.equal(normalizeProgress(2),1);
});
