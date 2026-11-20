import test from 'node:test';import assert from 'node:assert/strict';
import {SPATIAL_LIMITS,decideInteraction,pinchActive,workZoneTarget,workZoneScale,easedTransform,frameMetrics} from '../xr/spatial-intent.js';
const approx=(a,b)=>assert.ok(Math.abs(a-b)<1e-7,`${a} != ${b}`);
test('panel has priority; anatomical contact is near; far ray requires explicit hit',()=>{
 assert.equal(decideInteraction({panelHit:true,surfaceDistanceM:0,rayHit:true}),'panel');
 assert.equal(decideInteraction({mode:'path',surfaceDistanceM:.01,rayHit:true}),'locked');
 assert.equal(decideInteraction({surfaceDistanceM:.03,rayHit:false}),'near');
 assert.equal(decideInteraction({surfaceDistanceM:.6,rayHit:true}),'far');
 assert.equal(decideInteraction({surfaceDistanceM:Infinity,rayHit:false}),'none');
 assert.equal(decideInteraction({surfaceDistanceM:-.1,rayHit:true}),'far');
});
test('bare hand pinches use hysteresis rather than flicker or sticky state',()=>{
 assert.equal(pinchActive(false,.05),false);
 assert.equal(pinchActive(false,.024),true);
 assert.equal(pinchActive(true,.045),true);
 assert.equal(pinchActive(true,.056),false);
 assert.equal(pinchActive(true,NaN),false);
});
test('summon target is in front of headset, independent of desktop camera',()=>{
 const a=workZoneTarget({head:[1,1.22,2],forward:[0,0,-1]});
 approx(a[2],2-SPATIAL_LIMITS.summonDistanceM);
 approx(a[1],1.12);
 const b=workZoneTarget({head:[1,1.22,2],forward:[1,.9,0]});
 approx(b[0],1+SPATIAL_LIMITS.summonDistanceM);approx(b[2],2);
 assert.throws(()=>workZoneTarget({head:[0,0,0],forward:[0,1,0]}));
});
test('workzone geometry remains within arm-sized display envelope',()=>{
 const bounds=[.65,.8,.5];const scale=workZoneScale(bounds,1.4);
 assert.ok(scale*bounds[0]<=SPATIAL_LIMITS.summonWidthM+1e-8);
 assert.ok(scale*bounds[1]<=SPATIAL_LIMITS.summonHeightM+1e-8);
 approx(workZoneScale([1,1,1],.1),.1);
 assert.throws(()=>workZoneScale([0,1,1],1));
});
test('summon is continuous with finite arrival, never an instant teleport',()=>{
 const start={position:[0,1,-2.3],scale:1};
 const end={position:[0,1,-.9],scale:.65};
 const a=easedTransform(start,end,0),m=easedTransform(start,end,.5),b=easedTransform(start,end,1);
 assert.deepEqual(a.position,start.position);assert.equal(a.finished,false);
 assert.ok(m.position[2]>start.position[2]&&m.position[2]<end.position[2]);
 assert.ok(m.scale<start.scale&&m.scale>end.scale);
 assert.deepEqual(b.position,end.position);assert.equal(b.finished,true);
});
test('frame diagnostics are bounded and never imply clinical validation',()=>{
 const a=frameMetrics([],1/90);assert.equal(a.fps,90);assert.equal(a.slowFrames,0);
 const b=frameMetrics([1/90,1/90],1/45);
 assert.equal(b.samples.length,3);assert.equal(b.slowFrames,1);
 assert.ok(b.p95Ms>20);
 const c=frameMetrics(Array(100).fill(1/90),1/90);
 assert.equal(c.samples.length,90);
});
