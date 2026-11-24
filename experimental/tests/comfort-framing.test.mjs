import test from 'node:test';import assert from 'node:assert/strict';
import {fitSeatedAnatomy,isComfortFramed,COMFORT_DEFAULTS} from '../xr/comfort-framing.js';
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-7, `Expected ${a} ≈ ${b}`);
test('Quest 2 seated placement starts across the room, not inside the lungs',()=>{
  const head=[.13,1.2,-.18],forward=[0,0,-1],bounds=[.85,.65,.37];
  const fit=fitSeatedAnatomy({head,forward,bounds,localCenter:[0,.03,0]});
  near(fit.center[0],head[0]);near(fit.center[1],head[1]+COMFORT_DEFAULTS.verticalOffsetM);
  near(fit.center[2],head[2]-COMFORT_DEFAULTS.distanceM);
  assert.ok(fit.sizeM[0]<=.56+1e-7&&fit.sizeM[1]<=.55+1e-7);
  assert.ok(fit.distanceM>=2);
  assert.ok(isComfortFramed({head,center:fit.center,size:fit.sizeM}));
});
test('user turning head changes facing but never leaves model at inherited desktop camera origin',()=>{
 const head=[1.2,1.05,3.1],forward=[-1,0,0],bounds=[.82,.7,.39];
 const fit=fitSeatedAnatomy({head,forward,bounds});
 near(fit.center[0],head[0]-COMFORT_DEFAULTS.distanceM);
 near(fit.center[2],head[2]);
 assert.ok(fit.sizeM[0]<=.56+1e-7&&fit.sizeM[1]<=.55+1e-7);
});
test('reference coordinate center is explicitly returned for pivot-centering, never discarded',()=>{
 const fit=fitSeatedAnatomy({head:[0,1,0],forward:[0,0,-1],bounds:[1,1,.6],localCenter:[-.1,.25,.04]});
 assert.deepEqual(fit.localCenter,[-.1,.25,.04]);
 near(fit.scale,.55);
});
test('invalid or unreasonably large initial frames are rejected',()=>{
 assert.throws(()=>fitSeatedAnatomy({head:[NaN,0,0],forward:[0,0,-1],bounds:[1,1,1]}));
 assert.throws(()=>fitSeatedAnatomy({head:[0,1,0],forward:[0,0,0],bounds:[1,1,1]}));
 assert.throws(()=>fitSeatedAnatomy({head:[0,1,0],forward:[0,0,-1],bounds:[1,-1,1]}));
 assert.equal(isComfortFramed({head:[0,1,0],center:[0,1,-.3],size:[2,2,2]}),false);
});
