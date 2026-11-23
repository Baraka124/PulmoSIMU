import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {loadAirwayGraph} from '../anatomy/airway-graph.js';
import {createXRReferencePhysicsForGraph} from '../xr/reference-physics.js';
const data=JSON.parse(readFileSync(new URL('../anatomy/reference-airways.json',import.meta.url),'utf8'));
const graph=loadAirwayGraph(data);

test('XR-09 uses exactly the original R3 solver and source reference — never patient CT',()=>{
 const engine=createXRReferencePhysicsForGraph(graph);
 const source=engine.read();
 assert.equal(source.validation.solver,'R3 linear RC network');
 assert.equal(source.validation.geometry,'uncalibrated procedural reference');
 assert.equal(source.validation.clinicalValidated,false);
 assert.equal(source.validation.meshMotionCoupled,false);
 assert.equal(source.airwayCount,18);
 assert.ok(source.terminalCount>0);
 assert.equal(source.tSeconds,0);
});
test('XR-09 produces finite, conservation-respecting original-solver pressure and flow in SI units',()=>{
 const engine=createXRReferencePhysicsForGraph(graph);
 let current;
 for(let i=0;i<500;i++)current=engine.step(.02);
 assert.ok(current.tSeconds>9.99);
 assert.ok(current.pressurePa>=0);
 assert.ok(Number.isFinite(current.flowM3PerS));
 assert.ok(Number.isFinite(current.volumeM3));
 assert.ok(Math.abs(current.conservationResidualM3PerS)<1e-7);
 assert.ok(current.airwayCount===18);
 const replay=engine.reset();
 assert.equal(replay.pressurePa,0);
 assert.equal(replay.tSeconds,0);
});
test('frame-rate changes do not change integrated R3 time at equal total elapsed time',()=>{
 const a=createXRReferencePhysicsForGraph(graph),b=createXRReferencePhysicsForGraph(graph);
 for(let i=0;i<100;i++)a.step(.04);
 for(let i=0;i<200;i++)b.step(.02);
 const x=a.read(),y=b.read();
 assert.ok(Math.abs(x.tSeconds-y.tSeconds)<1e-7);
 assert.ok(Math.abs(x.pressurePa-y.pressurePa)<1e-7);
 assert.ok(Math.abs(x.volumeM3-y.volumeM3)<1e-10);
 assert.throws(()=>a.step(.5),/Invalid headset physics/);
 assert.throws(()=>createXRReferencePhysicsForGraph({schema:'patient-CT',nodes:[],root:{id:0}}));
});
