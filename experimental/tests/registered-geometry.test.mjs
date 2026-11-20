import test from 'node:test';
import assert from 'node:assert/strict';
import '../xr/registered-geometry.js';
const {assessRegistration,assessLumenClearance}=globalThis.PulmoXR08;
const metadata={
 version:'pneumovr-1.1',transform:{centre_mm:[10,20,30],scale:4},
 segments:[
  {id:0,parent:-1,r0:.085,r1:.07,pts:[[0,.8,0],[0,.2,0]]},
  {id:4,parent:0,r0:.07,r1:.045,pts:[[0,.2,0],[.4,-.2,.1]]},
  {id:5,parent:4,r0:.04,r1:.025,pts:[[.4,-.2,.1],[.7,-.6,.2]]}
 ]
};
const bounds={min:[-.5,-1,-.5],max:[1.1,1.1,.7]};
test('CT mesh and its source airway centerlines are coframed with derived physical transform, not clinically validated',()=>{
 const r=assessRegistration(metadata,bounds);
 assert.equal(r.source,'ct-derived');
 assert.equal(r.status,'coframed');
 assert.equal(r.sourceScaleEligible,true);
 assert.equal(r.sceneUnitMetres,.25);
 assert.equal(r.branchCount,3);
 assert.equal(r.fullDicomOrientationValidated,false);
 assert.equal(r.modelToPatientRegistrationValidated,false);
 assert.equal(r.anatomicalValidated,false);
 assert.ok(r.boundsDiagonalMetres>0);
 assert.ok(!JSON.stringify(r).includes('centre_mm'),'No patient transform origin or identity should leak to monitor');
});
test('synthetic model can be inspected but cannot claim patient-scale eligibility',()=>{
 const m={...metadata,transform:{...metadata.transform,is_synthetic:true},patient:{PatientName:'PRIVATE',PatientID:'987'}};
 const r=assessRegistration(m,bounds);
 assert.equal(r.source,'synthetic');
 assert.equal(r.sourceScaleEligible,false);
 assert.ok(!JSON.stringify(r).includes('PRIVATE'));
});
test('rejects mismatched, malformed, disconnected or cyclic CT source geometry before loading into procedure graph',()=>{
 assert.throws(()=>assessRegistration(metadata,{min:[50,50,50],max:[51,51,51]}),/not co-framed/);
 assert.throws(()=>assessRegistration(metadata,{min:[0,0,0],max:[0,1,1]}),/degenerate/);
 assert.throws(()=>assessRegistration({...metadata,transform:{scale:0,centre_mm:[0,0,0]}},bounds),/transform/);
 assert.throws(()=>assessRegistration({...metadata,version:'unknown'},bounds),/version/);
 assert.throws(()=>assessRegistration({...metadata,segments:[metadata.segments[0],metadata.segments[0]]},bounds),/duplicate/);
 assert.throws(()=>assessRegistration({...metadata,segments:metadata.segments.map((x,i)=>i===1?{...x,r0:-1}:x)},bounds),/radius/);
 assert.throws(()=>assessRegistration({...metadata,segments:[
  {...metadata.segments[0],parent:5},metadata.segments[1],metadata.segments[2]
 ]},bounds),/root|cyclic/);
});
test('a declared but incorrect child mapping and impossible physical scene extent fail closed',()=>{
 const bad=metadata.segments.map(s=>({...s,children:s.id===0?[5]:[]}));
 assert.throws(()=>assessRegistration({...metadata,segments:bad},bounds),/parent\/child topology/);
 const hugeScale={...metadata,transform:{...metadata.transform,scale:.1}};
 assert.throws(()=>assessRegistration(hugeScale,bounds),/extent/);
 const tinyScale={...metadata,transform:{...metadata.transform,scale:500}};
 assert.throws(()=>assessRegistration(tinyScale,bounds),/extent/);
});
test('scope tip cannot advance into a too-small reference airway; resistance estimate has explicit provenance',()=>{
 const segment={r0:.04,r1:.015};
 const start=assessLumenClearance({segment,t:0,scopeDiameterMm:3.8,mmPerSceneUnit:100});
 assert.equal(start.canAdvance,true);
 assert.ok(start.clearanceMm>0);
 const end=assessLumenClearance({segment,t:1,scopeDiameterMm:3.8,mmPerSceneUnit:100});
 assert.equal(end.canAdvance,false);
 assert.equal(end.diameterMm,3);
 assert.equal(end.wallContactSimulated,false);
 const ct=assessLumenClearance({segment,t:.5,scopeDiameterMm:4,mmPerSceneUnit:250});
 assert.equal(ct.canAdvance,true);
 assert.throws(()=>assessLumenClearance({segment,t:0,scopeDiameterMm:-1,mmPerSceneUnit:100}),/invalid/);
});
