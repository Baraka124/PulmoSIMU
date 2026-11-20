import test from 'node:test';
import assert from 'node:assert/strict';
import {validateSurfaceManifest,matchAtlasObjects,atlasCoverage} from '../anatomy/surface-atlas.js';
const fixture={
 schema:'pulmosimu-surface-atlas-v1',datasetId:'synthetic-validation-fixture',
 source:'Generated test geometry — not an anatomical reference',license:'Test fixture only',
 rightsStatement:'Synthetic geometry permitted for automated tests',
 coordinateFrame:'synthetic-local',unit:'mm',patientSpecific:false,
 structures:[
  {id:'RUL',kind:'lobe',objectName:'test_RUL',geometryOrigin:'authored',sourceValidated:false,evidence:'Artificial box for browser integration checks'},
  {id:'RPA',kind:'vessel',objectName:'test_RPA',geometryOrigin:'authored',sourceValidated:false,evidence:'Artificial tube for browser integration checks'}
 ]};
test('surface manifest has finite physical units and explicit provenance',()=>{
 const x=validateSurfaceManifest(fixture);
 assert.equal(x.metresPerSourceUnit,.001);
 assert.equal(x.structures.length,2);
 assert.equal(x.validation.anatomyIndependentlyValidated,false);
 assert.deepEqual(atlasCoverage(x).missingLobes,['RML','RLL','LUL','LLL']);
});
test('importing correctly named GLB objects cannot invent missing anatomy',()=>{
 const x=validateSurfaceManifest(fixture);
 assert.deepEqual(matchAtlasObjects(x,['test_RUL','test_RPA']),{test_RUL:'RUL',test_RPA:'RPA'});
 assert.throws(()=>matchAtlasObjects(x,['test_RUL']),/missing objects/);
});
test('reject ungrounded objects, duplicate mappings and unclear units',()=>{
 assert.throws(()=>validateSurfaceManifest({...fixture,unit:'scene-units'}),/Physical unit/);
 assert.throws(()=>validateSurfaceManifest({...fixture,patientSpecific:true}),/non-patient-specific/);
 assert.throws(()=>validateSurfaceManifest({...fixture,license:''}),/rights statement/);
 assert.throws(()=>validateSurfaceManifest({...fixture,structures:[...fixture.structures,fixture.structures[0]]}),/Duplicate structure/);
 assert.throws(()=>validateSurfaceManifest({...fixture,structures:[{...fixture.structures[0],id:'imaginary-lobe'}]}),/Unknown anatomical/);
});
test('reported validation flags never claim independently verified anatomical accuracy',()=>{
 const x=validateSurfaceManifest({...fixture,structures:fixture.structures.map(s=>({...s,sourceValidated:true}))});
 assert.equal(atlasCoverage(x).sourceValidatedCount,2);
 assert.equal(atlasCoverage(x).anatomicalAccuracyEstablished,false);
});
