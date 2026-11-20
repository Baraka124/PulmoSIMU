import test from 'node:test';import assert from 'node:assert/strict';
import {anatomyObjectRecord,tagAnatomyObject,canReportMetric,REFERENCE_SOURCE,SCHEMATIC_SOURCE,LOBES} from '../anatomy/object-contract.js';
test('source-identified airway and lobar meshes use stable nonpatient IDs',()=>{
 const b=anatomyObjectRecord({sourceId:REFERENCE_SOURCE,role:'airway',regionId:'RUL',branchId:5});
 const l=anatomyObjectRecord({sourceId:SCHEMATIC_SOURCE,role:'lobe',regionId:'RUL'});
 assert.equal(b.id,'pulmosimu-airway-reference-r2:airway:5');
 assert.equal(l.id,'pulmosimu-schematic-lobes-r1:lobe:RUL');
 assert.equal(b.physicalScaleValidated,false);
 assert.equal(l.measurementsAllowed,false);
 assert.equal(LOBES.length,5);
 const mesh={userData:{kind:'reference-airway'}};
 tagAnatomyObject(mesh,b);
 assert.equal(mesh.userData.kind,'reference-airway');
 assert.equal(mesh.userData.anatomyRecord.sourceId,REFERENCE_SOURCE);
});
test('uncalibrated reference cannot quietly report lengths, vessel diameter, CFD or therapeutic response',()=>{
 const b=anatomyObjectRecord({sourceId:REFERENCE_SOURCE,role:'airway',branchId:0});
 for(const metric of ['diameter','length','volume','resistance','compliance','therapy'])
  assert.equal(canReportMetric(b,metric),false);
 const forged={...b,measurementsAllowed:true,physicalScaleValidated:true};
 assert.equal(canReportMetric(forged,'resistance'),false);
 assert.equal(canReportMetric(forged,'length'),false,'Forged flags cannot override source provenance');
 assert.throws(()=>anatomyObjectRecord({sourceId:'patient-unknown',role:'airway',branchId:0}));
 assert.throws(()=>anatomyObjectRecord({sourceId:REFERENCE_SOURCE,role:'airway',branchId:-1}));
 assert.throws(()=>anatomyObjectRecord({sourceId:SCHEMATIC_SOURCE,role:'lobe',regionId:'UNKNOWN'}));
});
