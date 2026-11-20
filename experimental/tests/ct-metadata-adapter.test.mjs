import test from 'node:test';
import assert from 'node:assert/strict';
import {adaptCtMetadata,sceneToPatientMm,patientMmToScene} from '../anatomy/ct-metadata-adapter.js';

const transform={centre_mm:[100,-250,40],scale:4,target_span:2.4};
const sample={version:'pneumovr-1.1',transform,
 segments:[
 {id:19,parent:-1,name:'Trachea',gen:0,pts:[[0,1,0],[0,0,0]],r0:.08,r1:.07},
 {id:31,parent:19,name:'R Main Bronchus',gen:1,pts:[[0,0,0],[1,0,0]],r0:.06,r1:.05},
 {id:55,parent:19,name:'L Main Bronchus',gen:1,pts:[[0,0,0],[-1,0,0]],r0:.06,r1:.05}
 ]};
test('CT point inverse transform roundtrips physical millimetres',()=>{
 for(const pt of [[100,-250,40],[202.5,-57,17.7],[-100,201,320]]){
 const scene=patientMmToScene(pt,transform),back=sceneToPatientMm(scene,transform);
 back.forEach((v,i)=>assert.ok(Math.abs(v-pt[i])<1e-9));
 }
});
test('CT pipeline scaling gives physical metres per scene unit, retaining source IDs',()=>{
 const {graph,provenance}=adaptCtMetadata(sample);
 assert.equal(provenance.sceneUnitMetres,.25);
 assert.equal(provenance.kind,'ct-derived');
 assert.equal(provenance.validated,false);
 assert.deepEqual(graph.chain(1).map(n=>n.sourceId),[19,31]);
 assert.equal(graph.nodes.length,3);
 assert.equal(graph.get(1).radiusStart*.25,.015);
});
test('pathology metadata and patient identifiers are not retained in adapter output',()=>{
 const m={...sample,patient:{PatientName:'PRIVATE',PatientID:'987'},pathologies:[{label:'PRIVATE'}]};
 const result=JSON.stringify(adaptCtMetadata(m));
 assert.ok(!result.includes('PRIVATE'));
 assert.ok(!result.includes('987'));
});
test('CT data require scaling provenance, rooted parent tree and finite geometry',()=>{
 assert.throws(()=>adaptCtMetadata({...sample,transform:{...transform,scale:0}}),/scale/);
 assert.throws(()=>adaptCtMetadata({...sample,segments:[...sample.segments,{...sample.segments[1],id:99,parent:999}]}),/single proximal root/);
 assert.throws(()=>adaptCtMetadata({...sample,segments:sample.segments.map((x,i)=>i===1?{...x,r0:-1}:x)}),/radii/);
});
