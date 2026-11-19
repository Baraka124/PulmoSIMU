import test from 'node:test';
import assert from 'node:assert/strict';
import {adaptCtMetadata} from '../anatomy/ct-metadata-adapter.js';
import {validateCtSceneAlignment,ctBoundingBoxFromPoints} from '../anatomy/ct-geometry-validation.js';
const metadata={version:'pneumovr-1.1',transform:{centre_mm:[0,0,0],scale:2},segments:[
 {id:3,parent:-1,gen:0,name:'trachea',r0:.06,r1:.05,pts:[[0,.2,0],[0,0,0]]},
 {id:9,parent:3,gen:1,name:'branch',r0:.045,r1:.03,pts:[[0,0,0],[.2,-.2,.1]]}
]};
const graph=adaptCtMetadata(metadata).graph;
test('co-framed CT GLB bounds and metadata centerlines accepted',()=>{
 const b=ctBoundingBoxFromPoints([[-.4,-.5,-.2],[.4,.5,.3],[0,.1,0]]);
 const alignment=validateCtSceneAlignment(graph,b);
 assert.equal(alignment.centerlinePoints,4);
 assert.equal(alignment.anatomicallyValidated,false);
});
test('far-off-center mesh and malformed bounds are rejected',()=>{
 assert.throws(()=>validateCtSceneAlignment(graph,{min:[20,20,20],max:[21,21,21]}),/same scene frame/);
 assert.throws(()=>validateCtSceneAlignment(graph,{min:[0,0,0],max:[0,1,1]}),/Degenerate/);
 assert.throws(()=>ctBoundingBoxFromPoints([[0,0,NaN],[1,1,1],[1,0,1]]),/finite/);
});
