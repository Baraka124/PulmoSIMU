import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
import {createGuidedSession,makeR3Evidence,STEPS,SCHEMA} from '../shared/guided-session.js';
import {loadAirwayGraph} from '../anatomy/airway-graph.js';
import {createState} from '../core/mechanistic-lung.js';
import {buildNetwork,simulateBreathing} from '../core/breathing-network.js';
const guideHtml=readFileSync(new URL('../../studio.html',import.meta.url),'utf8');
const sourceHtml=readFileSync(new URL('../anatomy-inspector.html',import.meta.url),'utf8');
const sourceJs=readFileSync(new URL('../anatomy/inspector-app.js',import.meta.url),'utf8');
const reviewJs=readFileSync(new URL('../anatomy/source-review.js',import.meta.url),'utf8');
const original=readFileSync(new URL('../../index.html',import.meta.url),'utf8');
const sourcePage={};runInNewContext(reviewJs,{window:sourcePage});
const identityQA={
 source_sha256:'a'.repeat(64),mesh_sha256:'b'.repeat(64),
 source_shape_vox:[30,30,30],
 sform_RAS_mm:[[0,-2,0,10],[1.5,0,0,-20],[0,0,4,-300],[0,0,0,1]],
 model_from_ras_mm:[[.001,0,0,0],[0,.001,0,0],[0,0,.001,0],[0,0,0,1]]
};
test('S1: oblique anisotropic CT distance uses RAS affine, not voxel-index distance',()=>{
 const review=sourcePage.PulmoCTReview;
 const study=review.identity({ctSha:identityQA.source_sha256,
  meshSha:identityQA.mesh_sha256,qa:identityQA,headerValid:true});
 const metric=review.measureRAS(study,[2,4,5],[5,8,11]);
 assert.equal(metric.schema,'pulmosimu-ct-metric-v1');
 assert.equal(metric.distance_mm,25.695);
 assert.match(metric.interpretation,/NOT ANATOMICAL SEGMENTATION/);
 assert.throws(()=>review.measureRAS(study,[0,0,0],[31,4,5]),/outside/);
 assert.throws(()=>review.measureRAS(null,[0,0,0],[1,0,0]),/Verified source/);
 for(const control of ['metric-a','metric-b','metric-clear','metric-result'])
  assert.ok(sourceHtml.includes('id="'+control+'"'),'missing '+control);
 for(const check of ['window.PulmoCTReview.measureRAS','function renderMetricStatus',
  'drawReviewPoints(plane,ctx,w,h)','state.metric={a:null,b:null}'])
  assert.ok(sourceJs.includes(check),'missing '+check);
});
test('S2: guided session records self-attested checkpoints without claiming hardware telemetry',()=>{
 let tick=0;
 const session=createGuidedSession({clock:()=>new Date(Date.UTC(2026,10,24,2,0,tick++)).toISOString(),
  sessionId:'research-sim1'});
 assert.deepEqual(STEPS.map(s=>s.workspace),['source','procedure','xr','mechanisms','mechanisms']);
 session.complete();session.move(2);session.complete();
 const data=session.report();
 assert.equal(data.schema,SCHEMA);
 assert.equal(data.clinicalValidated,false);
 assert.equal(data.studyDataIncluded,false);
 assert.match(data.checkpointEvidence,/USER-ATTESTED/);
 assert.deepEqual(data.checkpoints.filter(x=>x.selfReportedComplete).map(x=>x.id),
  ['source','immersive']);
 assert.equal(data.r3Reference,null);
 assert.throws(()=>session.move(99),/Invalid/);
 session.reset();assert.equal(session.get().completed.length,0);
 for(const item of ['id="guideToggle"','id="guidePanel"','id="guideOpen"',
  'id="guideDone"','id="guideRunR3"','id="guideExport"'])
  assert.ok(guideHtml.includes(item),'Missing guided interaction '+item);
 assert.ok(guideHtml.includes('activate(target)'),'guide must use original workspace routes');
 assert.ok(guideHtml.includes("src:'./procedural-workspace.html'"),'original procedure must stay canonical');
 assert.ok(guideHtml.includes("src:'./index.html?entry=vr'"),
  'XR source must remain canonical');
});
test('S3: R3 evidence comes from original 18-branch numerical solver, with exact provenance',()=>{
 const data=JSON.parse(readFileSync(new URL('../anatomy/reference-airways.json',import.meta.url),'utf8'));
 const graph=loadAirwayGraph(data),network=buildNetwork(graph,createState('healthy'));
 const numerical=simulateBreathing(network,10);
 const evidence=makeR3Evidence(network,numerical,10);
 assert.equal(evidence.branchCount,18);
 assert.equal(evidence.clinicalValidated,false);
 assert.equal(evidence.patientSpecific,false);
 assert.equal(evidence.simulatedSeconds,10);
 assert.ok(evidence.approximateTidalVolumeL>=0);
 assert.ok(Number.isFinite(evidence.maxConservationResidualM3PerS));
 const session=createGuidedSession({sessionId:'research-sim2'});
 session.attachR3(evidence);
 assert.equal(session.report().r3Reference.graphSource,'original-procedural-SEGS-snapshot-r2');
 assert.throws(()=>session.attachR3({...evidence,clinicalValidated:true}),/canonical uncalibrated/);
 assert.throws(()=>session.attachR3({...evidence,branchCount:99}),/canonical uncalibrated/);
 assert.ok(guideHtml.includes("simulateBreathing(network,seconds)"));
 assert.ok(guideHtml.includes("makeR3Evidence(network,numerical,seconds)"));
 assert.ok(guideHtml.includes("studyDataIncluded")===false,'No separate patient-data export in Studio');
 // The three-step build is additive: original bronchoscopy and SEGS stay untouched.
 for(const piece of ['function updateScope()','function advance(dt)','window.exportSession=exportSession'])
   assert.ok(original.includes(piece),'Original procedural source invariant changed: '+piece);
});
