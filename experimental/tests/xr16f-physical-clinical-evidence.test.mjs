/* XR-16F bench evidence, clinical source gates and resilient Quest instrument controls. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
const js=path=>readFileSync(new URL(path,import.meta.url),'utf8');
const w={};
for(const p of ['../xr/physical-instrument-bench.js','../xr/clinical-review-gate.js',
                '../xr/bronchoscopy-station.js'])runInNewContext(js(p),{window:w});
const B=w.PulmoXRInstrumentBench,C=w.PulmoXRClinicalReview,S=w.PulmoXRBronchoscopyStation;
function trace(){
 const samples=Array.from({length:110},(_,i)=>{
   const x=i*.30;
   return {
    tMs:i*60,
    cmdInsertionMm:x,measInsertionMm:x+.8,
    cmdTipHDeg:Math.sin(i/20)*28,measTipHDeg:Math.sin(i/20)*28+1,
    cmdTipVDeg:Math.cos(i/19)*22,measTipVDeg:Math.cos(i/19)*22-.8,
    cmdRollDeg:i*4,measRollDeg:i*4+2,tracking:true
   };
 });
 return {
  version:'pulmosimu-instrument-trace/1',
  provenance:{deviceModel:'Training scope fixture',instrumentId:'Fixture QA A1',
    referenceSensor:'Optical angle encoder',operator:'Bench engineer',
    measuredAt:'2026-11-24T15:00:00Z',measurementSource:'external-bench'},
  units:{time:'ms',insertion:'mm',angles:'deg'},samples
 };
}
function packet(bench){
 return {
  version:'pulmosimu-clinical-review/1',model:'original-reference',
  reviewer:{name:'Clinical reviewer',role:'Pulmonologist',reviewedAt:'2026-11-24'},
  observations:Object.fromEntries(C.checks.map(c=>[c.id,{
   completed:true,evidence:'Reviewed against original source reference and recorded as bench evidence'
  }])),
  limitationsAcknowledged:'This reference simulator is not CT-registered and is not an approved patient device.',
  bench
 };
}
test('external physical instrument trace targets are self-contained and source-limited',()=>{
 const result=B.assess(trace());
 assert.equal(result.status,'BENCH_ENGINEERING_TARGET_MET');
 assert.equal(result.engineeringPassed,true);
 assert.equal(result.clinicalValidated,false);
 assert.equal(result.patientUseApproved,false);
 assert.equal(result.deviceCertified,false);
 assert.ok(result.metrics.insertionP95Mm<2);
 assert.ok(result.metrics.rollP95Deg<3);
 assert.equal(result.metrics.sampleCount,110);
 assert.match(result.interpretation,/external measurements/);
});
test('unsafe, corrupted and uncertified bench telemetry fails closed',()=>{
 assert.equal(B.assess({}).status,'REJECTED');
 const t=trace();delete t.provenance.referenceSensor;
 assert.equal(B.assess(t).engineeringPassed,false);
 assert.ok(B.assess(t).defects.some(x=>x.includes('REFERENCE')));
 const s=trace();s.provenance.measurementSource='simulated-Quest-pose';
 assert.equal(B.assess(s).status,'REJECTED');
 const z=trace();z.samples[5].tMs=z.samples[4].tMs;
 assert.match(B.assess(z).defects[0],/TIMESTAMP/);
 const bad=trace();bad.samples[20].tracking=false;
 assert.equal(B.assess(bad).status,'BENCH_TARGET_NOT_MET');
 assert.ok(B.assess(bad).defects.includes('TRACKING_LOSS_RECORDED'));
 const err=trace();err.samples.forEach(x=>x.measInsertionMm+=70);
 assert.equal(B.assess(err).engineeringPassed,false);
 const short=trace();short.samples=short.samples.slice(0,8);
 assert.equal(B.assess(short).status,'REJECTED');
});
test('clinical review packet is never a certification, even if complete',()=>{
 const bench=B.assess(trace());
 const result=C.assess(packet(bench));
 assert.equal(result.readyForIndependentReview,true);
 assert.equal(result.clinicalValidated,false);
 assert.equal(result.patientSpecificValidated,false);
 assert.equal(result.regulatoryApproval,false);
 const p=packet(bench);p.model='patient-CT';
 assert.equal(C.assess(p).readyForIndependentReview,false);
 assert.ok(C.assess(p).missing.includes('REFERENCE_MODEL_ONLY_NO_PATIENT_SOURCE'));
 const missing=packet(bench);missing.observations.landmarks.completed=false;
 assert.equal(C.assess(missing).readyForIndependentReview,false);
 const noBench=packet(null);assert.equal(C.assess(noBench).readyForIndependentReview,false);
});
test('NaN and missing gamepad axes cannot corrupt original bronchoscope pose',()=>{
 for(const bad of [NaN,Infinity,-Infinity,undefined,null,'non-numeric']){
  assert.equal(S.tipStep(.6,bad,.016),.6);
  assert.ok(Math.abs(S.rollStep(.2,bad,.016)-.2)<1e-12);
  assert.equal(S.velocity(bad,14),0);
 }
 assert.equal(S.tipStep(NaN,.5,.016)>0,true);
 assert.equal(S.rateLimit(NaN,Infinity,.016),0);
 assert.equal(S.tipStep(.4,0,.1),.4);
});
test('video governor uses actual CPU submission, never fabricates GPU frame time',()=>{
 const g=S.videoCostGovernor();
 for(let i=0;i<15;i++)assert.equal(g.note(20,'detail'),null);
 assert.equal(g.note(20,'detail'),'balanced');
 for(let i=0;i<15;i++)g.note(23,'balanced');
 assert.equal(g.note(23,'balanced'),'economy');
 for(let i=0;i<16;i++)assert.equal(g.note(40,'economy'),null);
 const status=g.snapshot();
 assert.equal(status.downgrades,2);
 assert.match(status.measurement,/NO_GPU_TIMING/);
});
test('the existing original-engine XR runtime integrates recovery and evidence contracts',()=>{
 const runtime=js('../xr/procedural-immersive.js');
 const launcher=js('../../quest-short/index.html');
 const validation=js('../validation-lab.html');
 for(const token of [
  "monitorRetryAt=now+Math.min(8000",
  "monitorRecoveries++;monitorFailures=0",
  "const controller=trackedController(i)",
  "health.event('controllerTrackingLost')",
  "instrument.videoCostGovernor()",
  "monitorBudget.note(monitorCpuMs,station.snapshot().video)",
  "clinicalAndPhysicalReview:'EXTERNAL_EVIDENCE_REQUIRED_LOCAL_VALIDATION_LAB'",
  "monitorSource:'same-renderer-offscreen'",
  "rBro.render(scene,camScope)",
  "getMeshMode()==='patient'"
 ])assert.ok(runtime.includes(token),'Missing actual-engine contract: '+token);
 assert.match(launcher,/clinical &amp; instrument verification/i);
 assert.ok(validation.includes('physical-instrument-bench.js'));
 assert.ok(validation.includes('clinical-review-gate.js'));
 assert.ok(!validation.includes('fetch('),'Validation lab must be local only');
});
