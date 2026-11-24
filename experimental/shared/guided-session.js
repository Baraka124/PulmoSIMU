/**
 * PulmoSIMU S3 - workstation-guided research session.
 * A provenance-first recorder, not an automatic VR performance score.
 * No patient image, CT mark, identity or patient-derived geometry is collected.
 */
export const SCHEMA='pulmosimu-guided-research-session-v1';
export const STEPS=Object.freeze([
 Object.freeze({id:'source',workspace:'source',title:'Inspect the anatomical source',
  instruction:'Load the local CT, derived GLB and matching manifest. Review all relevant slices; mark source geometry concerns and optionally measure in millimetres.'}),
 Object.freeze({id:'procedure',workspace:'procedure',title:'Rehearse reference bronchoscopy',
  instruction:'Use the original procedure workspace and its actual bronchoscopy view. Navigate reference airway branches and inspect the instrument state.'}),
 Object.freeze({id:'immersive',workspace:'xr',title:'Enter the spatial procedure',
  instruction:'Open the original-engine immersive scene on Quest. Test controller input, anatomical framing and the separate endoscopic monitor in the headset.'}),
 Object.freeze({id:'physiology',workspace:'mechanisms',title:'Run the R3 respiratory reference',
  instruction:'Run the existing 18-branch reduced-order R3 solver to record a reproducible numerical reference; this model is uncalibrated and not linked to CT or patient tissue.'}),
 Object.freeze({id:'report',workspace:'mechanisms',title:'Export a reproducible session record',
  instruction:'Review completed checkpoints and the R3 numerical evidence, then export JSON. No CT, GLB, images or simulated patient claims are included.'})
]);
const validId=new Set(STEPS.map(s=>s.id));
const finite=Number.isFinite;
function checkedClock(clock){
 const d=clock();
 if(typeof d!=='string'||!Number.isFinite(Date.parse(d)))throw Error('Invalid session time');
 return d;
}
export function createGuidedSession({clock=()=>new Date().toISOString(),sessionId}={}){
 const started=checkedClock(clock);
 const id=sessionId??('research-'+started.replace(/[^0-9]/g,'').slice(0,14));
 if(typeof id!=='string'||!/^[a-zA-Z0-9-]{5,80}$/.test(id))throw Error('Invalid session identifier');
 let step=0,completed=new Map(),r3=null;
 const get=()=>Object.freeze({id,started,step,total:STEPS.length,
  current:STEPS[step],completed:Object.freeze([...completed.keys()]),
  hasPhysics:Boolean(r3)});
 function move(index){
  if(!Number.isInteger(index)||index<0||index>=STEPS.length)throw Error('Invalid guide step');
  step=index;return get();
 }
 function complete(index=step){
  if(!Number.isInteger(index)||index<0||index>=STEPS.length)throw Error('Invalid guide step');
  completed.set(STEPS[index].id,checkedClock(clock));return get();
 }
 function attachR3(data){
  if(!data||data.model!=='linear reduced-order RC'||data.graphSource!=='original-procedural-SEGS-snapshot-r2'||
    data.clinicalValidated!==false||data.patientSpecific!==false||
    !Number.isInteger(data.branchCount)||data.branchCount!==18||
    !Number.isInteger(data.terminalCount)||data.terminalCount<1||
    !finite(data.simulatedSeconds)||data.simulatedSeconds<=0||
    !finite(data.breathsPerMin)||data.breathsPerMin<=0||
    !finite(data.driveCmH2O)||data.driveCmH2O<=0||
    !finite(data.approximateTidalVolumeL)||data.approximateTidalVolumeL<0||
    !finite(data.peakInspiratoryFlowLPerS)||
    !finite(data.maxConservationResidualM3PerS)||data.maxConservationResidualM3PerS<0)
      throw Error('R3 evidence must originate from the canonical uncalibrated reference solver');
  r3=Object.freeze({...data,recordedAt:checkedClock(clock)});
  return get();
 }
 function report(){
  const checkpoints=STEPS.map(s=>({
   id:s.id,title:s.title,workspace:s.workspace,
   selfReportedComplete:completed.has(s.id),
   completedAt:completed.get(s.id)||null
  }));
  return {schema:SCHEMA,id,started,exportedAt:checkedClock(clock),
   source:'PulmoSIMU reference research workspaces',
   studyDataIncluded:false,clinicalValidated:false,
   checkpointEvidence:'USER-ATTESTED; NOT TELEMETRY OR HARDWARE VERIFICATION',
   limitations:['Procedural and Quest checkpoint completion is manually recorded, not independently verified.',
    'The R3 solver is an illustrative reduced-order RC model with assumed physical parameters.',
    'The report contains no CT source data, image review marks or patient measurements.'],
   checkpoints,r3Reference:r3?{...r3}:null};
 }
 function reset(){
  step=0;completed=new Map();r3=null;return get();
 }
 return Object.freeze({get,move,complete,attachR3,report,reset});
}
export function makeR3Evidence(network,numerical,seconds){
 if(network?.graph?.schema!=='pulmosimu-airway-reference-r2'||
  !numerical||numerical.validity?.validated!==false||
  numerical.network?.branchCount!==18)throw Error('Only the original R3 reference run is reportable');
 const p=network.params,s=numerical.summary;
 return {
  model:'linear reduced-order RC',
  graphSource:'original-procedural-SEGS-snapshot-r2',
  clinicalValidated:false,patientSpecific:false,
  branchCount:numerical.network.branchCount,
  terminalCount:numerical.network.terminalCount,
  simulatedSeconds:seconds,breathsPerMin:p.breathingFrequencyPerMin,
  driveCmH2O:p.pressureAmplitudeCmH2O,
  approximateTidalVolumeL:s.approximateTidalVolumeL,
  peakInspiratoryFlowLPerS:s.peakInspiratoryFlowLPerS,
  maxConservationResidualM3PerS:s.maxConservationResidualM3PerS
 };
}
