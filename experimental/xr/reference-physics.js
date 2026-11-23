/**
 * PulmoSIMU XR-09 — existing R3 reference breathing solver adapter.
 *
 * One solver, two workspaces: reuse R3 from the mechanistic laboratory
 * inside the ORIGINAL procedural XR. The 18-branch source is an uncalibrated
 * procedural snapshot. No CT-to-R3 registration, simulated patient outcome,
 * mesh deformation or clinical pressure/flow claim is implied.
 */
import {fetchReferenceAirwayGraph} from '../anatomy/airway-graph.js';
import {createState} from '../core/mechanistic-lung.js';
import {buildNetwork,createBreathingState,stepBreathing} from '../core/breathing-network.js';

const finite=Number.isFinite;
export function createXRReferencePhysicsForGraph(graph){
 if(!graph||graph.schema!=='pulmosimu-airway-reference-r2'||
    graph.nodes?.length!==18||graph.root?.id!==0)
  throw Error('XR R3 requires the canonical original procedural airway snapshot');
 const network=buildNetwork(graph,createState('healthy'));
 let simulation=createBreathingState(network);
 let remainder=0,last=null;
 const validation=Object.freeze({
  solver:'R3 linear RC network',source:'original-procedural-SEGS-snapshot-r2',
  geometry:'uncalibrated procedural reference',clinicalValidated:false,
  patientSpecific:false,fullCFD:false,meshMotionCoupled:false,
  calibration:'assumed scene scale, distal resistance, compliance and drive',
  units:Object.freeze({pressure:'Pa',flow:'m3/s',volume:'m3',time:'s'})
 });
 function read(){
  return Object.freeze({
   schema:'pulmosimu.xr.reference-physics.v1',validation,
   tSeconds:simulation.elapsedSeconds,
   pressurePa:last?.pressurePa??0,
   flowM3PerS:last?.flowM3PerS??0,
   volumeM3:last?.netVolumeM3??0,
   conservationResidualM3PerS:last?.conservationResidualM3PerS??0,
   regionalFlowM3PerS:Object.freeze({...last?.regionalFlowM3PerS}),
   terminalCount:network.terminals.length,
   airwayCount:network.entries.size
  });
 }
 function step(dt){
  if(!finite(dt)||dt<0||dt>.25)throw Error('Invalid headset physics time increment');
  remainder+=dt;
  const stepSize=network.params.dtSeconds;
  let count=0;
  while(remainder+1e-10>=stepSize&&count<12){
   const output=stepBreathing(network,simulation,stepSize);
   if(![output.flowM3PerS,output.pressurePa,output.netVolumeM3,
     output.conservationResidualM3PerS].every(finite))
     throw Error('R3 solver generated nonfinite state');
   simulation=output.state;last=output;remainder-=stepSize;count++;
  }
  return read();
 }
 function reset(){
  simulation=createBreathingState(network);remainder=0;last=null;return read();
 }
 return Object.freeze({read,step,reset,validation});
}
export async function loadXRReferencePhysics(){
 const graph=await fetchReferenceAirwayGraph();
 return createXRReferencePhysicsForGraph(graph);
}
