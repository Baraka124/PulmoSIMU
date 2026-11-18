/**
 * PulmoSIMU R3 — reduced-order, pressure-driven breathing network.
 *
 * A linear resistance/compliance (RC) tree: each measured/illustrative
 * airway branch is a Poiseuille-type resistance, each terminal couples through
 * an explicitly ASSUMED distal resistance to a compliant acinar compartment.
 * Pressure is an imposed sinusoidal transpulmonary *driving surrogate*, not a
 * patient ventilator waveform. This is a conservation-respecting research
 * prototype, not calibrated pulmonary physiology.
 *
 * SI units internally: m, Pa, m^3, s.
 */
import {calculate} from './mechanistic-lung.js';

export const R3_DEFAULTS=Object.freeze({
  sceneUnitMetres:0.1,             // Original source hints at 100mm / scene unit: assumption, not CT calibrated.
  gasViscosityPaS:1.9e-5,
  distalResistancePaSPerM3:2.2e6, // assumed unmeasured distal airway/tissue contribution per terminal
  totalComplianceM3PerPa:1.0e-5,  // ~0.10 L/cmH2O, subject to demonstrative scaling
  breathingFrequencyPerMin:12,
  pressureAmplitudeCmH2O:5,       // 0..5 cmH2O sinusoidally shaped positive driving pressure
  diameterFloorFraction:0.18,
  dtSeconds:0.02
});
const CMH2O_PA=98.0665;
const FINITE=(n)=>Number.isFinite(n);
function assertParams(p){
 for(const [key,val] of Object.entries(p))if(!FINITE(val)||val<=0)throw Error('Invalid physical parameter: '+key);
 if(p.dtSeconds>0.2)throw Error('Time step too large for reference experiment');
}
function segmentLength(pts,scale){
 let n=0;for(let i=1;i<pts.length;i++)n+=Math.hypot(...pts[i].map((v,k)=>(v-pts[i-1][k])*scale));return n;
}
function branchResistance(node,region,params){
 const length=segmentLength(node.points,params.sceneUnitMetres);
 const meanRadius=(node.radiusStart+node.radiusEnd)/2*params.sceneUnitMetres;
 // A localized disease-dependent narrowing proxy; not patient-measured obstruction.
 const narrowing=region?Math.max(params.diameterFloorFraction,region.radiusIndex):1;
 const radius=Math.max(0.0004,meanRadius*narrowing);
 return 8*params.gasViscosityPaS*Math.max(length,0.001)/(Math.PI*Math.pow(radius,4));
}
export function buildNetwork(graph,state,overrides={}){
 const params={...R3_DEFAULTS,...overrides};
 assertParams(params);
 if(!graph?.nodes?.length||!graph?.byId||!graph?.regionOf)throw Error('Validated airway graph required');
 const snapshot=calculate(state),regional=new Map(snapshot.regions.map(x=>[x.id,x]));
 const terminals=graph.nodes.filter(n=>!n.children.length);
 if(!terminals.length)throw Error('Airway graph has no terminals');
 // Region weights distribute total compliance across terminal compartments.
 const counts={};for(const n of terminals){const id=graph.regionOf(n.id);if(!id)throw Error('Unmapped peripheral airway: '+n.id);counts[id]=(counts[id]||0)+1}
 const entries=new Map();
 for(const n of graph.nodes){
  const regionId=graph.regionOf(n.id);
  const region=regionId?regional.get(regionId):null;
  const terminal=n.children.length===0;
  const c=terminal?params.totalComplianceM3PerPa*
    region.weight/counts[regionId]*region.complianceIndex:0;
  const rPeripheral=terminal?params.distalResistancePaSPerM3*
    (region.resistanceIndex/Math.max(0.25,region.complianceIndex)):0;
  entries.set(n.id,{id:n.id,regionId,children:[...n.children],
    airwayResistance:branchResistance(n,region,params),
    peripheralResistance:rPeripheral,compliance:c,terminal});
 }
 return {params,entries,graph,terminals:terminals.map(n=>n.id),
   snapshot,provenance:{model:'linear tree RC, illustrative',units:'SI',validated:false,
   note:'Distal resistance, driving pressure, scene scale and compliance are assumed; not calibrated to any person.'}};
}
export function createBreathingState(network){
 const alveolarPressure={};
 for(const id of network.terminals)alveolarPressure[id]=0;
 return {elapsedSeconds:0,alveolarPressure,netVolumeM3:0,
   totalInhaledM3:0,totalExhaledM3:0};
}
function waveform(t,p){
 const angular=2*Math.PI*p.breathingFrequencyPerMin/60;
 return p.pressureAmplitudeCmH2O*CMH2O_PA*0.5*(1-Math.cos(angular*t));
}
export function stepBreathing(network,previous,dt=network.params.dtSeconds){
 if(!FINITE(dt)||dt<=0||dt>0.2)throw Error('Invalid time step');
 const equivalent=new Map();
 // Terminal reservoir is the compartment pressure from previous step.
 // Each terminal distal pathway has R + dt/C implicit in series.
 function equivalentAt(id){
  const b=network.entries.get(id);
  if(b.terminal){
   const reservoir=previous.alveolarPressure[id];
   const G=1/(b.peripheralResistance+dt/b.compliance);
   const result={G,eqPressure:reservoir};
   equivalent.set(id,result);return result;
  }
  let sumG=0,weighted=0;
  for(const child of b.children){
   const sub=equivalentAt(child),edge=network.entries.get(child);
   const effective=1/(edge.airwayResistance+1/sub.G);
   sumG+=effective;weighted+=effective*sub.eqPressure;
  }
  const result={G:sumG,eqPressure:weighted/sumG};
  equivalent.set(id,result);return result;
 }
 const root=network.graph.root.id,equivalentRoot=equivalentAt(root);
 const driving=waveform(previous.elapsedSeconds+dt,network.params);
 // Tracheal branch resistance is included between mouth driving and its node.
 const rootNode=network.entries.get(root);
 const rootPressure=(driving/rootNode.airwayResistance+
   equivalentRoot.G*equivalentRoot.eqPressure)/
   (1/rootNode.airwayResistance+equivalentRoot.G);
 const flows={},nodePressures={},regionalFlow={};
 const updated={...previous.alveolarPressure};
 function distribute(id,pNode){
  const b=network.entries.get(id);
  nodePressures[id]=pNode;
  if(b.terminal){
   const q=(pNode-previous.alveolarPressure[id])/
      (b.peripheralResistance+dt/b.compliance);
   flows[id]=q;
   updated[id]=previous.alveolarPressure[id]+dt*q/b.compliance;
   regionalFlow[b.regionId]=(regionalFlow[b.regionId]||0)+q;
   return q;
  }
  let total=0;
  for(const childId of b.children){
   const child=network.entries.get(childId),sub=equivalent.get(childId);
   const edgeG=1/child.airwayResistance;
   const pChild=(edgeG*pNode+sub.G*sub.eqPressure)/(edgeG+sub.G);
   const childFlow=distribute(childId,pChild);
   total+=childFlow;
  }
  flows[id]=total;return total;
 }
 const flowM3PerS=distribute(root,rootPressure);
 const netVolume=network.terminals.reduce((sum,id)=>
   sum+network.entries.get(id).compliance*updated[id],0);
 const totalInhaledM3=previous.totalInhaledM3+Math.max(flowM3PerS,0)*dt;
 const totalExhaledM3=previous.totalExhaledM3+Math.max(-flowM3PerS,0)*dt;
 return {state:{elapsedSeconds:previous.elapsedSeconds+dt,
     alveolarPressure:updated,netVolumeM3:netVolume,
     totalInhaledM3,totalExhaledM3},
   pressurePa:driving,flowM3PerS,netVolumeM3:netVolume,
   regionalFlowM3PerS:regionalFlow,nodePressuresPa:nodePressures,
   branchFlowM3PerS:flows,
   conservationResidualM3PerS:flowM3PerS-
      network.terminals.reduce((sum,id)=>sum+flows[id],0)};
}
export function simulateBreathing(network,seconds=20){
 if(!FINITE(seconds)||seconds<=0||seconds>300)throw Error('Invalid simulation duration');
 const samples=[];let state=createBreathingState(network);
 const steps=Math.ceil(seconds/network.params.dtSeconds);
 for(let i=0;i<steps;i++){
  const dt=Math.min(network.params.dtSeconds,seconds-state.elapsedSeconds);
  if(dt<=1e-10)break;
  const output=stepBreathing(network,state,dt);state=output.state;
  samples.push({t:state.elapsedSeconds,pressureCmH2O:output.pressurePa/CMH2O_PA,
   flowLPerS:output.flowM3PerS*1000,volumeL:output.netVolumeM3*1000,
   regionalFlowLPerS:Object.fromEntries(Object.entries(output.regionalFlowM3PerS).map(([k,v])=>[k,v*1000])),
   conservationResidualM3PerS:output.conservationResidualM3PerS});
 }
 const tail=samples.filter(x=>x.t>=Math.max(0,seconds-60/network.params.breathingFrequencyPerMin));
 return {network:{terminalCount:network.terminals.length,branchCount:network.entries.size},
   samples,summary:{
    approximateTidalVolumeL:Math.max(...tail.map(x=>x.volumeL))-Math.min(...tail.map(x=>x.volumeL)),
    peakInspiratoryFlowLPerS:Math.max(...tail.map(x=>x.flowLPerS)),
    peakExpiratoryFlowLPerS:Math.min(...tail.map(x=>x.flowLPerS)),
    maxConservationResidualM3PerS:Math.max(...samples.map(x=>Math.abs(x.conservationResidualM3PerS)))
   },validity:{validated:false,model:'linear reduced-order RC',units:'SI; outputs reported in L, L/s, cmH2O'}};
}
