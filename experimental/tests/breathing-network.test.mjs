import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {loadAirwayGraph} from '../anatomy/airway-graph.js';
import {createState,withTherapy,advance,withTumor} from '../core/mechanistic-lung.js';
import {buildNetwork,createBreathingState,stepBreathing,simulateBreathing} from '../core/breathing-network.js';
const raw=JSON.parse(readFileSync(new URL('../anatomy/reference-airways.json',import.meta.url),'utf8'));
const graph=loadAirwayGraph(raw);
const close=(a,b,tol=1e-7)=>assert.ok(Math.abs(a-b)<=tol,`Expected ${a} ~ ${b}`);

test('SI-unit respiratory network has connected terminal compartments',()=>{
 const net=buildNetwork(graph,createState('healthy'));
 assert.equal(net.entries.size,18);
 assert.equal(net.terminals.length,10);
 assert.ok(net.terminals.every(id=>net.entries.get(id).compliance>0));
 assert.ok(net.terminals.every(id=>net.entries.get(id).peripheralResistance>0));
});
test('mass flow conserved at root and leaves throughout cycle',()=>{
 const net=buildNetwork(graph,createState('healthy'));
 let state=createBreathingState(net);
 for(let i=0;i<350;i++){
  const step=stepBreathing(net,state);state=step.state;
  assert.ok(Number.isFinite(step.flowM3PerS));
  close(step.conservationResidualM3PerS,0,1e-11);
  const terminalFlow=net.terminals.reduce((s,id)=>s+step.branchFlowM3PerS[id],0);
  close(step.flowM3PerS,terminalFlow,1e-11);
 }
});
test('approximately periodic breathing with physiologically sized positive demonstrative output',()=>{
 const net=buildNetwork(graph,createState('healthy'));
 const out=simulateBreathing(net,20);
 assert.ok(out.summary.approximateTidalVolumeL>0.05);
 assert.ok(out.summary.approximateTidalVolumeL<3.0);
 assert.ok(out.summary.peakInspiratoryFlowLPerS>0);
 assert.ok(out.summary.peakExpiratoryFlowLPerS<0);
 assert.ok(out.summary.maxConservationResidualM3PerS<1e-10);
});
test('higher distal resistance reduces tidal volume under the same driving pressure',()=>{
 const s=createState('healthy');
 const a=simulateBreathing(buildNetwork(graph,s),20).summary.approximateTidalVolumeL;
 const b=simulateBreathing(buildNetwork(graph,s,{distalResistancePaSPerM3:15e6}),20).summary.approximateTidalVolumeL;
 assert.ok(b<a,`Expected high resistance tidal volume ${b} < ${a}`);
});
test('regional tumor imposes resistance increase in targeted airway branches',()=>{
 const a=buildNetwork(graph,createState('healthy'));
 const b=buildNetwork(graph,withTumor(createState('healthy'),'LLL',.95));
 assert.ok(b.entries.get(22).airwayResistance>a.entries.get(22).airwayResistance);
 assert.equal(b.entries.get(24).airwayResistance,a.entries.get(24).airwayResistance);
});
test('therapy affects state-dependent resistance under controlled scenario',()=>{
 const untreated=buildNetwork(graph,createState('asthma'));
 const treated=buildNetwork(graph,advance(withTherapy(createState('asthma'),'bronchodilator'),1));
 assert.ok(treated.entries.get(24).airwayResistance<untreated.entries.get(24).airwayResistance);
});
test('reject nonphysical parameters',()=>{
 assert.throws(()=>buildNetwork(graph,createState(),{gasViscosityPaS:-1}));
 assert.throws(()=>simulateBreathing(buildNetwork(graph,createState()),-2));
});
