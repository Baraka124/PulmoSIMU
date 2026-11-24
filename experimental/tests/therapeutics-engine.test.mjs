import test from 'node:test';
import assert from 'node:assert/strict';
import {createTherapyExperiment,responseProfile,INTERVENTIONS,PHENOTYPES} from '../core/therapeutics-engine.js';
import {calculate} from '../core/mechanistic-lung.js';
import {loadAirwayGraph} from '../anatomy/airway-graph.js';
import {buildNetwork,simulateBreathing} from '../core/breathing-network.js';
import {readFileSync} from 'node:fs';
const graph=loadAirwayGraph(JSON.parse(readFileSync(new URL('../anatomy/reference-airways.json',import.meta.url),'utf8')));
const near=(a,b,tol=1e-12)=>assert.ok(Math.abs(a-b)<tol,`Expected ${a}≈${b}`);

test('therapy catalogue includes independent biologic pathways',()=>{
 for(const k of ['anti_ige','anti_il5','anti_il5r','anti_il4r','anti_tslp','bronchodilator','ics'])assert.ok(INTERVENTIONS[k]);
});
test('zero elapsed time and zero response leave disease baseline unchanged',()=>{
 for(const cfg of [{intervention:'anti_il5',days:0},{intervention:'anti_il4r',days:120,responseScale:0},{intervention:'none',days:120}]){
  const e=createTherapyExperiment({disease:'asthma',...cfg});
  for(const [i,r] of e.before.regions.entries()){
    near(r.resistanceIndex,e.after.regions[i].resistanceIndex);
    near(r.mechanisms.inflammation,e.after.regions[i].mechanisms.inflammation);
  }
 }
});
test('eosinophilic pathway sensitivity differs by illustrative phenotype',()=>{
 const hi=responseProfile('anti_il5','eosinophilic',80);
 const lo=responseProfile('anti_il5','type2_low',80);
 assert.ok(hi.mechanismReduction.inflammation>lo.mechanismReduction.inflammation);
 assert.ok(hi.phenotypeSensitivity>lo.phenotypeSensitivity);
});
test('unrelated treatment does not reduce structural tumor or emphysema',()=>{
 const e=createTherapyExperiment({disease:'lung_cancer',focusRegion:'LLL',intervention:'anti_il4r',days:80});
 near(e.before.regions.find(x=>x.id==='LLL').mechanisms.tumor,e.after.regions.find(x=>x.id==='LLL').mechanisms.tumor);
 near(e.comparison.mechanisms.tumor.delta,0);
});
test('bronchodilator changes only smooth muscle mechanism in asthma',()=>{
 const e=createTherapyExperiment({disease:'asthma',intervention:'bronchodilator',days:1});
 assert.ok(e.comparison.mechanisms.bronchoconstriction.delta<0);
 near(e.comparison.mechanisms.inflammation.delta,0);
 assert.ok(e.comparison.resistanceIndex.delta<0);
});
test('comparison states propagate independently into R3',()=>{
 const e=createTherapyExperiment({disease:'asthma',intervention:'bronchodilator',days:1});
 const a=buildNetwork(graph,e.untreated),b=buildNetwork(graph,e.treated);
 assert.ok(b.entries.get(24).airwayResistance<a.entries.get(24).airwayResistance);
 for(const net of [a,b])assert.ok(simulateBreathing(net,10).summary.maxConservationResidualM3PerS<1e-10);
});
test('invalid interventions, phenotypes, response and time rejected',()=>{
 for(const cfg of [{intervention:'unknown'},{phenotype:'unknown'},{responseScale:1.2},{days:-1}])assert.throws(()=>createTherapyExperiment(cfg));
});
test('treated state is deterministic and input state never mutated',()=>{
 const e=createTherapyExperiment({disease:'asthma',phenotype:'type2_high',intervention:'anti_il4r',days:60,focusRegion:'RLL'});
 assert.deepEqual(calculate(e.treated),e.after);
 assert.equal(e.untreated.therapy,'none');
 assert.equal(e.treated.therapy,'none'); // prevents legacy therapeutic effect twice
});
