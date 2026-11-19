import test from 'node:test';
import assert from 'node:assert/strict';
import {createDiseaseExperiment,diseaseReadout,DISEASE_LIBRARY} from '../core/disease-engine.js';
import {calculate} from '../core/mechanistic-lung.js';
import {readFileSync} from 'node:fs';
import {loadAirwayGraph} from '../anatomy/airway-graph.js';
import {buildNetwork,simulateBreathing} from '../core/breathing-network.js';
const graph=loadAirwayGraph(JSON.parse(readFileSync(new URL('../anatomy/reference-airways.json',import.meta.url),'utf8')));
test('R4 disease catalog includes full initial disease spectrum',()=>{
 assert.deepEqual(Object.keys(DISEASE_LIBRARY),['healthy','asthma','copd','bronchiectasis','lung_cancer']);
});
test('bronchiectasis focus changes local inflammation mucus and infection',()=>{
 const state=createDiseaseExperiment({disease:'bronchiectasis',focusRegion:'LLL'});
 const out=calculate(state);
 const focus=out.regions.find(r=>r.id==='LLL');
 const away=out.regions.find(r=>r.id==='RUL');
 assert.ok(focus.mechanisms.mucus>away.mechanisms.mucus);
 assert.ok(focus.mechanisms.infection>away.mechanisms.infection);
 assert.ok(focus.mechanisms.inflammation>away.mechanisms.inflammation);
});
test('lung cancer focus can move without persisting the original RUL tumor',()=>{
 const a=diseaseReadout(createDiseaseExperiment({disease:'lung_cancer',focusRegion:'RML'}));
 assert.equal(a.regional.find(r=>r.id==='RUL').tumor,0);
 assert.ok(a.regional.find(r=>r.id==='RML').tumor>0);
 assert.equal(a.regional.filter(r=>r.tumor>0).length,1);
});
test('regional inflammation perturbation alters airway caliber and resistance',()=>{
 const before=calculate(createDiseaseExperiment({disease:'asthma',focusRegion:'LLL'}));
 const after=calculate(createDiseaseExperiment({disease:'asthma',focusRegion:'LLL',modifiers:{inflammation:.6}}));
 const b=before.regions.find(r=>r.id==='LLL'),a=after.regions.find(r=>r.id==='LLL');
 assert.ok(a.mechanisms.inflammation>b.mechanisms.inflammation);
 assert.ok(a.resistanceIndex>b.resistanceIndex);
 assert.equal(after.regions.find(r=>r.id==='RML').mechanisms.inflammation,before.regions.find(r=>r.id==='RML').mechanisms.inflammation);
});
test('disease modifications propagate into R3 physical network',()=>{
 const baseline=buildNetwork(graph,createDiseaseExperiment({disease:'healthy'}));
 const focal=buildNetwork(graph,createDiseaseExperiment({disease:'lung_cancer',focusRegion:'LLL'}));
 assert.ok(focal.entries.get(22).airwayResistance>baseline.entries.get(22).airwayResistance);
 const output=simulateBreathing(focal,10);
 assert.ok(output.summary.approximateTidalVolumeL>0);
 assert.ok(output.summary.maxConservationResidualM3PerS<1e-10);
});
test('invalid values rejected rather than silently manufacturing disease states',()=>{
 assert.throws(()=>createDiseaseExperiment({disease:'novel'}));
 assert.throws(()=>createDiseaseExperiment({focusRegion:'brain'}));
 assert.throws(()=>createDiseaseExperiment({modifiers:{unknown:.5}}));
 assert.throws(()=>createDiseaseExperiment({modifiers:{mucus:-1}}));
});
