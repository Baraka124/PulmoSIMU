import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {loadAirwayGraph} from '../anatomy/airway-graph.js';
import {createDiseaseExperiment} from '../core/disease-engine.js';
import {parameterSensitivity} from '../core/sensitivity-analysis.js';
const graph=loadAirwayGraph(JSON.parse(readFileSync(new URL('../anatomy/reference-airways.json',import.meta.url),'utf8')));
test('sensitivity analysis is finite, ranked and explicitly nonvalidated',()=>{
 const output=parameterSensitivity(graph,createDiseaseExperiment({disease:'asthma'}));
 assert.equal(output.effects.length,4);
 assert.ok(output.effects.every(e=>Number.isFinite(e.relativeSpread)&&e.relativeSpread>=0));
 assert.ok(output.effects.every((e,i)=>i===0||output.effects[i-1].relativeSpread>=e.relativeSpread));
 assert.equal(output.provenance.validated,false);
});
test('repeated uncertainty checks give deterministic numerical results',()=>{
 const state=createDiseaseExperiment({disease:'healthy'});
 assert.deepEqual(parameterSensitivity(graph,state),parameterSensitivity(graph,state));
});
test('invalid sensitivity settings rejected',()=>{
 assert.throws(()=>parameterSensitivity(graph,createDiseaseExperiment(),{perturbationFraction:1}));
 assert.throws(()=>parameterSensitivity(graph,createDiseaseExperiment(),{perturbationFraction:-.1}));
});
