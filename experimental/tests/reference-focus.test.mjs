import test from 'node:test';import assert from 'node:assert/strict';
import {CHANNEL,SOURCE,REGIONS,createFocus,readFocus,acceptFocus,regionForBranch} from '../shared/reference-focus.js';
import {readFileSync} from 'node:fs';import {loadAirwayGraph} from '../anatomy/airway-graph.js';
const graph=loadAirwayGraph(JSON.parse(readFileSync(new URL('../anatomy/reference-airways.json',import.meta.url),'utf8')));
test('workspace protocol is explicitly reference-only and versioned',()=>{
 const m=createFocus({regionId:'LLL',branchId:3,origin:'procedural',revision:1});
 assert.equal(m.channel,CHANNEL);assert.equal(m.source,SOURCE);assert.deepEqual(readFocus(m),m);
 assert.equal(regionForBranch(graph,3),'LLL');assert.equal(regionForBranch(graph,0),null);
});
test('invalid, patient-CT and untrusted payloads fail closed',()=>{
 const valid=createFocus({regionId:'RUL',origin:'mechanistic',revision:2});
 for(const patch of [{regionId:'liver'},{source:'patient-ct'},{type:'treatment'},{origin:'external'},{branchId:-2},{revision:NaN}]){
  assert.equal(readFocus({...valid,...patch}),null);
 }
 assert.equal(regionForBranch({schema:'ct-derived',regionOf:()=> 'RUL'},5),null);
});
test('revision ordering forbids stale navigation overriding newer selection',()=>{
 const earlier=createFocus({regionId:'RUL',origin:'procedural',revision:3});
 const later=createFocus({regionId:'LLL',origin:'mechanistic',revision:4});
 assert.deepEqual(acceptFocus(earlier,later),later);
 assert.deepEqual(acceptFocus(later,earlier),later);
 assert.deepEqual(acceptFocus(later,{...earlier,source:'patient-ct'}),later);
});
