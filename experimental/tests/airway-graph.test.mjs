import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {loadAirwayGraph} from '../anatomy/airway-graph.js';

const raw=JSON.parse(readFileSync(new URL('../anatomy/reference-airways.json',import.meta.url),'utf8'));
test('faithfully mapped PulmoSIMU procedural reference tree has 18 connected branches',()=>{
 const graph=loadAirwayGraph(raw);
 assert.equal(graph.nodes.length,18);
 assert.equal(graph.root.name,'Trachea');
 assert.equal(graph.chain(29).map(x=>x.id).join(','),'0,4,7,29');
});
test('RUL/RML/RLL and LUL/LLL map to their source parents',()=>{
 const graph=loadAirwayGraph(raw);
 for(const [id,region] of [[2,'LUL'],[3,'LLL'],[5,'RUL'],[6,'RML'],[7,'RLL'],[20,'LUL'],[29,'RLL']])assert.equal(graph.regionOf(id),region);
 assert.equal(graph.regionOf(0),null);
});
test('reject broken connectivity instead of silently producing incorrect anatomy',()=>{
 const copy=structuredClone(raw);copy.nodes.find(x=>x.id===1).children.push(999);
 assert.throws(()=>loadAirwayGraph(copy),/Missing child/);
});
test('source geometry remains annotated as procedural rather than patient CT',()=>{
 assert.match(raw.anatomicalFidelity,/procedural/i);
 assert.equal(raw.nodes[0].points.length,6);
});
