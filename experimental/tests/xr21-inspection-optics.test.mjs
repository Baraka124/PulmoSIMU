import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
const html=readFileSync(new URL('../../index.html',import.meta.url),'utf8');
const xr=readFileSync(new URL('../xr/procedural-immersive.js',import.meta.url),'utf8');
const root={};runInNewContext(readFileSync(new URL('../xr/procedure-guide.js',import.meta.url),'utf8'),{window:root});
test('route uses source parent graph, distance and all three children',()=>{const segments={0:{id:0,name:'Trachea',children:[1]},1:{id:1,name:'Right main',children:[2,3,4]},2:{id:2,name:'Upper',children:[]},3:{id:3,name:'Middle',children:[]},4:{id:4,name:'Lower',children:[]}};const g=root.PulmoXRProcedureGuide;const route=g.route({segments,segmentId:1,position:.5,lengthMm:()=>100});assert.deepEqual(Array.from(route.labels),['Trachea','Right main']);assert.equal(route.distanceMm,150);assert.equal(g.describe({segments,segmentId:1,segment:segments[1],atBranch:true}).branches.length,3);});
test('source optics and reversible lobe inspection are integrated',()=>{for(const s of ['uSceneScale','uOpticalGlare','getReferenceLobes:()=>[','scopeNear','setEndoscopeLighting:'])assert.ok(html.includes(s),s);for(const s of ['resetLobes()','lobeGrabs[i]','if(inspectMode)setInspection(false)','opticalModes','scope-forward','guide-branch2'])assert.ok(xr.includes(s),s);});
