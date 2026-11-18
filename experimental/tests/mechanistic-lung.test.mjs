import test from 'node:test';
import assert from 'node:assert/strict';
import {createState,calculate,advance,withTherapy,withTumor,withRegionModifier,REGIONS} from '../core/mechanistic-lung.js';

test('healthy baseline has normalized ventilation and finite proxies',()=>{
 const o=calculate(createState('healthy'));
 assert.equal(o.regions.length,6-1);
 assert.ok(Math.abs(o.regions.reduce((s,r)=>s+r.ventilationFraction,0)-1)<1e-12);
 assert.ok(o.regions.every(r=>Number.isFinite(r.resistanceIndex)&&r.resistanceIndex>0));
});
test('therapy approaches a mechanism target after elapsed time',()=>{
 const untreated=calculate(createState('asthma')).regions[0].mechanisms.bronchoconstriction;
 const treated=calculate(advance(withTherapy(createState('asthma'),'bronchodilator'),1)).regions[0].mechanisms.bronchoconstriction;
 assert.ok(treated<untreated);
});
test('lung cancer example remains localized to the designated region',()=>{
 const o=calculate(createState('lung_cancer'));
 assert.ok(o.regions.find(x=>x.id==='RUL').mechanisms.tumor>0);
 assert.ok(o.regions.filter(x=>x.id!=='RUL').every(x=>x.mechanisms.tumor===0));
});
test('regional modifiers do not mutate baseline state',()=>{
 const initial=createState('healthy'),modified=withRegionModifier(initial,'LLL','mucus',.7);
 assert.equal(initial.regions.find(r=>r.id==='LLL').modifiers.mucus,undefined);
 assert.ok(calculate(modified).regions.find(r=>r.id==='LLL').mechanisms.mucus>.6);
});
test('regional tumor insertion is spatially restricted',()=>{
 const o=calculate(withTumor(createState(),'LLL',.8));
 assert.equal(o.regions.find(r=>r.id==='LLL').mechanisms.tumor,.8);
 assert.equal(o.regions.find(r=>r.id==='RUL').mechanisms.tumor,0);
});
test('invalid scenarios and negative time are rejected',()=>{
 assert.throws(()=>createState('unknown'));
 assert.throws(()=>advance(createState(),-1));
});
