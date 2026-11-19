import test from 'node:test';
import assert from 'node:assert/strict';
import {LOBES,SEGMENTS,FISSURES,VASCULAR_LANDMARKS,ATLAS_PROVENANCE,validateAtlas,describeLobe} from '../anatomy/whole-lung-atlas.js';
test('five lobes and distinct anatomical segment labels allocated exactly once',()=>{
 const c=validateAtlas();
 assert.equal(c.lobes,5);
 assert.equal(c.segments,18);
 assert.equal(LOBES.filter(l=>l.side==='right').length,3);
 assert.equal(LOBES.filter(l=>l.side==='left').length,2);
 assert.equal(LOBES.flatMap(l=>l.segmentIds).length,SEGMENTS.length);
});
test('right lung has horizontal/oblique fissures while left has oblique',()=>{
 assert.equal(FISSURES.length,3);
 assert.equal(describeLobe('RML').fissures.length,2);
 assert.equal(describeLobe('LLL').fissures.length,1);
});
test('vascular landmarks remain topological placeholders without fabricated spatial coordinates',()=>{
 assert.equal(VASCULAR_LANDMARKS.length,6);
 assert.ok(VASCULAR_LANDMARKS.every(x=>x.geometry==='not-provided'));
 assert.equal(ATLAS_PROVENANCE.measured,false);
 assert.equal(describeLobe('LUL').shapeProvenance,'no anatomical surface data');
});
test('anatomical segment variants preserved as metadata, not misleading fixed patient assignments',()=>{
 assert.match(SEGMENTS.find(s=>s.id==='L-S7+8').variantNote,/variable/);
 assert.throws(()=>describeLobe('UNKNOWN'));
});
