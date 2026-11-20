import test from 'node:test';import assert from 'node:assert/strict';
import {STEPS,createWelcomeSequence} from '../xr/experience-sequence.js';
test('first-use headset guide is finite, safe and complete',()=>{
 const seq=createWelcomeSequence();
 assert.equal(seq.get().active,true);assert.equal(seq.get().index,0);
 assert.equal(seq.get().total,5);
 assert.equal(STEPS.every(s=>typeof s.title==='string'&&s.text&&s.hint),true);
 for(let i=1;i<STEPS.length;i++){assert.equal(seq.next(),true);assert.equal(seq.get().index,i)}
 assert.equal(seq.next(),false);assert.equal(seq.get().active,false);
 assert.equal(seq.next(),false);
 seq.restart();assert.equal(seq.get().index,0);assert.equal(seq.get().active,true);
 seq.skip();assert.equal(seq.get().active,false);
 assert.equal(seq.previous(),false);
});
test('headset introduction never performs forced locomotion or patient diagnosis',()=>{
 for(const x of STEPS){
   assert.doesNotMatch(x.text,/(teleport|walk around|diagnose|medication|patient record)/i);
 }
 assert.match(STEPS[1].text,/SIDE GRIP/);
 assert.match(STEPS[3].text,/TRIGGER/);
});
