import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
const root={};runInNewContext(readFileSync(new URL('../xr/calibrated-control.js',import.meta.url),'utf8'),{window:root});
function fixture(){let s={armed:false,corrected:null};const control=root.PulmoXRCalibratedControl.create({calibration:{arm(){s={armed:true,corrected:null}},stop(){s={armed:false,corrected:null}},snapshot(){return s}}});return {control,set(v){s={armed:true,corrected:v}}}}
const v=(insertionMm,shaftRollDeg=0,tipPitchDeg=0,tipYawDeg=0)=>({insertionMm,shaftRollDeg,tipPitchDeg,tipYawDeg});
const args={now:100,deadman:true,reference:true,procedure:true,visible:true,connected:true,segmentLengthMm:100,deflectionMax:30};
test('deadman baseline, calibrated delta, release and no backlog',()=>{const {control,set}=fixture();control.arm();set(v(10));assert.equal(control.step(args),null);set(v(12,3,2,1));const cmd=control.step({...args,now:150});assert.equal(cmd.insertionT,.02);assert.equal(cmd.tipH,1/30);assert.equal(cmd.tipV,2/30);set(v(13));assert.equal(control.step({...args,now:170,deadman:false}),null);set(v(20));assert.equal(control.step({...args,now:200}),null);set(v(21));assert.equal(control.step({...args,now:230}).insertionT,.01);});
test('jump, patient context, connection and tracking failures disarm',()=>{const {control,set}=fixture();control.arm();set(v(10));control.step(args);set(v(100));assert.equal(control.step({...args,now:120}),null);assert.equal(control.snapshot().armed,false);for(const patch of [{reference:false},{connected:false},{visible:false}]){control.arm();assert.equal(control.step({...args,...patch}),null);assert.equal(control.snapshot().armed,false)}});
