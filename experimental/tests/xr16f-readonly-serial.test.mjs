import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
const s=readFileSync(new URL('../xr/instrument-serial.js',import.meta.url),'utf8');
const ctx={window:{}};runInNewContext(s,ctx);
const API=ctx.window.PulmoXRReadOnlySerial;
test('XR16F serial decoder respects frame and overflow bounds',()=>{
 const d=API.decoder({maxLine:64});
 assert.equal(d.push('{"seq":1}').length,0);
 assert.deepEqual(Array.from(d.push('\n{"seq":2}\r\n')),[ '{"seq":1}', '{"seq":2}' ]);
 assert.equal(d.push('x'.repeat(900)+'\n').length,0);
 assert.equal(d.peek().buffered,0);
 assert.deepEqual(Array.from(d.push('good\n')),['good']);
});
test('XR16F serial pairing is opt in and unsupported browsers reject safely',async()=>{
 const s=API.create({navigatorObject:{}});
 assert.equal(s.status().state,'DISCONNECTED');
 await assert.rejects(()=>s.connect(),/WEB_SERIAL_UNSUPPORTED/);
 assert.equal(s.status().deviceToScopeCoupled,false);
 assert.equal(s.status().readOnly,true);
});
test('XR16F physical acquisition is never a simulator actuator or network client',()=>{
 const page=readFileSync(new URL('../validation-lab.html',import.meta.url),'utf8');
 assert.ok(page.includes('id="connectSerial"'));
 assert.ok(page.includes('id="disconnectSerial"'));
 assert.ok(page.includes('PulmoXRReadOnlySerial?.create'));
 for(const keyword of ['.write(', 'getWriter(', 'fetch(', 'WebSocket(', 'scope.velocity', 'fireAction('])
  assert.equal(s.includes(keyword),false,'no command/output path: '+keyword);
 assert.ok(s.includes('await navigatorObject.serial.requestPort()'));
});
