import test from 'node:test';
import assert from 'node:assert/strict';
import {readWorkspaceLaunch,workspaceLaunch,safeReferenceRegion,safeWorkspace,WORKSPACES} from '../shared/workspace-route.js';
import {createFocus,readFocus} from '../shared/reference-focus.js';
test('Every workspace uses a canonical single Studio route and trusted focus origin',()=>{
 assert.deepEqual(Object.keys(WORKSPACES),['procedure','mechanisms','xr']);
 assert.deepEqual(Object.values(WORKSPACES).map(w=>w.origin),['procedural','mechanistic','xr']);
 assert.equal(WORKSPACES.xr.path,'./index.html?xr=procedural&release=XR09-UNIFIED');
 const msg=createFocus({regionId:'LLL',origin:'xr',revision:4});
 assert.deepEqual(readFocus(msg),msg);
});
test('Quest handoff carries only one canonical lobe, not physiologic or patient state',()=>{
 const path=workspaceLaunch('../studio.html',{mode:'procedure',regionId:'LUL'});
 assert.deepEqual(readWorkspaceLaunch(path.slice(path.indexOf('?'))),{mode:'procedure',regionId:'LUL'});
 assert.deepEqual(readWorkspaceLaunch('?mode=xr&focus=RUL'),{mode:'xr',regionId:'RUL'});
 assert.deepEqual(readWorkspaceLaunch('?mode=patient&focus=personal-data'),{mode:'procedure',regionId:null});
 assert.deepEqual(readWorkspaceLaunch('?mode=mechanisms&focus=RML&patient=SECRET'),{mode:'mechanisms',regionId:'RML'});
 for(const region of ['CT-RUL','liver','RUL%20','random'])assert.equal(safeReferenceRegion(region),null);
 assert.equal(safeWorkspace('__proto__'),null);
 assert.throws(()=>workspaceLaunch('../studio.html',{mode:'procedure',regionId:'CT-001'}));
 assert.throws(()=>workspaceLaunch('../studio.html',{mode:'diagnosis',regionId:'RUL'}));
});
