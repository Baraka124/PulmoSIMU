import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
const source=readFileSync(new URL('../xr/native-quad.js',import.meta.url),'utf8');
const root={};runInNewContext(source,{window:root});
test('native quad stays inactive without projection-layer capability',()=>{
 const q=root.PulmoXRNativeQuad.create();
 q.bind({xr:{getSession:()=>({enabledFeatures:[],renderState:{}})},getContext:()=>({}),capabilities:{isWebGL2:false}});
 assert.equal(q.update({visible:true}),false);assert.equal(q.snapshot().compositorQuadActive,false);
});
test('projection layer is retained and allocation does not claim native pixels',()=>{
 const projection={},quad={destroy(){}};const session={enabledFeatures:['layers'],renderState:{layers:[projection]},updateRenderState(s){this.renderState=s}};
 const q=root.PulmoXRNativeQuad.create({transformConstructor:class{}});
 const renderer={xr:{getSession:()=>session,getBinding:()=>({createQuadLayer:()=>quad,getSubImage:()=>null}),getBaseLayer:()=>projection},getContext:()=>({RGBA8:1}),capabilities:{isWebGL2:true}};
 q.bind(renderer);const args={renderer,frame:{},target:{},space:{},position:{},orientation:{},visible:true};
 assert.equal(q.update(args),false);assert.equal(session.renderState.layers[0],projection);assert.equal(session.renderState.layers[1],quad);
 assert.equal(q.update(args),false);assert.equal(session.renderState.layers.length,1);assert.equal(q.snapshot().compositorQuadActive,false);
});
