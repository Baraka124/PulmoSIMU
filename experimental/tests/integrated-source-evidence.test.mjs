// Numerical fixture only: no patient geometry or clinical acceptance.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {runInNewContext} from 'node:vm';
const U={};for(const path of ["anatomy/patient-study.js","anatomy/segmentation-review.js","anatomy/airway-route.js","anatomy/shaft-mechanics.js","anatomy/source-optics.js","anatomy/research-intelligence.js","xr/research-snapshot.js"])runInNewContext(readFileSync(new URL('../'+path,import.meta.url),'utf8'),{window:U,Uint8Array});
function fixture(U){
 const affine=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],dims=[15,15,15],ctSha='a'.repeat(64),maskSha='b'.repeat(64),routeSha='c'.repeat(64);
 const study=U.PulmoStudy.build({volume:{affine,dims,spacing:[1,1,1],spatialUnit:2,slope:1,intercept:0,datatype:4,sformCode:1},ctSha});
 const bytes=new Uint8Array(15**3);for(let z=1;z<14;z++)for(let y=1;y<14;y++)for(let x=1;x<14;x++)bytes[x+15*(y+15*z)]=1;
 const manifest={schema:'pulmosimu.mask/1',ct_sha256:ctSha,mask_sha256:maskSha,dimensions:dims,voxel_to_ras_mm:affine,structure:'airway_lumen',encoding:'uint8-x-fast-binary'};
 const seg=U.PulmoSegmentation.bind(study,manifest,bytes,maskSha);
 const source={schema:'pulmosimu.airway-skeleton/1',ct_sha256:ctSha,mask_sha256:maskSha,structure:'airway_lumen',navigation_allowed:false,nodes_ras_mm:Array.from({length:7},(_,i)=>[4+i,7,7]),edges_index_length_mm:Array.from({length:6},(_,i)=>[i,i+1,1])};
 const graph=U.PulmoAirwayRoute.open(study,seg,source),route=graph.plan(0,6,2,.5);
 return {study,seg,route,graph,routeSha};
}
function integrated(U,assert){
 const {study,seg,route,routeSha}=fixture(U);
 assert.equal(route.length_mm,6);
 const shaft=U.PulmoShaftMechanics.solve({route,study,seg});
 assert.ok(shaft.maximum_nodal_overlap_mm===0&&shaft.energy<1e-8);
 const frame=U.PulmoOptics.render({study,seg,route,index:3});
 assert.ok(frame.counts.WALL>0);
 const evidence=U.PulmoResearchIntelligence.assess({study,seg,route,routeSha,shaft,optics:frame});
 assert.equal(evidence.findings.find(x=>x.id==='L_MECHANICS').state,'SAMPLED_ONLY');
 assert.equal(evidence.findings.find(x=>x.id==='M_OPTICS').state,'ASSUMED_APPEARANCE');
 const packet=U.PulmoXRResearchSnapshot.build(route,routeSha,frame,shaft);
 assert.equal(packet.route_sha256,evidence.route_sha256);assert.equal(packet.frame.index,3);assert.equal(packet.mechanics.overlap_mm,0);
 const received=U.PulmoXRResearchSnapshot.sanitize(JSON.parse(JSON.stringify(packet)));
 assert.equal(received.clinical_navigation_allowed,false);assert.equal(evidence.clinical_navigation_allowed,false);assert.equal(received.frame.rgba.length,76800);
 assert.equal(received.route.points_ras_mm,undefined);assert.equal(received.ct_sha256,study.source.sha256);
}
function invalidation(U,assert){
 const {study,seg,route,graph,routeSha}=fixture(U);
 seg.set([7,7,7],0);assert.throws(()=>graph.plan(0,6,2,.5));assert.throws(()=>U.PulmoOptics.render({study,seg,route}));assert.throws(()=>U.PulmoResearchIntelligence.assess({study,seg,route,routeSha}));
}
test('integrated H/I/J/L/M/N/O source evidence to read-only XR snapshot',()=>integrated(U,assert));
test('mask edit invalidates route, optics and research assessment',()=>invalidation(U,assert));
