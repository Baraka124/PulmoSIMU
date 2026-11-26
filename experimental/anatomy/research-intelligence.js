(function intelligence(root){
'use strict';
const check=(x,m)=>{if(!x)throw Error(m)};
const hash=x=>typeof x==='string'&&/^[a-f0-9]{64}$/.test(x);
const finite=Number.isFinite;
function descriptor(rgba,w,h){
 check(Number.isInteger(w)&&Number.isInteger(h)&&w>=16&&h>=16&&w*h<=32768&&rgba?.length===w*h*4,'Invalid image dimensions');
 const lum=new Float32Array(w*h),grad=new Float32Array(w*h).fill(NaN);
 for(let i=0;i<lum.length;i++)lum[i]=.2126*rgba[4*i]+.7152*rgba[4*i+1]+.0722*rgba[4*i+2];
 for(let y=1;y<h-1;y++)for(let x=1;x<w-1;x++){
  const i=x+y*w;if([i,i-1,i+1,i-w,i+w].some(j=>rgba[j*4+3]<250))continue;
  grad[i]=Math.hypot(lum[i+1]-lum[i-1],lum[i+w]-lum[i-w]);
 }return grad;
}
function score(a,b){
 check(a.length===b.length,'Descriptor size mismatch');let n=0,sa=0,sb=0,saa=0,sbb=0,sab=0;
 for(let i=0;i<a.length;i++){const x=a[i],y=b[i];if(!finite(x)||!finite(y))continue;n++;sa+=x;sb+=y;saa+=x*x;sbb+=y*y;sab+=x*y;}
 if(n<100||n/a.length<.35)return null;
 const va=saa-sa*sa/n,vb=sbb-sb*sb/n;if(va/n<1||vb/n<1)return null;
 return {correlation:Math.max(-1,Math.min(1,(sab-sa*sb/n)/Math.sqrt(va*vb))),pixels:n,support_fraction:n/a.length};
}
function candidates(length,index,radius=8){
 check(Number.isInteger(length)&&length>=2&&length<=12000&&Number.isInteger(index)&&index>=0&&index<length&&Number.isInteger(radius)&&radius>=1&&radius<=100,'Invalid search bounds');
 const lo=Math.max(0,index-radius),hi=Math.min(length-1,index+radius),ids=new Set([lo,index,hi]);
 for(let i=0;i<9;i++)ids.add(Math.round(lo+(hi-lo)*i/8));
 return [...ids].sort((a,b)=>a-b).map(index=>[-30,0,30].map(roll_offset=>({index,roll_offset}))).flat();
}
function rank(rows){
 const r=rows.filter(x=>finite(x.correlation)&&x.support_fraction>=.35).sort((a,b)=>b.correlation-a.correlation||a.index-b.index||a.roll-b.roll);
 if(!r.length)return {status:'NO_COMPARABLE_CANDIDATE',candidates:[],score_gap:null,localization_verified:false};
 const top=r[0],different=r.find(x=>x.index!==top.index),gap=different?top.correlation-different.correlation:null;
 // Heuristic review flags, never a probability or acceptance gate.
 const status=top.correlation<.25?'WEAK_SIMILARITY':gap===null?'INSUFFICIENT_SPATIAL_ALTERNATIVES':gap<.08?'AMBIGUOUS_CANDIDATES':'RANKED_CANDIDATES_UNVERIFIED';
 return {status,candidates:r.slice(0,8),score_gap:gap,compared_candidates:r.length,localization_verified:false};
}
function assess({study,seg,route,shaft=null,optics=null,localization=null,routeSha}){
 check(study?.schema==='pulmosimu.study/1'&&seg?.structure==='airway_lumen'&&seg.patch().edits.length===0&&route?.schema==='pulmosimu.airway-route/1'&&route.ct_sha256===study.source.sha256&&route.mask_sha256===seg.digest&&hash(routeSha),'Exact unchanged study, mask and route required');
 const p=route.points_ras_mm;check(Array.isArray(p)&&p.length>=2&&p.length<=12000,'Invalid route');
 let length=0,maxAngle=0;const vectors=[];
 for(let i=1;i<p.length;i++){const v=p[i].map((x,j)=>x-p[i-1][j]),l=Math.hypot(...v);check(finite(l)&&l>0,'Invalid route segment');length+=l;vectors.push(v.map(x=>x/l));}
 for(let i=1;i<vectors.length;i++)maxAngle=Math.max(maxAngle,Math.acos(Math.max(-1,Math.min(1,vectors[i].reduce((s,x,j)=>s+x*vectors[i-1][j],0))))*180/Math.PI);
 const required=route.diameter_mm/2+route.margin_mm,margin=route.minimum_clearance_lower_bound_mm-required;check(finite(margin)&&finite(length),'Invalid clearance evidence');
 const findings=[{id:'J_GEOMETRY',state:margin<0?'CONFLICT':'CANDIDATE_ONLY',basis:'J route sampled conservative clearance bound',remaining_radial_margin_mm:margin},{id:'J_TURN',state:'OBSERVED',basis:'Polyline turning angle, not instrument bending limit',maximum_turn_deg:maxAngle}];
 const linked=shaft&&shaft.ct_sha256===route.ct_sha256&&shaft.mask_sha256===route.mask_sha256&&shaft.route_root_node===route.root_node&&shaft.route_target_node===route.target_node&&shaft.assumptions?.diameter_mm===route.diameter_mm;
 if(linked) findings.push({id:'L_MECHANICS',state:shaft.maximum_nodal_overlap_mm>0||shaft.maximum_midpoint_overlap_mm>0?'OVERLAP_OBSERVED':'SAMPLED_ONLY',basis:'Assumed model; no continuous collision or measured forces',phase:shaft.phase,active_mask_sha256:shaft.active_mask_sha256,field_sha256:shaft.field_sha256,maximum_nodal_overlap_mm:shaft.maximum_nodal_overlap_mm,maximum_midpoint_overlap_mm:shaft.maximum_midpoint_overlap_mm,termination:shaft.termination,assumptions:shaft.assumptions});
 else findings.push({id:'L_MECHANICS',state:'MISSING',basis:'No current route mechanics evidence'});
 if(optics&&optics.ct_sha256===route.ct_sha256&&optics.mask_sha256===route.mask_sha256&&optics.root_node===route.root_node&&optics.target_node===route.target_node) findings.push({id:'M_OPTICS',state:'ASSUMED_APPEARANCE',basis:'Source mask boundary rendering',index:optics.index,fov:optics.fov,counts:{...optics.counts}});
 else findings.push({id:'M_OPTICS',state:'MISSING',basis:'No current optical frame'});
 findings.push({id:'N_LOCALIZATION',state:localization?.route_sha256===routeSha?localization.status:'MISSING',basis:'Bounded image-gradient similarity only; not registered position'});
 return {schema:'pulmosimu.research-evidence/1',ct_sha256:route.ct_sha256,mask_sha256:route.mask_sha256,route_sha256:routeSha,root_node:route.root_node,target_node:route.target_node,route_length_mm:length,findings,localization:localization?.route_sha256===routeSha?localization:null,review:'PENDING',clinical_navigation_allowed:false};
}
root.PulmoResearchIntelligence=Object.freeze({descriptor,score,candidates,rank,assess});
})(typeof window!=='undefined'?window:globalThis);
