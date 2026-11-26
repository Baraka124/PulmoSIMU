(function handoff(root){
'use strict';
const check=(x,m)=>{if(!x)throw Error(m)},hash=x=>typeof x==='string'&&/^[a-f0-9]{64}$/.test(x),num=(x,min,max)=>Number.isFinite(x)&&x>=min&&x<=max;
function sanitize(packet){
 check(packet?.schema==='pulmosimu.xr-research-snapshot/1'&&packet.review==='PENDING'&&packet.clinical_navigation_allowed===false,'Invalid research snapshot contract');
 check([packet.ct_sha256,packet.mask_sha256,packet.route_sha256].every(hash),'Invalid source identifiers');
 const r=packet.route;check(r&&num(r.length_mm,.001,1000)&&num(r.diameter_mm,1,12)&&num(r.margin_mm,0,2)&&num(r.clearance_lower_bound_mm,-100,100)&&Number.isInteger(r.node_count)&&r.node_count>=2&&r.node_count<=12000,'Invalid route metrics');
 let frame=null;
 if(packet.frame!==null){
  const f=packet.frame;check(f&&f.width===160&&f.height===120&&Number.isInteger(f.index)&&f.index>=0&&f.index<r.node_count&&num(f.fov,60,120)&&Array.isArray(f.rgba)&&f.rgba.length===76800&&f.rgba.every(v=>Number.isInteger(v)&&v>=0&&v<=255),'Invalid optical snapshot');
  frame={width:f.width,height:f.height,index:f.index,fov:f.fov,rgba:f.rgba.slice()};
 }
 let mechanics=null;
 if(packet.mechanics!==null){const m=packet.mechanics;check(m&&[0,1].includes(m.phase)&&num(m.overlap_mm,0,1000)&&num(m.curvature_per_mm,0,10000),'Invalid mechanics evidence');mechanics={phase:m.phase,overlap_mm:m.overlap_mm,curvature_per_mm:m.curvature_per_mm};}
 return {schema:packet.schema,ct_sha256:packet.ct_sha256,mask_sha256:packet.mask_sha256,route_sha256:packet.route_sha256,route:{length_mm:r.length_mm,diameter_mm:r.diameter_mm,margin_mm:r.margin_mm,clearance_lower_bound_mm:r.clearance_lower_bound_mm,node_count:r.node_count},frame,mechanics,review:'PENDING',clinical_navigation_allowed:false};
}
function build(route,routeSha,frame=null,shaft=null){
 check(route?.schema==='pulmosimu.airway-route/1'&&route.clinical_navigation_allowed===false,'Current research route required');
 const match=x=>x&&x.ct_sha256===route.ct_sha256&&x.mask_sha256===route.mask_sha256;
 const f=match(frame)&&frame.root_node===route.root_node&&frame.target_node===route.target_node?{width:frame.width,height:frame.height,index:frame.index,fov:frame.fov,rgba:Array.from(frame.rgba)}:null;
 const m=match(shaft)&&shaft.route_root_node===route.root_node&&shaft.route_target_node===route.target_node?{phase:shaft.phase,overlap_mm:Math.max(shaft.maximum_nodal_overlap_mm,shaft.maximum_midpoint_overlap_mm),curvature_per_mm:shaft.maximum_curvature_per_mm}:null;
 return sanitize({schema:'pulmosimu.xr-research-snapshot/1',ct_sha256:route.ct_sha256,mask_sha256:route.mask_sha256,route_sha256:routeSha,route:{length_mm:route.length_mm,diameter_mm:route.diameter_mm,margin_mm:route.margin_mm,clearance_lower_bound_mm:route.minimum_clearance_lower_bound_mm,node_count:route.nodes.length},frame:f,mechanics:m,review:'PENDING',clinical_navigation_allowed:false});
}
root.PulmoXRResearchSnapshot=Object.freeze({build,sanitize});
})(typeof window!=='undefined'?window:globalThis);
