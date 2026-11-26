(function mechanics(root){
'use strict';
const check=(ok,msg)=>{if(!ok)throw Error(msg)};
const sub=(a,b)=>a.map((x,i)=>x-b[i]),norm=a=>Math.hypot(...a),dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
const apply=(m,p)=>m.slice(0,3).map(r=>dot(r.slice(0,3),p)+r[3]);
function resample(points,count){
 check(Array.isArray(points)&&points.length>=2&&points.length<=12000&&points.every(p=>Array.isArray(p)&&p.length===3&&p.every(Number.isFinite)),'Invalid route points');
 const arcs=[0];for(let i=1;i<points.length;i++){const l=norm(sub(points[i],points[i-1]));check(l>1e-8,'Coincident route points');arcs.push(arcs[i-1]+l);}
 const length=arcs.at(-1);check(length>=.5&&length<=1000,'Route length outside solver budget');
 const n=count??Math.max(3,Math.min(80,Math.ceil(length/2)+1));check(Number.isInteger(n)&&n>=3&&n<=80,'Invalid rod discretization');
 let j=1;const out=Array.from({length:n},(_,i)=>{const s=length*i/(n-1);while(j<arcs.length-1&&arcs[j]<s)j++;const t=(s-arcs[j-1])/(arcs[j]-arcs[j-1]);return points[j-1].map((x,k)=>x+t*(points[j][k]-x));});
 return {points:out,length,spacing:length/(n-1)};
}
function maskContact(study,seg,limit){
 check(study?.schema==='pulmosimu.study/1'&&seg?.structure==='airway_lumen'&&seg.patch().edits.length===0,'Exact unchanged lumen required');
 check(seg.patch().ct_sha256===study.source.sha256,'Lumen belongs to another CT');
 const a=study.spatial.voxel_to_ras_mm,inv=study.spatial.ras_mm_to_voxel,d=study.source.dimensions,sp=study.spatial.spacing_mm;
 const axes=[0,1,2].map(j=>a.slice(0,3).map(r=>r[j]/sp[j]));
 for(let i=0;i<3;i++)for(let j=i+1;j<3;j++)check(Math.abs(dot(axes[i],axes[j]))<1e-5,'Sheared voxel frame unsupported');
 const radius=sp.map(s=>Math.ceil(limit/s+1));const work=radius.reduce((n,v)=>n*(2*v+1),1);check(work<=8000,'Contact grid too fine for local solver budget');
 const inside=p=>p.every((x,i)=>x>=0&&x<d[i]),occupied=p=>inside(p)&&seg.at(p)===1;
 return function(p){
  const q=apply(inv,p),c=q.map(Math.round),isInside=occupied(c);let best=limit,grad=[0,0,0],found=false;
  for(let x=-radius[0];x<=radius[0];x++)for(let y=-radius[1];y<=radius[1];y++)for(let z=-radius[2];z<=radius[2];z++){
   const v=[c[0]+x,c[1]+y,c[2]+z];
   const delta=q.map((u,j)=>(u-Math.max(v[j]-.5,Math.min(v[j]+.5,u)))*sp[j]);
   const l=norm(delta);if(l>=best||occupied(v)===isInside)continue;
   best=l;found=true;
   grad=l>1e-9?[0,1,2].map(i=>delta.reduce((s,x,j)=>s+x*axes[j][i],0)/l*(isInside?1:-1)):[0,0,0];
  }
  return {signed_mm:isInside?best:-best,gradient:grad,censored:!found,inside:isInside};
 };
}
function solve({route,study,seg,motion=null,movingStudy=null,movingSeg=null,phase=0,EI=5,stretch=2,wall=.5,anchor=.05,friction=.2,iterations=30}){
 check(route?.schema==='pulmosimu.airway-route/1'&&route.ct_sha256===study?.source?.sha256&&route.mask_sha256===seg?.digest&&seg.patch().edits.length===0,'Unchanged source-linked J route required');
 check([EI,stretch,wall,anchor,friction].every(Number.isFinite)&&EI>=0&&EI<=500&&stretch>0&&stretch<=100&&wall>0&&wall<=100&&anchor>=0&&anchor<=10&&friction>=0&&friction<=1,'Invalid assumed material parameters');
 check(Number.isInteger(iterations)&&iterations>=0&&iterations<=50&&[0,1].includes(phase),'Only source endpoints supported; invalid iteration budget');
 const base=resample(route.points_ras_mm),rest=base.spacing;
 let ref=base.points,activeStudy=study,activeSeg=seg;
 if(phase===1){
  check(motion&&movingStudy&&movingSeg,'Expiration field and exact expiration lumen required');
  const mapped=motion.routeMap({...route,points_ras_mm:base.points},1);
  ref=mapped.points.map(p=>p.position_ras_mm);activeStudy=movingStudy;activeSeg=movingSeg;
 }
 const radius=route.diameter_mm/2;check(Number.isFinite(radius)&&radius>=.5&&radius<=6,'Invalid instrument radius');
 const contact=maskContact(activeStudy,activeSeg,radius+1);
 let x=ref.map(p=>p.slice());
 function evaluate(points,gradient){
  const g=points.map(()=>[0,0,0]);let es=0,eb=0,ew=0,ea=0,maxOverlap=0,maxK=0,minGap=Infinity,unresolved=0,contactNodes=0,frictionIndex=0;
  const add=(i,v,s)=>{if(gradient)for(let k=0;k<3;k++)g[i][k]+=s*v[k];};
  for(let i=1;i<points.length;i++){
   const v=sub(points[i],points[i-1]),l=norm(v);if(l<1e-8)throw Error('Degenerate shaft segment');
   const e=l-rest;es+=.5*stretch*e*e;add(i,v,stretch*e/l);add(i-1,v,-stretch*e/l);
  }
  const kb=EI/(rest**3);
  for(let i=1;i<points.length-1;i++){
   const v=points[i-1].map((z,k)=>z-2*points[i][k]+points[i+1][k]);
   eb+=.5*kb*dot(v,v);add(i-1,v,kb);add(i,v,-2*kb);add(i+1,v,kb);
   const a=sub(points[i],points[i-1]),b=sub(points[i+1],points[i]),la=norm(a),lb=norm(b);
   const angle=Math.acos(Math.max(-1,Math.min(1,dot(a,b)/(la*lb))));maxK=Math.max(maxK,angle/((la+lb)/2));
  }
  for(let i=0;i<points.length;i++){
   const a=sub(points[i],ref[i]);ea+=.5*anchor*dot(a,a);add(i,a,anchor);
   const c=contact(points[i]),overlap=Math.max(0,radius-c.signed_mm);
   minGap=Math.min(minGap,c.signed_mm-radius);maxOverlap=Math.max(maxOverlap,overlap);
   if(overlap>0){contactNodes++;if(c.censored||norm(c.gradient)<1e-8)unresolved++;ew+=.5*wall*overlap**2;add(i,c.gradient,-wall*overlap);frictionIndex+=friction*wall*overlap;}
  }
  g[0]=[0,0,0];g[g.length-1]=[0,0,0];
  return {energy:es+eb+ew+ea,stretch_energy_Nmm:es,bending_energy_Nmm:eb,wall_penalty_Nmm:ew,anchor_energy_Nmm:ea,maximum_curvature_per_mm:maxK,maximum_nodal_overlap_mm:maxOverlap,minimum_nodal_gap_lower_bound_mm:minGap,contact_nodes:contactNodes,unresolved_contact_nodes:unresolved,assumed_coulomb_sum_N:frictionIndex,gradient:g};
 }
 let cur=evaluate(x,true),accepted=0,reason='ITERATION_LIMIT';
 const initial=cur.energy;
 for(let i=0;i<iterations;i++){
  const gmax=Math.max(...cur.gradient.map(norm));if(gmax<1e-6){reason='SMALL_GRADIENT';break;}
  let step=Math.min(.1,.25/gmax),ok=false;
  for(let trial=0;trial<8;trial++){
   const y=x.map((p,j)=>p.map((v,k)=>v-step*cur.gradient[j][k]));
   const next=evaluate(y,true);
   if(next.energy<cur.energy-1e-10){x=y;cur=next;accepted++;ok=true;break;}step*=.5;
  }
  if(!ok){reason='NO_DESCENT_STEP';break;}
 }
 const {gradient,...metrics}=cur;
 let shaftLength=0;for(let i=1;i<x.length;i++)shaftLength+=norm(sub(x[i],x[i-1]));
 // Nodal contact is not swept-segment verification. Report additional midpoint observations.
 let midOverlap=0;for(let i=1;i<x.length;i++){const p=x[i].map((v,k)=>(v+x[i-1][k])/2);midOverlap=Math.max(midOverlap,Math.max(0,radius-contact(p).signed_mm));}
 return {schema:'pulmosimu.shaft-research/1',ct_sha256:study.source.sha256,mask_sha256:seg.digest,active_ct_sha256:activeStudy.source.sha256,active_mask_sha256:activeSeg.digest,field_sha256:phase===1?motion.field_sha256:null,phase,route_root_node:route.root_node,route_target_node:route.target_node,assumptions:{EI_Nmm2:EI,stretch_N_per_mm:stretch,wall_penalty_N_per_mm:wall,anchor_N_per_mm:anchor,friction_coefficient:friction,diameter_mm:route.diameter_mm,rest_segment_mm:rest},initial_energy_Nmm:initial,accepted_iterations:accepted,termination:reason,shaft_length_mm:shaftLength,maximum_midpoint_overlap_mm:midOverlap,...metrics,points_ras_mm:x,contact_is_sampled:true,friction_is_diagnostic_only:true,tissue_deformation_solved:false,clinical_navigation_allowed:false,review:'PENDING'};
}
root.PulmoShaftMechanics=Object.freeze({solve,resample,maskContact});
})(typeof window!=='undefined'?window:globalThis);
