(function rod(root){
'use strict';
const check=(v,m)=>{if(!v)throw Error(m)},finite=Number.isFinite,sub=(a,b)=>a.map((x,i)=>x-b[i]),dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0),norm=a=>Math.hypot(...a),mul=(a,s)=>a.map(x=>x*s),add=(a,b)=>a.map((x,i)=>x+b[i]),clip=(v,a,b)=>Math.max(a,Math.min(b,v)),range=(v,a,b)=>finite(v)&&v>=a&&v<=b;
function solve({route,study,seg,EI=5,GJ=3,EA=30,wall=.5,mu=.2,anchor=.02,advanceMm=0,rollDeg=0,iterations=12}){
 check(route?.schema==='pulmosimu.airway-route/1'&&route.ct_sha256===study?.source?.sha256&&route.mask_sha256===seg?.digest&&seg?.structure==='airway_lumen'&&seg.patch().edits.length===0&&route.clinical_navigation_allowed!==true,'Exact unchanged J airway route required');
 check(range(EI,0,500)&&range(GJ,.01,500)&&range(EA,.01,500)&&range(wall,.01,50)&&range(mu,0,1)&&range(anchor,0,1)&&range(advanceMm,-2,2)&&range(rollDeg,-45,45)&&Number.isInteger(iterations)&&iterations>=1&&iterations<=16,'Invalid assumed mechanics or command parameters');
 const L=root.PulmoShaftMechanics,base=L.resample(route.points_ras_mm),radius=route.diameter_mm/2;
 check(range(radius,.5,6)&&base.length>=4,'Route too short or invalid instrument');
 const n=Math.max(4,Math.min(24,Math.ceil(base.length/3)+1)),ref=L.resample(route.points_ras_mm,n).points,rest=ref.slice(1).map((p,i)=>norm(sub(p,ref[i])));
 check(rest.every(v=>v>.1),'Degenerate resampled shaft');
 const sp=study.spatial.spacing_mm,lim=radius+1,work=sp.reduce((p,v)=>p*(2*Math.ceil(lim/v+1)+1),1),contact=L.maskContact(study,seg,lim);
 let samples=0,voxelBudget=0;
 function sample(p){samples++;voxelBudget+=work;check(voxelBudget<=16000000,'Local contact work budget exceeded; reduce route or instrument diameter');const c=contact(p);if(!c.inside&&c.censored)throw Error('Outside sampled lumen support');return c;}
 let x=ref.map(p=>p.slice()),theta=Array(n).fill(0);
 const direction=mul(sub(ref[1],ref[0]),1/rest[0]);
 let previous=null;
 function evaluate(p,t,frictionState){
  const g=p.map(()=>[0,0,0]),h=t.map(()=>0),energies={stretch:0,bending:0,torsion:0,contact:0,anchor:0,incrementalFriction:0};
  const push=(i,v,k=1)=>{g[i]=add(g[i],mul(v,k));};
  const e=p.slice(1).map((q,i)=>{const d=sub(q,p[i]),l=norm(d);check(l>.05,'Collapsed segment');return {d,l,u:mul(d,1/l)};});
  let maxCurv=0,maxOverlap=0,maxMid=0,unresolved=0,normalSum=0;const contacts=[];
  for(let i=0;i<n-1;i++){
   const v=e[i],strain=v.l-rest[i],k=EA/rest[i],tw=t[i+1]-t[i],kt=GJ/rest[i];
   energies.stretch+=.5*k*strain*strain;push(i,v.u,-k*strain);push(i+1,v.u,k*strain);
   energies.torsion+=.5*kt*tw*tw;h[i]-=kt*tw;h[i+1]+=kt*tw;
  }
  // Finite-angle tangent energy, invariant under rigid-body translation and rotation.
  for(let i=1;i<n-1;i++){
   const a=e[i-1],b=e[i],d=clip(dot(a.u,b.u),-1,1),dual=(rest[i-1]+rest[i])/2,k=EI/dual;
   energies.bending+=k*(1-d);
   const ga=mul(sub(b.u,mul(a.u,d)),-k/a.l),gb=mul(sub(a.u,mul(b.u,d)),-k/b.l);
   push(i-1,ga,-1);push(i,sub(ga,gb));push(i+1,gb);
   maxCurv=Math.max(maxCurv,Math.acos(d)/((a.l+b.l)/2));
  }
  function wallAt(q,i,j=null){
   const c=sample(q),overlap=Math.max(0,radius-c.signed_mm),weight=j===null?.5:1,load=wall*weight*overlap;
   energies.contact+=.5*wall*weight*overlap*overlap;normalSum+=load;
   if(overlap>0){
    if(c.censored||norm(c.gradient)<1e-8)unresolved++;
    if(j===null)push(i,c.gradient,-load);else{push(i,c.gradient,-load*.5);push(j,c.gradient,-load*.5);}
   }
   if(j===null){maxOverlap=Math.max(maxOverlap,overlap);contacts.push({normal:c.gradient.slice(),load,overlap});}
   else maxMid=Math.max(maxMid,overlap);
  }
  for(let i=0;i<n;i++){
   const delta=sub(p[i],ref[i]);energies.anchor+=.5*anchor*dot(delta,delta);push(i,delta,anchor);wallAt(p[i],i);
  }
  for(let i=1;i<n;i++)wallAt(mul(add(p[i-1],p[i]),.5),i-1,i);
  // Incremental regularized Coulomb potential with previous nodal normal load frozen.
  // The scalar twist is an isotropic material spin, not a full director-frame rod.
  if(frictionState&&mu>0)for(let i=0;i<n;i++){
   const c=frictionState.contacts[i],m=c.normal;if(c.load<=0||norm(m)<.99)continue;
   const d=sub(p[i],frictionState.p[i]),tangent=sub(d,mul(m,dot(d,m))),spin=radius*(t[i]-frictionState.t[i]),eps=.05;
   const s=Math.sqrt(dot(tangent,tangent)+spin*spin+eps*eps),k=mu*c.load;
   energies.incrementalFriction+=k*(s-eps);push(i,tangent,k/s);h[i]+=k*spin*radius/s;
  }
  const energy=Object.values(energies).reduce((s,v)=>s+v,0);
  check(finite(energy)&&g.every(v=>v.every(finite))&&h.every(finite),'Nonfinite rod solution');
  return {energy,energies,g,h,maxCurv,maxOverlap,maxMid,unresolved,normalSum,contacts};
 }
 function relax(p,t,prev){
  let cur=evaluate(p,t,prev),initial=cur.energy,accepted=0,termination='ITERATION_LIMIT';
  for(let i=0;i<iterations;i++){
   // Proximal two nodes are position-clamped; only first twist is prescribed.
   const gp=cur.g.map((v,j)=>j<2?[0,0,0]:v),ht=cur.h.map((v,j)=>j===0?0:v),maxP=Math.max(...gp.map(norm)),maxT=Math.max(...ht.map(Math.abs));
   if(maxP<1e-5&&maxT<1e-5){termination='SMALL_GRADIENT';break;}
   let alpha=Math.min(.12,.18/Math.max(maxP,1e-12),.08/Math.max(maxT,1e-12)),ok=false;
   for(let k=0;k<6;k++){
    const q=p.map((v,j)=>sub(v,mul(gp[j],alpha))),s=t.map((v,j)=>v-alpha*ht[j]);
    let next;try{next=evaluate(q,s,prev);}catch(e){if(String(e.message).includes('budget'))throw e;alpha*=.5;continue;}
    if(next.energy<cur.energy-1e-10){p=q;t=s;cur=next;accepted++;ok=true;break;}alpha*=.5;
   }
   if(!ok){termination='NO_DESCENT_STEP';break;}
  }
  return {p,t,cur,initial,accepted,termination};
 }
 const baseResult=relax(x,theta,null);
 previous={p:baseResult.p.map(v=>v.slice()),t:baseResult.t.slice(),contacts:baseResult.cur.contacts};
 x=previous.p.map(v=>v.slice());theta=previous.t.slice();
 for(let i=0;i<2;i++)x[i]=add(x[i],mul(direction,advanceMm));
 theta[0]=rollDeg*Math.PI/180;
 const r=relax(x,theta,previous),c=r.cur,delta=sub(r.p.at(-1),previous.p.at(-1));
 const reaction=add(c.g[0],c.g[1]);
 const residual=Math.max(...c.g.slice(2).map(norm));
 let length=0;for(let i=1;i<n;i++)length+=norm(sub(r.p[i],r.p[i-1]));
 return {schema:'pulmosimu.nonlinear-shaft/1',ct_sha256:study.source.sha256,mask_sha256:seg.digest,route_root_node:route.root_node,route_target_node:route.target_node,
  assumptions:{EI_Nmm2:EI,GJ_Nmm2:GJ,EA_N:EA,wall_penalty_N_per_mm:wall,friction_coefficient:mu,anchor_N_per_mm:anchor,diameter_mm:radius*2,rest_lengths_mm:rest},
  command:{proximal_translation_mm:advanceMm,proximal_roll_deg:rollDeg},
  initial_energy_Nmm:r.initial,final_energy_Nmm:c.energy,energy_terms_Nmm:c.energies,baseline_energy_Nmm:baseResult.cur.energy,
  baseline_termination:baseResult.termination,baseline_iterations:baseResult.accepted,termination:r.termination,accepted_iterations:r.accepted,
  maximum_free_node_residual_N:residual,maximum_free_twist_residual_Nmm:Math.max(...c.h.slice(1).map(Math.abs)),shaft_length_mm:length,
  maximum_curvature_per_mm:c.maxCurv,maximum_nodal_overlap_mm:c.maxOverlap,maximum_midpoint_overlap_mm:c.maxMid,unresolved_contact_samples:c.unresolved,
  assumed_sampled_normal_load_sum_N:c.normalSum,assumed_proximal_reaction_N:reaction,assumed_proximal_torque_Nmm:c.h[0],
  distal_displacement_mm:delta,distal_displacement_magnitude_mm:norm(delta),distal_spin_deg:r.t.at(-1)*180/Math.PI,
  baseline_points_ras_mm:previous.p,points_ras_mm:r.p,twist_rad:r.t,contact_samples:samples,voxel_work_upper_bound:voxelBudget,
  model:'DISCRETE_ISOTROPIC_ROD_INCREMENTAL_RIGID_MASK',contact_is_sampled:true,friction_load_lagged:true,tissue_deformation_solved:false,material_feed_solved:false,full_cosserat_directors_solved:false,physical_force_measured:false,clinical_navigation_allowed:false,review:'PENDING'};
}
root.PulmoNonlinearShaft=Object.freeze({solve});
})(typeof window!=='undefined'?window:globalThis);
