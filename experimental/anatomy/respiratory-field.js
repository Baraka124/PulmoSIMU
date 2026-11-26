(function respiratory(root){
'use strict';
const check=(ok,msg)=>{if(!ok)throw Error(msg)};
const hash=x=>typeof x==='string'&&/^[0-9a-f]{64}$/.test(x);
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
const det=m=>m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])-m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])+m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]);
const apply=(m,p)=>m.slice(0,3).map(r=>dot(r.slice(0,3),p)+r[3]);
const same=(a,b)=>Array.isArray(a)&&a.length===4&&a.every((r,i)=>Array.isArray(r)&&r.length===4&&r.every((x,j)=>Number.isFinite(x)&&Math.abs(x-b[i][j])<=.001));
function bind(fixed,moving,manifest,bytes,digest){
 check(fixed?.schema==='pulmosimu.study/1'&&moving?.schema==='pulmosimu.study/1','Two H-ready CT studies required');
 check(hash(fixed.source.sha256)&&hash(moving.source.sha256)&&fixed.source.sha256!==moving.source.sha256,'Distinct CT sources required');
 check(manifest?.schema==='pulmosimu.respiratory-field/1'&&manifest.fixed_ct_sha256===fixed.source.sha256&&manifest.moving_ct_sha256===moving.source.sha256,'Field source identity mismatch');
 check(manifest.fixed_phase==='inspiration'&&manifest.moving_phase==='expiration'&&manifest.direction==='fixed_RAS_to_moving_RAS_mm'&&manifest.encoding==='float32le-xyz-interleaved-x-fast','Unsupported field convention');
 check(hash(digest)&&manifest.field_sha256===digest,'Field checksum mismatch');
 check(same(manifest.fixed_voxel_to_ras_mm,fixed.spatial.voxel_to_ras_mm)&&same(manifest.moving_voxel_to_ras_mm,moving.spatial.voxel_to_ras_mm),'CT frame mismatch');
 const d=manifest.dimensions,s=manifest.stride,fd=fixed.source.dimensions;
 check(Number.isInteger(s)&&s>=1&&s<=16&&Array.isArray(d)&&d.length===3&&d.every((n,i)=>Number.isInteger(n)&&n>=2&&n===Math.floor((fd[i]-1)/s)+1),'Invalid sampled field grid');
 const count=d.reduce((n,x)=>n*x,1);check(count<=10000000&&bytes.byteLength===12*count,'Field size mismatch or budget exceeded');
 const data=new DataView(bytes);
 for(let i=0;i<count*3;i++){const v=data.getFloat32(4*i,true);check(Number.isFinite(v)&&Math.abs(v)<=200,'Nonfinite or out-of-budget displacement');}
 const val=(i,j,k,c)=>data.getFloat32(4*(3*(i+d[0]*(j+d[1]*k))+c),true);
 function sample(p){
  check(Array.isArray(p)&&p.length===3&&p.every(Number.isFinite),'Invalid reference position');
  const q=apply(fixed.spatial.ras_mm_to_voxel,p).map(x=>x/s);
  check(q.every((x,i)=>x>=-1e-6&&x<=d[i]-1+1e-6),'Outside sampled field support');
  const t=q.map((x,i)=>Math.max(0,Math.min(d[i]-1,x))),lo=t.map((x,i)=>Math.min(d[i]-2,Math.floor(x))),w=t.map((x,i)=>x-lo[i]);
  const u=[0,0,0],du=Array.from({length:3},()=>[0,0,0]);
  for(let x=0;x<2;x++)for(let y=0;y<2;y++)for(let z=0;z<2;z++){
   const v=[x,y,z],weights=v.map((n,i)=>n?w[i]:1-w[i]),weight=weights[0]*weights[1]*weights[2];
   for(let c=0;c<3;c++){const value=val(lo[0]+x,lo[1]+y,lo[2]+z,c);u[c]+=weight*value;for(let j=0;j<3;j++)du[c][j]+=value*(v[j]?1:-1)*weights[(j+1)%3]*weights[(j+2)%3];}
  }
  return {u,du};
 }
 function map(p,phase=1){
  check(Number.isFinite(phase)&&phase>=0&&phase<=1,'Phase fraction out of range');
  const {u,du}=sample(p),inv=fixed.spatial.ras_mm_to_voxel;
  const jac=du.map((r,i)=>[0,1,2].map(j=>(i===j?1:0)+phase*r.reduce((sum,x,k)=>sum+x*inv[k][j]/s,0)));
  const jacobian=det(jac);check(Number.isFinite(jacobian)&&jacobian>0,'Nonpositive local Jacobian; mapping rejected');
  const mapped=p.map((x,i)=>x+phase*u[i]),movingVoxel=apply(moving.spatial.ras_mm_to_voxel,mapped);
  const inMoving=movingVoxel.every((x,i)=>x>=0&&x<=moving.source.dimensions[i]-1);
  return {position_ras_mm:mapped,displacement_mm:u.map(x=>phase*x),jacobian,moving_voxel:movingVoxel,in_moving_volume:inMoving};
 }
 function routeMap(route,phase){
  check(route?.ct_sha256===fixed.source.sha256&&route.review==='PENDING'&&Array.isArray(route.points_ras_mm)&&route.points_ras_mm.length>=2&&route.points_ras_mm.length<=12000,'Route is not linked to fixed CT');
  const points=route.points_ras_mm.map(p=>map(p,phase));check(points.every(p=>p.in_moving_volume),'Mapped route leaves moving volume');
  let length=0;for(let i=1;i<points.length;i++)length+=Math.hypot(...points[i].position_ras_mm.map((x,k)=>x-points[i-1].position_ras_mm[k]));
  return {length_mm:length,minimum_sampled_jacobian:Math.min(...points.map(p=>p.jacobian)),points,clearance_revalidated:false,clinical_navigation_allowed:false};
 }
 return Object.freeze({map,routeMap,field_sha256:digest});
}
root.PulmoRespiratoryField=Object.freeze({bind});
})(typeof window!=='undefined'?window:globalThis);
