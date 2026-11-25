/* XR28: bounded discrete elastic centerline relaxation in millimetres.
 * Fixed endpoints, approximate stretch and bend energies, compliant-wall
 * penalty from original reference lumen. Not a full Cosserat rod, measured
 * stiffness, torsion, force prediction or physical actuator command.
 */
(function(root){
'use strict';
const add=(a,b)=>a.map((v,i)=>v+b[i]),sub=(a,b)=>a.map((v,i)=>v-b[i]),mul=(a,s)=>a.map(v=>v*s),norm=a=>Math.hypot(...a),sq=a=>a.reduce((s,v)=>s+v*v,0);
const valid=p=>Array.isArray(p)&&p.length===3&&p.every(Number.isFinite);
const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
function solve({points,field,radiusMm,iterations=8,maxNodes=36,stretch=5,bend=.65,wall=10,anchor=.025}={}){
 if(!Array.isArray(points)||points.length<3||!points.every(valid)||typeof field!=='function'||!Number.isFinite(radiusMm)||radiusMm<=0||!Number.isInteger(iterations)||iterations<1||iterations>12||!Number.isInteger(maxNodes)||maxNodes<3||maxNodes>48||![stretch,bend,wall,anchor].every(v=>Number.isFinite(v)&&v>=0&&v<=50))throw Error('INVALID_ROD_INPUT');
 const n=Math.min(maxNodes,points.length),ref=Array.from({length:n},(_,i)=>points[Math.round(i*(points.length-1)/(n-1))].slice());
 const rest=ref.slice(1).map((p,i)=>Math.max(.05,norm(sub(p,ref[i]))));
 const margin=p=>-field(p)-radiusMm;
 function energy(p){
  let e=0,pen=0,min=Infinity;
  for(let i=0;i<n;i++){
   const m=margin(p[i]);if(!Number.isFinite(m))throw Error('INVALID_FIELD');
   const d=Math.max(0,-m);min=Math.min(min,m);pen+=d*d;
   if(i>0)e+=stretch*(norm(sub(p[i],p[i-1]))-rest[i-1])**2;
   if(i>0&&i<n-1)e+=bend*sq(add(sub(p[i-1],mul(p[i],2)),p[i+1]))+anchor*sq(sub(p[i],ref[i]));
   if(i>0){const mid=mul(add(p[i],p[i-1]),.5),m2=margin(mid);if(!Number.isFinite(m2))throw Error('INVALID_FIELD');min=Math.min(min,m2);pen+=Math.max(0,-m2)**2;}
  }return {total:e+wall*pen,penalty:pen,minMarginMm:min};
 }
 function gradient(p){
  const g=p.map(()=>[0,0,0]);
  for(let i=1;i<n;i++){
   const d=sub(p[i],p[i-1]),len=norm(d);
   if(len>1e-8){const v=mul(d,2*stretch*(len-rest[i-1])/len);g[i]=add(g[i],v);g[i-1]=sub(g[i-1],v);}
  }
  for(let i=1;i<n-1;i++){
   const b=add(sub(p[i-1],mul(p[i],2)),p[i+1]);
   g[i-1]=add(g[i-1],mul(b,2*bend));g[i]=add(g[i],mul(b,-4*bend));g[i+1]=add(g[i+1],mul(b,2*bend));
   g[i]=add(g[i],mul(sub(p[i],ref[i]),2*anchor));
  }
  function wallGrad(q){const m=margin(q);if(m>=0)return [0,0,0];const eps=.10;
   return q.map((_,j)=>{const a=q.slice(),b=q.slice();a[j]+=eps;b[j]-=eps;return -2*wall*m*(field(a)-field(b))/(2*eps);});
  }
  // dE/dp = 2*wall*(field+radius)*grad(field).
  for(let i=1;i<n-1;i++)g[i]=add(g[i],wallGrad(p[i]));
  for(let i=1;i<n;i++){const w=mul(wallGrad(mul(add(p[i],p[i-1]),.5)),.5);g[i]=add(g[i],w);g[i-1]=add(g[i-1],w);}
  g[0]=[0,0,0];g[n-1]=[0,0,0];return g;
 }
 let p=ref.map(a=>a.slice()),start=energy(p),curr=start,accepted=0;
 for(let k=0;k<iterations;k++){
  const g=gradient(p);if(!g.every(valid))break;
  let moved=false;
  for(let step=.035;step>=.001;step*=.5){
   const q=p.map((v,i)=>{const d=mul(g[i],-step),len=norm(d);return add(v,len>.65?mul(d,.65/len):d);});
   const e=energy(q);
   if(e.total<=curr.total+1e-9&&e.minMarginMm>=Math.min(curr.minMarginMm,0)-.05){p=q;curr=e;accepted++;moved=true;break;}
  }if(!moved)break;
 }
 const length=p.slice(1).reduce((a,v,i)=>a+norm(sub(v,p[i])),0),refLength=rest.reduce((a,b)=>a+b,0);
 return {points:p,initialEnergy:start.total,finalEnergy:curr.total,minMarginMm:curr.minMarginMm,wallPenalty:curr.penalty,iterationsAccepted:accepted,lengthMm:length,referenceLengthMm:refLength,maxDisplacementMm:Math.max(...p.map((v,i)=>norm(sub(v,ref[i])))),state:curr.minMarginMm<0?'REFERENCE_CONTACT_PROXY':'REFERENCE_CLEAR',coefficientStatus:'ASSUMED_NUMERICAL_WEIGHTS',physicalForceValidated:false,tissueDeformationSolved:false};
}
root.PulmoXRElasticRod=Object.freeze({solve});
})(typeof window!=='undefined'?window:globalThis);
