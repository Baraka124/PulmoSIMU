/* XR22: one source-derived implicit reference lumen.
 * Union of tapered swept centerline pieces, no new branches or patient geometry.
 * The zero surface removes internal overlap walls. Terminal ends are bounded
 * spherical closures, not inferred distal anatomy. Numerical view only.
 */
(function(root){
'use strict';
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const sub=(a,b)=>a.map((v,i)=>v-b[i]);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const unit=a=>{const n=Math.hypot(...a);return n>1e-10?a.map(v=>v/n):[0,1,0]};
function piece(a,b,r0,r1,t0,t1,id){
 const d=sub(b,a),dd=dot(d,d);
 if(dd<1e-12||!(r0>0&&r1>0)||![...a,...b,r0,r1].every(Number.isFinite))throw Error('INVALID_SOURCE_PIECE');
 const tangent=unit(d),normal=unit(cross(tangent,Math.abs(tangent[1])>.92?[1,0,0]:[0,1,0]));
 return {a,b,d,dd,r0,r1,t0,t1,id,normal,binormal:cross(tangent,normal)};
}
function sample(p,x,y,z){
 const dx=x-p.a[0],dy=y-p.a[1],dz=z-p.a[2];
 const t=clamp((dx*p.d[0]+dy*p.d[1]+dz*p.d[2])/p.dd,0,1);
 return Math.hypot(dx-t*p.d[0],dy-t*p.d[1],dz-t*p.d[2])-(p.r0+(p.r1-p.r0)*t);
}
function sourcePieces(segments,curves,{spacing=.026}={}){
 const pieces=[];
 for(const seg of segments){
  const curve=curves[seg.id];if(!curve)throw Error('MISSING_SOURCE_CURVE');
  const n=clamp(Math.ceil(curve.getLength()/spacing),4,96);
  const pts=curve.getPoints(n).map(p=>[p.x,p.y,p.z]);
  for(let i=0;i<n;i++)pieces.push(piece(pts[i],pts[i+1],
   seg.r0+(seg.r1-seg.r0)*i/n,seg.r0+(seg.r1-seg.r0)*(i+1)/n,i/n,(i+1)/n,seg.id));
 }
 return pieces;
}
function fieldAt(pieces,p){return pieces.reduce((v,s)=>Math.min(v,sample(s,...p)),Infinity)}
function build(pieces,{step=.011,maxGrid=4_500_000,maxTriangles=160000}={}){
 if(!Array.isArray(pieces)||!pieces.length||pieces.length>2048||!Number.isFinite(step)||step<.004||step>.05)throw Error('INVALID_BUILD_INPUT');
 const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];
 for(const p of pieces)for(let a=0;a<3;a++){const r=Math.max(p.r0,p.r1)+3*step;
  lo[a]=Math.min(lo[a],p.a[a]-r,p.b[a]-r);hi[a]=Math.max(hi[a],p.a[a]+r,p.b[a]+r)}
 let dims=hi.map((h,i)=>Math.ceil((h-lo[i])/step)+1);
 if(dims.reduce((a,b)=>a*b,1)>maxGrid)throw Error('LUMEN_GRID_BUDGET');
 const [nx,ny,nz]=dims,slice=nx*ny,N=slice*nz;
 const field=new Float32Array(N);field.fill(4*step);
 const owners=new Int16Array(N);owners.fill(-1);
 const idx=(x,y,z)=>x+y*nx+z*slice;
 for(let j=0;j<pieces.length;j++){
  const p=pieces[j],r=Math.max(p.r0,p.r1)+3*step;
  const low=p.a.map((a,i)=>clamp(Math.floor((Math.min(a,p.b[i])-r-lo[i])/step),0,dims[i]-1));
  const high=p.a.map((a,i)=>clamp(Math.ceil((Math.max(a,p.b[i])+r-lo[i])/step),0,dims[i]-1));
  for(let z=low[2];z<=high[2];z++)for(let y=low[1];y<=high[1];y++)for(let x=low[0];x<=high[0];x++){
   const k=idx(x,y,z),v=sample(p,lo[0]+x*step,lo[1]+y*step,lo[2]+z*step);
   if(v<field[k]){field[k]=v;owners[k]=j}
  }
 }
 const positions=[],normals=[],uvs=[];
 const cornerOffsets=[[0,0,0],[1,0,0],[1,1,0],[0,1,0],[0,0,1],[1,0,1],[1,1,1],[0,1,1]];
 const tetra=[[0,5,1,6],[0,1,2,6],[0,2,3,6],[0,3,7,6],[0,7,4,6],[0,4,5,6]];
 function grad(k){return [field[k+1]-field[k-1],field[k+nx]-field[k-nx],field[k+slice]-field[k-slice]]}
 function interpolate(a,b,ka,kb,xyz){
  const fa=field[ka],fb=field[kb],t=clamp(fa/(fa-fb),0,1);
  const aa=cornerOffsets[a],bb=cornerOffsets[b];
  const p=aa.map((v,i)=>lo[i]+(xyz[i]+v+(bb[i]-v)*t)*step);
  const ga=grad(ka),gb=grad(kb);
  return {p,n:unit(ga.map((v,i)=>v+(gb[i]-v)*t)),owner:fa<0?owners[ka]:owners[kb]};
 }
 function uv(p,s){
  const delta=sub(p,s.a),t=clamp(dot(delta,s.d)/s.dd,0,1);
  const rad=delta.map((v,i)=>v-t*s.d[i]);
  return [Math.atan2(dot(rad,s.binormal),dot(rad,s.normal))/(2*Math.PI)+.5,s.t0+(s.t1-s.t0)*t];
 }
 function emit(a,b,c){
  if(positions.length/9>=maxTriangles)throw Error('LUMEN_TRIANGLE_BUDGET');
  const area=cross(sub(b.p,a.p),sub(c.p,a.p));if(Math.hypot(...area)<1e-12)return;
  if(dot(area,a.n.map((v,i)=>v+b.n[i]+c.n[i]))<0)[b,c]=[c,b];
  const s=pieces[a.owner]||pieces[b.owner]||pieces[c.owner]||pieces[0];
  const verts=[a,b,c],tex=verts.map(v=>uv(v.p,s));
  if(Math.max(...tex.map(t=>t[0]))-Math.min(...tex.map(t=>t[0]))>.5)for(const t of tex)if(t[0]<.5)t[0]+=1;
  for(let i=0;i<3;i++){positions.push(...verts[i].p);normals.push(...verts[i].n);uvs.push(...tex[i]);}
 }
 for(let z=1;z<nz-2;z++)for(let y=1;y<ny-2;y++)for(let x=1;x<nx-2;x++){
  const k=idx(x,y,z);
  // Most of the volume is homogeneous exterior: avoid allocating a
  // corner array per empty cell on memory-constrained headset browsers.
  if(field[k]>=0&&field[k+1]>=0&&field[k+nx]>=0&&field[k+nx+1]>=0&&field[k+slice]>=0&&field[k+slice+1]>=0&&field[k+slice+nx]>=0&&field[k+slice+nx+1]>=0)continue;
  const ks=[k,k+1,k+1+nx,k+nx,k+slice,k+slice+1,k+slice+1+nx,k+slice+nx];
  let mask=0;for(let i=0;i<8;i++)if(field[ks[i]]<0)mask|=1<<i;
  if(mask===0||mask===255)continue;
  const xyz=[x,y,z],edge=(a,b)=>interpolate(a,b,ks[a],ks[b],xyz);
  for(const t of tetra){
   const ins=t.filter(i=>mask&(1<<i)),out=t.filter(i=>!(mask&(1<<i)));
   if(ins.length===1)emit(...out.map(o=>edge(ins[0],o)));
   else if(ins.length===3)emit(...ins.map(i=>edge(i,out[0])));
   else if(ins.length===2){
    const a=edge(ins[0],out[0]),b=edge(ins[0],out[1]),c=edge(ins[1],out[0]),d=edge(ins[1],out[1]);
    emit(a,b,c);emit(b,d,c);
   }
  }
 }
 if(!positions.length)throw Error('EMPTY_LUMEN_SURFACE');
 return Object.freeze({positions:new Float32Array(positions),normals:new Float32Array(normals),uvs:new Float32Array(uvs),triangles:positions.length/9,gridCells:N,step,pieceCount:pieces.length});
}
root.PulmoXRUnifiedLumen=Object.freeze({piece,sourcePieces,fieldAt,build});
})(typeof window!=='undefined'?window:globalThis);
