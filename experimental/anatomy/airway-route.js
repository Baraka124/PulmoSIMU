(function airway(root){
'use strict';
const check=(v,m)=>{if(!v)throw Error(m)};
const dist=(a,b)=>Math.hypot(...a.map((x,i)=>x-b[i]));
const hash=x=>typeof x==='string'&&/^[a-f0-9]{64}$/.test(x);
function open(study,seg,source){
 check(study?.schema==='pulmosimu.study/1'&&seg?.structure==='airway_lumen'&&seg.patch().edits.length===0,'Load unchanged airway mask');
 check(source?.schema==='pulmosimu.airway-skeleton/1'&&source.ct_sha256===study.source.sha256&&hash(seg.digest)&&source.mask_sha256===seg.digest&&source.structure==='airway_lumen'&&source.navigation_allowed===false,'Skeleton identity or status mismatch');
 const a=study.spatial.voxel_to_ras_mm,inv=study.spatial.ras_mm_to_voxel,d=study.source.dimensions,sp=study.spatial.spacing_mm;
 const col=j=>a.slice(0,3).map(r=>r[j]);
 for(let i=0;i<3;i++)for(let j=i+1;j<3;j++)check(Math.abs(col(i).reduce((s,x,k)=>s+x*col(j)[k],0))/(sp[i]*sp[j])<1e-5,'Sheared voxel frame not supported for clearance');
 const transform=(m,p)=>m.slice(0,3).map(r=>r[0]*p[0]+r[1]*p[1]+r[2]*p[2]+r[3]);
 const inside=p=>p.every((x,i)=>Number.isInteger(x)&&x>=0&&x<d[i]);
 const raw=source.nodes_ras_mm;
 check(Array.isArray(raw)&&raw.length>0&&raw.length<=100000,'Invalid skeleton size');
 const seen=new Set(),vox=raw.map(p=>{
  check(Array.isArray(p)&&p.length===3&&p.every(Number.isFinite),'Invalid RAS node');
  const q=transform(inv,p),v=q.map(Math.round);check(dist(q,v)<.001&&inside(v)&&seg.at(v)===1,'Skeleton node outside airway mask or voxel grid');
  const key=v.join(',');check(!seen.has(key),'Duplicate skeleton voxel');seen.add(key);return v;
 });
 const adj=vox.map(()=>[]),edges=source.edges_index_length_mm,edgeSeen=new Set();
 check(Array.isArray(edges)&&edges.length<=1300000,'Invalid edge count');
 let rejected=0;
 for(const e of edges){
  check(Array.isArray(e)&&e.length===3,'Malformed edge');
  const [i,j,l]=e;check(Number.isInteger(i)&&Number.isInteger(j)&&i>=0&&j>=0&&i<vox.length&&j<vox.length&&i!==j,'Invalid edge endpoints');
  const key=Math.min(i,j)+':'+Math.max(i,j);check(!edgeSeen.has(key),'Duplicate edge');edgeSeen.add(key);
  const delta=vox[j].map((v,k)=>v-vox[i][k]);check(delta.every(x=>Math.abs(x)<=1),'Nonlocal skeleton edge');
  const length=dist(raw[i],raw[j]);check(Number.isFinite(l)&&Math.abs(l-length)<.001&&length>0,'Invalid edge length');
  // Every corner of the voxel block crossed by diagonal edges must be foreground.
  let ok=true;for(let x=0;x<2;x++)for(let y=0;y<2;y++)for(let z=0;z<2;z++)if(seg.at([vox[i][0]+x*delta[0],vox[i][1]+y*delta[1],vox[i][2]+z*delta[2]])!==1)ok=false;
  if(!ok){rejected++;continue;}adj[i].push([j,length]);adj[j].push([i,length]);
 }
 function nearest(p,maxMm=5){
  check(inside(p),'Invalid target voxel');let best=-1,bestD=Infinity;
  const q=transform(a,p);for(let i=0;i<raw.length;i++){const v=dist(raw[i],q);if(v<bestD){bestD=v;best=i;}}
  check(bestD<=maxMm,'No skeleton node within '+maxMm+' mm');
  return {node:best,offset_mm:bestD};
 }
 const cache=new Map();
 function clearance(i,limit){
  const key=i+':'+limit;if(cache.has(key))return cache.get(key);
  const p=vox[i],radius=sp.map(s=>Math.ceil(limit/s+.5));let best=limit;
  check(radius.reduce((n,x)=>n*(2*x+1),1)<=100000,'Clearance sampling budget exceeded for this spacing');
  for(let x=-radius[0];x<=radius[0];x++)for(let y=-radius[1];y<=radius[1];y++)for(let z=-radius[2];z<=radius[2];z++){
   const delta=[x,y,z],dd=Math.hypot(...delta.map((v,k)=>Math.max(0,Math.abs(v)-.5)*sp[k]));if(dd>=best)continue;
   const q=p.map((v,k)=>v+delta[k]);if(!inside(q)||seg.at(q)===0)best=dd;
  }
  cache.set(key,best);return best;
 }
 function plan(start,end,diameter,margin=.5){
  check(seg.patch().edits.length===0,'Mask edited; rebuild skeleton');
  check([start,end].every(i=>Number.isInteger(i)&&i>=0&&i<raw.length)&&start!==end,'Choose distinct graph endpoints');
  check(Number.isFinite(diameter)&&diameter>=1&&diameter<=12&&Number.isFinite(margin)&&margin>=0&&margin<=2,'Invalid instrument dimensions');
  const needed=diameter/2+margin,limit=needed+Math.hypot(...sp)/2+.001;
  const costs=new Float64Array(raw.length).fill(Infinity),prev=new Int32Array(raw.length).fill(-1),heap=[];
  function push(i,v){let k=heap.length;heap.push([i,v]);while(k>0){const p=(k-1)>>1;if(heap[p][1]<=v)break;heap[k]=heap[p];k=p;heap[k]=[i,v];}}
  function pop(){const top=heap[0],last=heap.pop();if(heap.length){let k=0;while(2*k+1<heap.length){let c=2*k+1;if(c+1<heap.length&&heap[c+1][1]<heap[c][1])c++;if(heap[c][1]>=last[1])break;heap[k]=heap[c];k=c;}heap[k]=last;}return top;}
  let visits=0;const budget=12000;
  function bound(i,j,l){return Math.min(clearance(i,limit),clearance(j,limit))-l/2;}
  costs[start]=0;push(start,0);
  while(heap.length){const [i,v]=pop();if(v!==costs[i])continue;if(++visits>budget)throw Error('Route computation budget exceeded; use smaller ROI');if(i===end)break;
   for(const [j,l] of adj[i]){if(v+l>=costs[j])continue;if(bound(i,j,l)<needed)continue;costs[j]=v+l;prev[j]=i;push(j,v+l);}
  }
  check(Number.isFinite(costs[end]),'No route satisfies conservative mask clearance');
  const nodes=[];for(let i=end;i!==-1;i=prev[i])nodes.push(i);nodes.reverse();
  let low=Infinity;for(let k=1;k<nodes.length;k++)low=Math.min(low,bound(nodes[k-1],nodes[k],dist(raw[nodes[k-1]],raw[nodes[k]])));
  return {schema:'pulmosimu.airway-route/1',ct_sha256:study.source.sha256,mask_sha256:seg.digest,root_node:start,target_node:end,diameter_mm:diameter,margin_mm:margin,length_mm:costs[end],minimum_clearance_lower_bound_mm:low,nodes,points_ras_mm:nodes.map(i=>raw[i].slice()),points_voxel:nodes.map(i=>vox[i].slice()),review:'PENDING',clinical_navigation_allowed:false};
 }
 function pose(route,k){
  check(Number.isInteger(k)&&k>=0&&k<route.points_ras_mm.length,'Invalid route frame');const p=route.points_ras_mm[k],next=route.points_ras_mm[Math.min(k+1,route.points_ras_mm.length-1)],last=route.points_ras_mm[Math.max(k-1,0)],v=k===route.points_ras_mm.length-1?p.map((x,i)=>x-last[i]):next.map((x,i)=>x-p[i]),n=Math.hypot(...v);
  return {position_ras_mm:p.slice(),forward_ras:v.map(x=>x/n),voxel:route.points_voxel[k].slice()};
 }
 return Object.freeze({nearest,plan,pose,nodeCount:vox.length,rejectedEdges:rejected});
}
root.PulmoAirwayRoute=Object.freeze({open});
})(typeof window!=='undefined'?window:globalThis);
