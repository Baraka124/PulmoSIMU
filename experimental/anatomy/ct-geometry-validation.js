/**
 * CT geometry validation gates. These check that the selected GLB and
 * centreline metadata plausibly occupy the same display coordinate system.
 * They do NOT verify segmentations, anatomical accuracy, orientation or labels.
 * Reject ambiguous or far-away geometry instead of silently mixing frames.
 */
const valid=n=>typeof n==='number'&&Number.isFinite(n);
const point=p=>Array.isArray(p)&&p.length===3&&p.every(valid);
export function validateCtSceneAlignment(graph,bounds,{tolerance=0.75}={}){
 if(!graph?.nodes?.length||!bounds||!point(bounds.min)||!point(bounds.max))throw Error('CT geometry bounds unavailable');
 if(!valid(tolerance)||tolerance<0||tolerance>2)throw Error('Invalid alignment tolerance');
 const min=bounds.min,max=bounds.max;
 if(min.some((x,i)=>!(max[i]>x)))throw Error('Degenerate CT mesh bounds');
 const extent=min.map((x,i)=>max[i]-x),span=Math.hypot(...extent),margin=tolerance*span;
 const pts=graph.nodes.flatMap(n=>n.points);
 if(!pts.length)throw Error('No airway points');
 const outside=pts.filter(p=>p.some((v,i)=>v<min[i]-margin||v>max[i]+margin));
 if(outside.length)throw Error('CT centerlines and GLB do not share the same scene frame');
 const centroid=axis=>pts.reduce((s,p)=>s+p[axis],0)/pts.length;
 const axisGap=Math.hypot(...[0,1,2].map(i=>centroid(i)-(min[i]+max[i])/2));
 if(axisGap>span*(0.5+tolerance))throw Error('Centerline and mesh centres disagree');
 return Object.freeze({aligned:true,centerlinePoints:pts.length,meshDiagonalSceneUnits:span,
  kind:'basic spatial frame plausibility only',anatomicallyValidated:false});
}
export function ctBoundingBoxFromPoints(vertices){
 if(!Array.isArray(vertices)||vertices.length<3||vertices.some(p=>!point(p)))throw Error('Mesh must contain finite vertices');
 const min=[Infinity,Infinity,Infinity],max=[-Infinity,-Infinity,-Infinity];
 for(const p of vertices)for(let i=0;i<3;i++){min[i]=Math.min(min[i],p[i]);max[i]=Math.max(max[i],p[i])}
 return {min,max};
}
