/**
 * PulmoSIMU Spatial Intent v1 — source-agnostic interaction decisions.
 * Distances here are display-world metres in WebXR, NEVER anatomy source units.
 * Operations are reversible transforms, not biomechanical deformation.
 */
const finite3=v=>Array.isArray(v)&&v.length===3&&v.every(Number.isFinite);
const norm=v=>{const n=Math.hypot(...v);return n>1e-6?v.map(x=>x/n):null};
export const SPATIAL_LIMITS=Object.freeze({
  nearSurfaceM:.19,
  touchSurfaceM:.065,
  nearExitM:.24,
  summonDistanceM:.88,
  summonWidthM:.38,
  summonHeightM:.42,
  summonDurationS:.46,
  pinchStartM:.026,
  pinchReleaseM:.055
});
export function decideInteraction({mode='anatomy',panelHit=false,surfaceDistanceM=Infinity,rayHit=false}={}){
 if(panelHit)return 'panel';
 if(mode!=='anatomy')return 'locked';
 if(Number.isFinite(surfaceDistanceM)&&surfaceDistanceM>=0&&surfaceDistanceM<=SPATIAL_LIMITS.nearSurfaceM)return 'near';
 return rayHit?'far':'none';
}
export function pinchActive(previous,distanceM){
 if(!Number.isFinite(distanceM)||distanceM<0)return false;
 return previous?distanceM<SPATIAL_LIMITS.pinchReleaseM:distanceM<=SPATIAL_LIMITS.pinchStartM;
}
export function workZoneTarget({head,forward,verticalOffsetM=-.10,distanceM=SPATIAL_LIMITS.summonDistanceM}){
 if(!finite3(head)||!finite3(forward)||!(distanceM>.5&&distanceM<=1.5))throw Error('Invalid XR workzone inputs');
 const f=norm([forward[0],0,forward[2]]);
 if(!f)throw Error('Invalid XR forward vector');
 return [head[0]+f[0]*distanceM,head[1]+verticalOffsetM,head[2]+f[2]*distanceM];
}
export function workZoneScale(bounds,currentScale,{maxWidth=SPATIAL_LIMITS.summonWidthM,maxHeight=SPATIAL_LIMITS.summonHeightM}={}){
 if(!finite3(bounds)||bounds.some(x=>x<=0)||!Number.isFinite(currentScale)||currentScale<=0)
  throw Error('Invalid XR source geometry');
 if(!(maxWidth>0&&maxHeight>0))throw Error('Invalid display target');
 return Math.min(currentScale,maxWidth/bounds[0],maxHeight/bounds[1]);
}
export function easedTransform(start,target,t){
 if(!start||!target||!finite3(start.position)||!finite3(target.position)||
   !Number.isFinite(start.scale)||!Number.isFinite(target.scale)||!Number.isFinite(t))
   throw Error('Invalid XR transition');
 const u=Math.min(1,Math.max(0,t)),e=1-Math.pow(1-u,3);
 return {position:start.position.map((v,i)=>v+(target.position[i]-v)*e),
   scale:start.scale+(target.scale-start.scale)*e,finished:u>=1};
}
/** Immutable aggregate measurement; not anatomy/physics accuracy. */
export function frameMetrics(samples,dt,maxSamples=90){
 const v=Array.isArray(samples)?samples.filter(x=>Number.isFinite(x)&&x>0&&x<1):[];
 if(Number.isFinite(dt)&&dt>0&&dt<1)v.push(dt);
 const tail=v.slice(-maxSamples);
 if(!tail.length)return {samples:[],fps:null,p95Ms:null,slowFrames:0};
 const sorted=[...tail].sort((a,b)=>a-b);
 return {samples:tail,fps:Math.round(1/(tail.reduce((a,b)=>a+b,0)/tail.length)),
   p95Ms:Math.round(sorted[Math.min(sorted.length-1,Math.ceil(sorted.length*.95)-1)]*1e4)/10,
   slowFrames:tail.filter(x=>x>1/72).length};
}
