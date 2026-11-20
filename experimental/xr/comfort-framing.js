/**
 * XR-04: deterministic seated comfort framing for reference anatomy.
 * Display-only units. No inference of actual human anatomy measurements.
 *
 * Fit a 3D object into a small frontal "tabletop" envelope at a measured
 * headset-relative distance rather than relying on a desktop camera origin.
 */
const v3=v=>Array.isArray(v)&&v.length===3&&v.every(Number.isFinite);
const norm=v=>{const n=Math.hypot(...v);return n>1e-6?v.map(x=>x/n):null};
const add=(a,b)=>a.map((x,i)=>x+b[i]);
const mul=(v,k)=>v.map(x=>x*k);
export const COMFORT_DEFAULTS=Object.freeze({
 distanceM:2.15,
 widthM:0.56,
 heightM:0.55,
 verticalOffsetM:-0.08,
 maxInitialScale:0.85,
 minUserScale:0.35,
 maxUserScale:1.9
});
export function fitSeatedAnatomy({
 head,forward,bounds,localCenter=[0,0,0],defaults=COMFORT_DEFAULTS
}){
 if(!v3(head)||!v3(forward)||!v3(bounds)||!v3(localCenter)||
  bounds.some(x=>x<=0)||!norm(forward))throw Error('Invalid XR reference framing inputs');
 const horizontal=norm([forward[0],0,forward[2]])||[0,0,-1];
 const scale=Math.min(
   defaults.maxInitialScale,
   defaults.widthM/bounds[0],
   defaults.heightM/bounds[1]
 );
 if(!(scale>0))throw Error('Invalid XR initial frame');
 const targetCenter=add(add(head,mul(horizontal,defaults.distanceM)),[0,defaults.verticalOffsetM,0]);
 // Rotation is applied by renderer to the centered anatomy group, so
 // its render-space local origin must be its visual bbox center.
 // The nonzero localCenter is provided for explicit inspection/validation.
 return Object.freeze({
  distanceM:defaults.distanceM,
  sizeM:bounds.map(x=>x*scale),
  scale,
  center:targetCenter,
  localCenter:[...localCenter],
  forward:horizontal
 });
}
export function isComfortFramed({head,center,size,limits=COMFORT_DEFAULTS}){
 if(!v3(head)||!v3(center)||!v3(size)||size.some(x=>x<=0))return false;
 const delta=center.map((x,i)=>x-head[i]);
 const distance=Math.hypot(...delta);
 return distance>=limits.distanceM*.7&&distance<=limits.distanceM*1.5&&
   size[0]<=limits.widthM*1.01&&size[1]<=limits.heightM*1.01;
}
