/* PulmoSIMU XR-11 · World-space interaction limits.
 * Pure numeric helpers, independent of anatomy, CT, THREE and the headset.
 * Normal comfort view is bounded in real-world metres, never absolute model units.
 */
(function(global){
'use strict';
const CONFIG=Object.freeze({minimumSpanM:.22,maximumSpanM:.92,
  minimumViewerDistanceM:.47,maximumViewerDistanceM:2.30,
  minimumTwoHandSeparationM:.18,maximumTwoHandRatio:2.25});
const finite=(v)=>typeof v==='number'&&Number.isFinite(v);
const clamp=(v,low,high)=>Math.max(low,Math.min(high,v));
function scaleRange(maxExtent){
 if(!finite(maxExtent)||maxExtent<=.0001)throw Error('Missing original model extent');
 return Object.freeze({min:CONFIG.minimumSpanM/maxExtent,
  max:CONFIG.maximumSpanM/maxExtent});
}
function clampScale(desired,maxExtent){
 if(!finite(desired)||desired<=0)throw Error('Invalid proposed scale');
 const bounds=scaleRange(maxExtent);
 return clamp(desired,bounds.min,bounds.max);
}
function scaleFromHands(originalScale,startSeparation,currentSeparation,maxExtent){
 if(![originalScale,startSeparation,currentSeparation].every(finite)||
   originalScale<=0||startSeparation<CONFIG.minimumTwoHandSeparationM||
   currentSeparation<CONFIG.minimumTwoHandSeparationM)
   return Object.freeze({valid:false,scale:clampScale(originalScale,maxExtent)});
 const ratio=clamp(currentSeparation/startSeparation,
   1/CONFIG.maximumTwoHandRatio,CONFIG.maximumTwoHandRatio);
 return Object.freeze({valid:true,scale:clampScale(originalScale*ratio,maxExtent)});
}
function clampFocus(focus,viewer,maxDistance=CONFIG.maximumViewerDistanceM){
 if(!Array.isArray(focus)||!Array.isArray(viewer)||focus.length!==3||viewer.length!==3||
    ![...focus,...viewer,maxDistance].every(finite)||maxDistance<=CONFIG.minimumViewerDistanceM)
   throw Error('Invalid scene focus/viewer coordinates');
 const delta=focus.map((x,i)=>x-viewer[i]);
 const distance=Math.hypot(...delta);
 const bounded=clamp(distance,CONFIG.minimumViewerDistanceM,maxDistance);
 const direction=distance>1e-5?delta.map(x=>x/distance):[0,0,-1];
 return Object.freeze({clamped:Math.abs(bounded-distance)>1e-5,
    center:viewer.map((v,i)=>v+direction[i]*bounded),distanceM:bounded});
}
global.PulmoXRInteractionMath=Object.freeze({CONFIG,scaleRange,clampScale,scaleFromHands,clampFocus});
})(typeof window!=='undefined'?window:globalThis);
