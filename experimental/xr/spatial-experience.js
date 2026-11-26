/* UI04: deliberate contextual controls and geometric visual-angle evidence. */
(function(root){'use strict';
const finite=Number.isFinite;
function angularMinutes(heightM,distanceM){return finite(heightM)&&finite(distanceM)&&heightM>0&&distanceM>0?2*Math.atan(heightM/(2*distanceM))*180/Math.PI*60:null;}
function readability({heightM,canvasHeight,fontPx,scale=1,distanceM,incidence=1}){
 if(![heightM,canvasHeight,fontPx,scale,distanceM,incidence].every(finite)||Math.min(heightM,canvasHeight,fontPx,scale,distanceM)<=0||incidence<0||incidence>1)return null;
 const nominal=heightM*fontPx/canvasHeight*scale*.72;
 const arcmin=angularMinutes(nominal*Math.max(.01,incidence),distanceM);
 return {estimatedGlyphArcmin:arcmin,distanceM,incidence,designTargetArcmin:24,state:arcmin>=24?'TARGET_GEOMETRY_MET':'ENLARGE_OR_MOVE_CLOSER',physicalReadabilityVerified:false};
}
function layoutRail(mode){
 const items=mode==='procedure'?[['dock-open','Controls',.24],['dock-mode','Anatomy',.23],['dock-channel','Channel',.22],['dock-guide','Guide',.17],['dock-exit','Exit',.14]]:[['dock-open','Controls',.28],['dock-mode','Bronchoscopy',.34],['dock-recenter','Recenter',.23],['dock-exit','Exit',.15]];
 let u=.012;return items.map(([id,label,w])=>{const a={id,label,u,v:.10,w:w*.976-.014,h:.80};u+=w*.976;return a;});
}
function pulse(elapsed){if(!finite(elapsed)||elapsed<0||elapsed>170)return 0;return -.008*Math.sin(Math.PI*elapsed/170);}
root.PulmoXRSpatialExperience=Object.freeze({angularMinutes,readability,layoutRail,pulse});
})(typeof window!=='undefined'?window:globalThis);
