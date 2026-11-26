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
// Angular bounding envelopes in a deliberately chosen viewer-relative basis.
// Conservative rectangular projection, not a stereo visibility or collision solver.
function envelope(points){
 if(!Array.isArray(points)||!points.length||!points.every(p=>Array.isArray(p)&&p.length===3&&p.every(finite)&&p[2]>.08))return null;
 const a=points.map(p=>[Math.atan2(p[0],p[2]),Math.atan2(p[1],Math.hypot(p[0],p[2]))]);
 return {left:Math.min(...a.map(p=>p[0])),right:Math.max(...a.map(p=>p[0])),bottom:Math.min(...a.map(p=>p[1])),top:Math.max(...a.map(p=>p[1]))};
}
function overlap(a,b){if(!a||!b)return 0;return Math.max(0,Math.min(a.right,b.right)-Math.max(a.left,b.left))*Math.max(0,Math.min(a.top,b.top)-Math.max(a.bottom,b.bottom));}
function panelEnvelope(spec,width,height,lowerExtension=0){
 if(!spec||![spec.depth,spec.right,spec.y,spec.scale,width,height,lowerExtension].every(finite)||Math.min(spec.depth,spec.scale,width,height)<=0)return null;
 const center=[spec.right,spec.y,spec.depth],d=Math.hypot(...center),h=Math.hypot(center[0],center[2]);if(d<.1||h<.1)return null;
 const right=[center[2]/h,0,-center[0]/h],up=[-center[0]*center[1]/(h*d),h/d,-center[2]*center[1]/(h*d)];
 const pts=[];for(const x of [-width/2,width/2])for(const y of [-height/2-lowerExtension,height/2])pts.push(center.map((v,i)=>v+spec.scale*(x*right[i]+y*up[i])));
 return envelope(pts);
}
function compose({items,protectedEnvelope=null,fixed=[]}){
 const placed=fixed.filter(Boolean).slice(),poses={},reports={};
 for(const item of items){const {id,preferred,width,height,lowerExtension=0}=item;
  const candidates=[preferred,...(item.alternatives||[])].filter(Boolean);
  let best=null;
  for(let i=0;i<candidates.length;i++){const p=candidates[i],e=panelEnvelope(p,width,height,lowerExtension);if(!e)continue;
   const area=Math.max(.000001,(e.right-e.left)*(e.top-e.bottom));
   const blocked=overlap(e,protectedEnvelope)/area;
   const shared=placed.reduce((s,v)=>s+overlap(e,v)/area,0);
   const eccentric=Math.max(0,Math.abs((e.left+e.right)/2)-.95)+Math.max(0,-.95-e.bottom)+Math.max(0,e.top-.75);
   const score=blocked*12+shared*8+eccentric*3+i*.035;
   if(!best||score<best.score)best={p,e,score,blocked,shared};
  }
  if(best){poses[id]=best.p;placed.push(best.e);reports[id]={estimatedAnatomyOverlapFraction:best.blocked,estimatedPanelOverlapFraction:best.shared,angularEnvelope:best.e};}
 }
 return {poses,reports,method:'BOUNDED_ANGULAR_ENVELOPE_HEURISTIC',physicalOcclusionVerified:false};
}
root.PulmoXRSpatialExperience=Object.freeze({angularMinutes,readability,layoutRail,pulse,envelope,overlap,panelEnvelope,compose});
})(typeof window!=='undefined'?window:globalThis);
