/* PulmoSIMU XR-12: deterministic source-first spatial pointer arbitration.
 * Plain numbers and source hit metadata only. No anatomy, CT, network or
 * scene ownership. If the panel is closer, its full visible surface blocks
 * click-through even when the ray is not over a selectable button.
 */
(function(global){
'use strict';
function finiteDistance(x){return typeof x==='number'&&Number.isFinite(x)&&x>=0}
function choose({panel=null,anatomy=null,mode='anatomy',toleranceM=.002}={}){
 const pd=panel?.distance;
 const ad=anatomy?.distanceM;
 const panelValid=finiteDistance(pd);
 const anatomyValid=mode==='anatomy'&&anatomy?.hit===true&&finiteDistance(ad);
 if(panelValid&&(!anatomyValid||pd<=ad+Math.max(0,toleranceM))){
   return Object.freeze({kind:'panel',distanceM:pd,action:panel.action??null});
 }
 if(anatomyValid)return Object.freeze({kind:'anatomy',distanceM:ad,action:null});
 return Object.freeze({kind:'none',distanceM:null,action:null});
}
global.PulmoXRSpatialTarget=Object.freeze({choose});
})(typeof window!=='undefined'?window:globalThis);
