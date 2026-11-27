(function recovery(root){
'use strict';
function create(){
 let source=null,ready=false,tracked=false;
 return {reset(){source=null;ready=false;tracked=false;},step({id,pose,buttons=[],axes=[]}){
  if(!id||!pose){source=null;ready=false;tracked=false;return false;}
  if(id!==source){source=id;ready=false;}
  tracked=true;
  if(!ready){if(buttons.some(b=>b?.pressed)||axes.some(a=>!Number.isFinite(a)||Math.abs(a)>.15))return false;ready=true;return false;}
  return true;
 },snapshot:()=>({tracked,ready})};
}
function createTravel(){
 let ready=false;
 return {reset(){ready=false;},step({side,x,y,allowed}){
  if(!allowed||!['left','right'].includes(side)||!Number.isFinite(x)||!Number.isFinite(y)){ready=false;return null;}
  if(Math.max(Math.abs(x),Math.abs(y))<=.20){ready=true;return null;}
  if(!ready||Math.max(Math.abs(x),Math.abs(y))<.70)return null;
  ready=false;
  if(side==='right')return Math.abs(x)>=Math.abs(y)&&Math.abs(x)>=.70?(x<0?'turn-left':'turn-right'):null;
  return Math.abs(y)>=Math.abs(x)?(y<0?'forward':'back'):(x<0?'left':'right');
 },snapshot:()=>({ready})};
}
function floorY({floorGranted,headY}){if(!Number.isFinite(headY))throw Error('Invalid head');return floorGranted?0:headY-1.18;}
root.PulmoXRRecovery=Object.freeze({create,createTravel,floorY});
})(typeof window!=='undefined'?window:globalThis);
