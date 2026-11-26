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
function floorY({floorGranted,headY}){if(!Number.isFinite(headY))throw Error('Invalid head');return floorGranted?0:headY-1.18;}
root.PulmoXRRecovery=Object.freeze({create,floorY});
})(typeof window!=='undefined'?window:globalThis);
