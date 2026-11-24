/* PulmoSIMU XR-15 · Reversible transforms in a single research scene.
 * No original mesh vertices, source coordinate systems, patient CT or clinical
 * geometry are modified by view transforms.
 */
(function(global){
'use strict';
function valid(p){return p&&Array.isArray(p.position)&&p.position.length===3&&
  Array.isArray(p.quaternion)&&p.quaternion.length===4&&
  Number.isFinite(p.scale)&&p.scale>0&&
  [...p.position,...p.quaternion].every(Number.isFinite)}
function copy(p){if(!valid(p))throw Error('Invalid source-view pose');return Object.freeze({
  position:Object.freeze([...p.position]),quaternion:Object.freeze([...p.quaternion]),scale:p.scale})}
function same(a,b){if(!a||!b)return false;
 const d=a.position.reduce((n,x,i)=>n+(x-b.position[i])**2,0);
 const q=Math.abs(a.quaternion.reduce((n,x,i)=>n+x*b.quaternion[i],0));
 return d<1e-7&&1-q<.00003&&Math.abs(a.scale-b.scale)<.00001}
function create(limit=24){
 const undo=[],redo=[];let initial=null;
 function begin(p){if(initial)return false;initial=copy(p);return true}
 function finish(p){
  if(!initial)return false;const before=initial;initial=null;
  if(same(before,p))return false;
  undo.push(before);if(undo.length>limit)undo.shift();redo.length=0;
  return true;
 }
 function undoMove(p){initial=null;if(!undo.length)return null;redo.push(copy(p));return copy(undo.pop())}
 function redoMove(p){initial=null;if(!redo.length)return null;undo.push(copy(p));return copy(redo.pop())}
 function clear(){undo.length=redo.length=0;initial=null}
 function status(){return Object.freeze({undo:undo.length,redo:redo.length,inProgress:Boolean(initial)})}
 return Object.freeze({begin,finish,undo:undoMove,redo:redoMove,clear,status})
}
global.PulmoXRViewHistory=Object.freeze({create,valid});
})(typeof window!=='undefined'?window:globalThis);
