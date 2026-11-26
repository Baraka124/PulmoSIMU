(function spatial(root){
'use strict';
const finite=Number.isFinite,v3=v=>Array.isArray(v)&&v.length===3&&v.every(finite),add=(a,b)=>a.map((x,i)=>x+b[i]),sub=(a,b)=>a.map((x,i)=>x-b[i]),mul=(a,k)=>a.map(x=>x*k),dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0),norm=a=>Math.hypot(...a),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],unit=a=>{const n=norm(a);if(n<1e-6)throw Error('DEGENERATE_AXIS');return mul(a,1/n);};
const require=(x,m)=>{if(!x)throw Error(m)};
const dimensions=Object.freeze({xM:.20,zM:.15});
function fitJig(points){
 require(Array.isArray(points)&&points.length===4&&points.every(v3),'FOUR_LANDMARKS_REQUIRED');
 const [o,a,b,c]=points,xRaw=sub(a,o),zRaw=sub(b,o),x=unit(xRaw),z0=unit(zRaw);
 require(Math.abs(norm(xRaw)-dimensions.xM)<=.008&&Math.abs(norm(zRaw)-dimensions.zM)<=.008,'JIG_LENGTH_MISMATCH');
 require(Math.abs(dot(x,z0))<=.06,'JIG_NOT_ORTHOGONAL');
 const y=unit(cross(z0,x)),z=unit(cross(x,y));require(y[1]>=.85,'JIG_MUST_BE_HORIZONTAL_UP');
 const predicted=add(o,add(mul(x,dimensions.xM),mul(z,dimensions.zM))),residual=norm(sub(c,predicted));
 require(residual<=.012,'INDEPENDENT_LANDMARK_RESIDUAL_EXCEEDED');
 return {origin:o.slice(),basis:[x,y,z],residualM:residual,measuredXM:norm(xRaw),measuredZM:norm(zRaw),operatorVerified:false,clinicalRegistration:false};
}
function rotation(q){require(Array.isArray(q)&&q.length===4&&q.every(finite)&&Math.abs(Math.hypot(...q)-1)<.02,'INVALID_ROTATION');const [x,y,z,w]=q.map(v=>v/Math.hypot(...q));return [[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]];}
function point(pose,offset){require(v3(pose?.position)&&v3(offset),'INVALID_TRACKED_POSE');return add(pose.position,rotation(pose.orientation).map(r=>dot(r,offset)));}
function solve(A,b){const n=b.length,M=A.map((r,i)=>[...r,b[i]]);let min=Infinity,max=0;for(let i=0;i<n;i++){let k=i;for(let j=i+1;j<n;j++)if(Math.abs(M[j][i])>Math.abs(M[k][i]))k=j;[M[i],M[k]]=[M[k],M[i]];const d=M[i][i];require(Math.abs(d)>1e-6,'INSUFFICIENT_POSE_DIVERSITY');min=Math.min(min,Math.abs(d));max=Math.max(max,Math.abs(d));for(let j=i;j<=n;j++)M[i][j]/=d;for(let k=0;k<n;k++){if(k===i)continue;const s=M[k][i];for(let j=i;j<=n;j++)M[k][j]-=s*M[i][j];}}require(min/max>.0001,'ILL_CONDITIONED_PIVOT');return M.map(r=>r[n]);}
function fitPivot(poses){
 require(Array.isArray(poses)&&poses.length===5&&poses.every(p=>v3(p.position)),'FIVE_POSES_REQUIRED');
 const rows=[],values=[];for(const p of poses.slice(0,4)){const R=rotation(p.orientation);for(let i=0;i<3;i++){rows.push([...R[i],...Array.from({length:3},(_,j)=>i===j?-1:0)]);values.push(-p.position[i]);}}
 const A=Array.from({length:6},(_,i)=>Array.from({length:6},(_,j)=>rows.reduce((s,r)=>s+r[i]*r[j],0))),b=Array.from({length:6},(_,i)=>rows.reduce((s,r,j)=>s+r[i]*values[j],0)),solution=solve(A,b),offset=solution.slice(0,3),pivot=solution.slice(3);
 require(norm(offset)>=.015&&norm(offset)<=.35,'OFFSET_OUTSIDE_PASSIVE_POINTER_RANGE');
 const errors=poses.slice(0,4).map(p=>norm(sub(point(p,offset),pivot))),rms=Math.sqrt(errors.reduce((s,x)=>s+x*x,0)/4),holdout=norm(sub(point(poses[4],offset),pivot));
 require(rms<=.006&&holdout<=.012,'PIVOT_RESIDUAL_EXCEEDED');
 return {offsetM:offset,fitRmsM:rms,holdoutM:holdout,fitPoses:4,holdoutPoses:1,physicalDeviceCertified:false};
}
function create(){
 let kind=null,points=[],poses=[],jig=null,pivot=null,state='IDLE',armed=false;
 function reset(){kind=null;points=[];poses=[];jig=null;pivot=null;armed=false;state='IDLE';}
 function begin(next){require(['pivot','jig'].includes(next),'UNKNOWN_CALIBRATION');if(next==='jig')require(pivot,'CALIBRATE_POINTER_FIRST');kind=next;points=[];poses=[];jig=null;if(next==='pivot')pivot=null;armed=false;state='READY_FOR_CAPTURE';}
 function arm(){if(!kind||state==='REJECTED'||state==='COMPLETE')return false;armed=true;state='AWAITING_RIGHT_TRIGGER';return true;}
 function capture(pose){if(!armed)return false;armed=false;try{if(kind==='pivot'){rotation(pose.orientation);require(v3(pose.position),'NO_TRACKED_POSITION');poses.push({position:pose.position.slice(),orientation:pose.orientation.slice()});if(poses.length===5){pivot=fitPivot(poses);state='COMPLETE';kind=null;}else state='READY_FOR_CAPTURE';}else{points.push(point(pose,pivot.offsetM));if(points.length===4){jig=fitJig(points);kind=null;state='COMPLETE';}else state='READY_FOR_CAPTURE';}return true;}catch(e){state='REJECTED';kind=null;jig=null;return false;}}
 function snapshot(){return {kind,state,armed,captured:kind==='pivot'?poses.length:kind==='jig'?points.length:0,pointerCalibrated:!!pivot,jigCalibrated:!!jig,pointerRmsMm:pivot?pivot.fitRmsM*1000:null,pointerHoldoutMm:pivot?pivot.holdoutM*1000:null,jigHoldoutMm:jig?jig.residualM*1000:null,physicalMetrologyCertified:false,patientRegistration:false};}
 return Object.freeze({reset,begin,arm,capture,snapshot,jig:()=>jig,pivot:()=>pivot,tip:pose=>pivot?point(pose,pivot.offsetM):null});
}
function createVisual(THREE,scene){
 const group=new THREE.Group();group.name='S tracked passive calibration pointer and jig';scene.add(group);group.visible=false;
 const markerMat=new THREE.MeshBasicMaterial({color:0xd5b68a}),lineMat=new THREE.MeshBasicMaterial({color:0x84acbc,transparent:true,opacity:.65}),geo=new THREE.CylinderGeometry(1,1,1,8),Y=new THREE.Vector3(0,1,0);
 const pointer=new THREE.Mesh(geo,lineMat),tip=new THREE.Mesh(new THREE.SphereGeometry(.006,12,8),markerMat);group.add(pointer,tip);
 const jig=new THREE.Group();group.add(jig);
 const frame=new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints([[0,0,0],[.20,0,0],[.20,0,.15],[0,0,.15]].map(p=>new THREE.Vector3(...p))),new THREE.LineBasicMaterial({color:0x8db4be}));jig.add(frame);
 for(const p of [[0,0,0],[.2,0,0],[0,0,.15],[.2,0,.15]]){const m=new THREE.Mesh(new THREE.SphereGeometry(.004,8,6),markerMat);m.position.set(...p);jig.add(m);}
 function update(pose,cal,show){group.visible=Boolean(show);if(!show)return;const target=pose&&cal.tip(pose);pointer.visible=tip.visible=!!target;if(target){const a=new THREE.Vector3(...pose.position),b=new THREE.Vector3(...target),d=b.clone().sub(a);pointer.position.copy(a).lerp(b,.5);pointer.quaternion.setFromUnitVectors(Y,d.clone().normalize());pointer.scale.set(.0025,d.length(),.0025);tip.position.copy(b);}
 const s=cal.jig();jig.visible=!!s;if(s){jig.position.set(...s.origin);jig.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(...s.basis.map(v=>new THREE.Vector3(...v))));}}
 return {group,update,hide(){group.visible=false;}};
}
// Only runtime plane polygons. Depth-writing, colorless geometry masks virtual objects;
// never reconstructs room walls or invents per-eye depth.
function createOccluder(THREE,scene){
 const group=new THREE.Group();group.name='S03 optional detected-plane depth mask';scene.add(group);group.visible=false;const material=new THREE.MeshBasicMaterial({colorWrite:false,depthWrite:true,depthTest:true,side:THREE.DoubleSide}),records=new Map();let state='OFF';
 function clear(){for(const m of records.values()){group.remove(m);m.geometry.dispose();}records.clear();group.visible=false;state='OFF';}
 function step(frame,space,enabled){group.visible=false;if(!enabled){clear();return;}let detected;try{detected=frame?.detectedPlanes;}catch{}if(!detected||!space){clear();state='PLANE_DATA_UNAVAILABLE';return;}const seen=new Set();for(const p of detected){if(seen.size>=8)break;try{const polygon=Array.from(p.polygon||[]);if(polygon.length<3||polygon.length>64||!polygon.every(v=>[v.x,v.y,v.z].every(finite)&&Math.abs(v.y)<.03&&Math.abs(v.x)<6&&Math.abs(v.z)<6))continue;
 const transform=frame.getPose(p.planeSpace,space)?.transform;if(!transform?.position||!transform?.orientation)continue;const coords=[transform.position.x,transform.position.y,transform.position.z,transform.orientation.x,transform.orientation.y,transform.orientation.z,transform.orientation.w];if(!coords.every(finite))continue;
 seen.add(p);let m=records.get(p);const key=polygon.map(v=>[v.x,v.z].join(',')).join(';');if(!m||m.userData.key!==key){const shape=new THREE.Shape(polygon.map(v=>new THREE.Vector2(v.x,-v.z))),geom=new THREE.ShapeGeometry(shape);geom.rotateX(-Math.PI/2);if(m){group.remove(m);m.geometry.dispose();}m=new THREE.Mesh(geom,material);m.userData.key=key;m.renderOrder=-90;group.add(m);records.set(p,m);}m.position.set(...coords.slice(0,3));m.quaternion.set(...coords.slice(3));}catch{}}
 for(const [p,m] of records)if(!seen.has(p)){group.remove(m);m.geometry.dispose();records.delete(p);}group.visible=seen.size>0;state=seen.size?'RUNTIME_PLANE_MASK':'NO_VALID_PLANES';
 }
 return {group,step,clear,snapshot:()=>({state,planeCount:records.size,perPixelDepth:false,roomReconstruction:false,physicalOcclusionValidated:false})};
}
root.PulmoXRSpatialCalibration=Object.freeze({dimensions,fitJig,fitPivot,point,create,createVisual,createOccluder});
})(typeof window!=='undefined'?window:globalThis);
