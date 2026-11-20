/**
 * PulmoSIMU XR spatial interaction mathematics.
 * Standalone and deterministic. Coordinates are *display metres* in XR world
 * space. They do NOT establish physical dimensions of the source anatomy.
 */
const finite3=v=>Array.isArray(v)&&v.length===3&&v.every(Number.isFinite);
const add=(a,b)=>a.map((v,i)=>v+b[i]);
const sub=(a,b)=>a.map((v,i)=>v-b[i]);
const mul=(a,k)=>a.map(v=>v*k);
const dot=(a,b)=>a.reduce((n,v,i)=>n+v*b[i],0);
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const length=a=>Math.hypot(...a);
const normalize=a=>mul(a,1/(length(a)||1));
const midpoint=(a,b)=>mul(add(a,b),.5);
export function clampDisplayScale(s,min=.55,max=1.9){
 if(!Number.isFinite(s)||!(min>0)||!(max>=min))throw Error('Invalid XR display scale');
 return Math.min(max,Math.max(min,s));
}
export function normalizedQuaternion(q){
 if(!Array.isArray(q)||q.length!==4||!q.every(Number.isFinite))throw Error('Invalid quaternion');
 const n=Math.hypot(...q);if(n<1e-9)throw Error('Zero quaternion');
 return q.map(v=>v/n);
}
export function multiplyQuaternion(a,b){
 const [x,y,z,w]=normalizedQuaternion(a),[X,Y,Z,W]=normalizedQuaternion(b);
 return normalizedQuaternion([w*X+x*W+y*Z-z*Y,w*Y-x*Z+y*W+z*X,w*Z+x*Y-y*X+z*W,w*W-x*X-y*Y-z*Z]);
}
export function rotationBetweenVectors(a,b){
 if(!finite3(a)||!finite3(b)||length(a)<1e-6||length(b)<1e-6)throw Error('Invalid grab axis');
 const u=normalize(a),v=normalize(b),d=Math.max(-1,Math.min(1,dot(u,v)));
 if(d>1-1e-7)return [0,0,0,1];
 if(d<-1+1e-7){
   const ortho=Math.abs(u[0])<.9?[1,0,0]:[0,1,0];
   const axis=normalize(cross(u,ortho));
   return [...axis,0];
 }
 return normalizedQuaternion([...cross(u,v),1+d]);
}
export function rotateVector(q,v){
 if(!finite3(v))throw Error('Invalid position');
 const [x,y,z,w]=normalizedQuaternion(q),u=[x,y,z],t=mul(cross(u,v),2);
 return add(v,add(mul(t,w),cross(u,t)));
}
/**
 * initial: {left:[x,y,z],right:[x,y,z],position:[x,y,z],
 *           quaternion:[x,y,z,w],scale:number}
 * Each frame supplies current left/right world grip points. Keeps transform
 * centered on hands' midpoint and orients using the hands' connecting axis.
 * Degenerate coincident grips fail closed to the initial pose.
 */
export function bimanualPose(initial,left,right,{minScale=.55,maxScale=1.9}={}){
 if(!initial||![initial.left,initial.right,initial.position,left,right].every(finite3)||
   !Number.isFinite(initial.scale)||initial.scale<=0)throw Error('Invalid XR hand pose');
 const initialAxis=sub(initial.right,initial.left),axis=sub(right,left);
 const initialDist=length(initialAxis),currentDist=length(axis);
 if(initialDist<.06||currentDist<.06)return {position:[...initial.position],quaternion:normalizedQuaternion(initial.quaternion),scale:clampDisplayScale(initial.scale,minScale,maxScale)};
 const ratio=clampDisplayScale(initial.scale*currentDist/initialDist,minScale,maxScale)/initial.scale;
 const dq=rotationBetweenVectors(initialAxis,axis);
 const pos=add(midpoint(left,right),mul(rotateVector(dq,sub(initial.position,midpoint(initial.left,initial.right))),ratio));
 return {position:pos,quaternion:multiplyQuaternion(dq,initial.quaternion),scale:initial.scale*ratio};
}
export function normalizeProgress(v){return Math.max(0,Math.min(1,Number.isFinite(v)?v:0))}
export function dominantThumbstick(axes){
 if(!Array.isArray(axes)&&!ArrayBuffer.isView(axes))return [0,0];
 const a=Array.from(axes);const i=a.length>=4?2:0;
 return [Number.isFinite(a[i])?a[i]:0,Number.isFinite(a[i+1])?a[i+1]:0].map(v=>Math.abs(v)<.14?0:Math.max(-1,Math.min(1,v)));
}
