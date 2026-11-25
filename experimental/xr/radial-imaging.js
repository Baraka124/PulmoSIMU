/* XR30 source-derived radial interface echo research view.
 * The signed lumen field supplies first boundary crossings. Idealized
 * fluid coupling and pulse envelope only; air mode suppresses distal echoes.
 * No tissue acoustic map, measured transducer, speckle or EBUS diagnosis. */
(function(root){
'use strict';
const finite=Number.isFinite,clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
function scan({origin,right,up,field,mmPerUnit,rangeMm=20,rays=64,coupling='air'}){
 if(![origin,right,up].every(v=>Array.isArray(v)&&v.length===3&&v.every(finite))||typeof field!=='function'||!finite(mmPerUnit)||mmPerUnit<=0||!finite(rangeMm)||rangeMm<5||rangeMm>40||!Number.isInteger(rays)||rays<16||rays>96||!['air','ideal-fluid'].includes(coupling))throw Error('INVALID_RADIAL_SCAN');
 const f0=field(origin);if(!finite(f0)||f0>=0)return {state:'ORIGIN_OUTSIDE_LUMEN',rays:[],rangeMm,coupling};
 const values=[];const steps=24;
 for(let i=0;i<rays;i++){
  const a=i*2*Math.PI/rays,d=right.map((v,j)=>v*Math.cos(a)+up[j]*Math.sin(a));
  const point=mm=>origin.map((v,j)=>v+d[j]*mm/mmPerUnit);
  let last=0,hit=null;
  for(let k=1;k<=steps;k++){const mm=rangeMm*k/steps,v=field(point(mm));if(!finite(v))throw Error('INVALID_FIELD');if(v>=0){let lo=last,hi=mm;for(let j=0;j<5;j++){const mid=(lo+hi)/2;if(field(point(mid))>=0)hi=mid;else lo=mid;}hit=(lo+hi)/2;break;}last=mm;}
  values.push(hit);
 }
 const hits=values.filter(finite);
 return {state:'REFERENCE_INTERFACE_ONLY',rays:values,rangeMm,coupling,rayCount:rays,hitCount:hits.length,minWallMm:hits.length?Math.min(...hits):null,maxWallMm:hits.length?Math.max(...hits):null,patientSpecific:false,acousticValidated:false,tissueBeyondWallModeled:false};
}
function echo(distance,wall,coupling){
 if(!finite(distance)||!finite(wall)||wall<0||distance<0||coupling!=='ideal-fluid')return 0;
 return clamp(Math.exp(-.025*wall)*Math.exp(-.5*((distance-wall)/.32)**2),0,1);
}
function paint(canvas,result){
 const c=canvas.getContext('2d'),w=canvas.width,h=canvas.height,cx=w*.5,cy=h*.46,R=Math.min(w*.45,h*.39);
 c.fillStyle='#080d12';c.fillRect(0,0,w,h);
 c.strokeStyle='#31424c';c.lineWidth=1;
 for(let k=1;k<=4;k++){c.beginPath();c.arc(cx,cy,R*k/4,0,Math.PI*2);c.stroke();}
 if(result?.rays?.length){
  const N=result.rays.length;
  for(let i=0;i<N;i++){const a=i*2*Math.PI/N,d=result.rays[i];if(!finite(d))continue;
   const r=d/result.rangeMm*R;
   // Source boundary is a geometry contour in both modes; echo only in ideal-fluid.
   c.strokeStyle='#647985';c.lineWidth=1.5;c.beginPath();c.arc(cx,cy,r,a-.5*Math.PI/N,a+.5*Math.PI/N);c.stroke();
   if(result.coupling==='ideal-fluid')for(let k=-2;k<=2;k++){const mm=d+k*.26,v=echo(mm,d,result.coupling),rr=mm/result.rangeMm*R;if(rr<=0||rr>R)continue;const g=Math.round(220*v);c.strokeStyle='rgb('+g+','+g+','+g+')';c.lineWidth=2;c.beginPath();c.arc(cx,cy,rr,a-Math.PI/N,a+Math.PI/N);c.stroke();}
  }
 }
 c.fillStyle='#b1c4ce';c.font='600 17px system-ui';c.textAlign='center';
 c.fillText('RADIAL INTERFACE · REFERENCE',cx,21);
 c.fillStyle='#90a8b5';c.font='13px system-ui';
 c.fillText(result?.coupling==='ideal-fluid'?'IDEALIZED COUPLING · NO TISSUE DATA':'AIR MODE · NO TRANSMITTED TISSUE ECHO',cx,h-34);
 c.fillText(result?.state==='REFERENCE_INTERFACE_ONLY'?(result.rangeMm+' mm radius · '+result.hitCount+'/'+result.rayCount+' rays'):'SCAN UNAVAILABLE',cx,h-15);
}
root.PulmoXRRadialImaging=Object.freeze({scan,echo,paint});
})(typeof window!=='undefined'?window:globalThis);
