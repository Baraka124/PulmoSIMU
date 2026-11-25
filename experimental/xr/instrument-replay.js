/* PulmoSIMU XR-17C — read-only augmented-virtuality instrument trace review.
 * Data are loaded locally and assessed using the EXISTING XR-16F measured
 * instrument validator. No unverified hardware may control the bronchoscope.
 * This is a time-indexed comparison, not spatial tracking or clinical coupling.
 */
(function(root){
'use strict';
const finite=n=>typeof n==='number'&&Number.isFinite(n);
const clamp=(n,a,b)=>Math.min(b,Math.max(a,n));
const arcDiff=(a,b)=>((a-b+540)%360)-180;
function create({assess}={}){
 if(typeof assess!=='function')throw Error('BENCH_ASSESSOR_REQUIRED');
 let trace=null,assessment=null,index=0,playing=false,origin=null,elapsedMs=0,
   state='empty',lastError=null,loads=0,blocked=0;
 function sample(){
  if(!trace)return null;
  const s=trace[index];
  return Object.freeze({
   index,total:trace.length,tMs:s.tMs,tracking:s.tracking,
   commanded:Object.freeze({insertionMm:s.cmdInsertionMm,
     tipHDeg:s.cmdTipHDeg,tipVDeg:s.cmdTipVDeg,rollDeg:s.cmdRollDeg}),
   measured:Object.freeze({insertionMm:s.measInsertionMm,
     tipHDeg:s.measTipHDeg,tipVDeg:s.measTipVDeg,rollDeg:s.measRollDeg}),
   residual:Object.freeze({
     insertionMm:+(s.measInsertionMm-s.cmdInsertionMm).toFixed(3),
     tipHDeg:+(s.measTipHDeg-s.cmdTipHDeg).toFixed(3),
     tipVDeg:+(s.measTipVDeg-s.cmdTipVDeg).toFixed(3),
     rollDeg:+arcDiff(s.measRollDeg,s.cmdRollDeg).toFixed(3)
   })
  });
 }
 function snapshot(){
  return Object.freeze({
   version:'XR17C_TRACE_REVIEW_V1',
   state,loaded:Boolean(trace),playing,reviewIndex:index,
   total:trace?.length||0,elapsedMs,loadCount:loads,rejectedCount:blocked,
   engineeringStatus:assessment?.status||'NOT_EVALUATED',
   defects:assessment?.defects?.slice(0,18)||[],
   // Summary only. No external device name/serial, original trace or
   // patient information is copied into XR session export.
   measuredMetrics:assessment?.metrics?Object.freeze({...assessment.metrics}):null,
   observation:sample(),readOnly:true,deviceToScopeCoupled:false,
   clinicalValidated:false,actualLiveInstrumentTracking:false,
   temporalAlignment:'LOCAL_TRACE_TIME_ONLY_NOT_SYNCHRONIZED_WITH_XR'
  });
 }
 function clear(){
  trace=null;assessment=null;index=0;playing=false;origin=null;elapsedMs=0;
  state='empty';lastError=null;return snapshot();
 }
 function load(file){
  let result;
  try{result=assess(file)}catch(e){
   blocked++;lastError=String(e?.message||e).slice(0,110);
   return Object.freeze({ok:false,reason:'ASSESSOR_ERROR'});
  }
  if(!result||result.status==='REJECTED'||!Array.isArray(file?.samples)||
      file.samples.length>30000){
   blocked++;lastError=(result?.defects||['INVALID_TRACE']).join(',').slice(0,150);
   return Object.freeze({ok:false,reason:'TRACE_REJECTED',defects:result?.defects||[]});
  }
  // Copy only numeric/boolean columns, never patient/source metadata.
  const cols=['tMs','cmdInsertionMm','measInsertionMm','cmdTipHDeg',
    'measTipHDeg','cmdTipVDeg','measTipVDeg','cmdRollDeg','measRollDeg'];
  const cleaned=[];
  for(const item of file.samples){
   if(!item||typeof item.tracking!=='boolean'||cols.some(k=>!finite(item[k]))){
    blocked++;return Object.freeze({ok:false,reason:'UNSAFE_TRACE_ROW'});
   }
   cleaned.push(Object.freeze(Object.fromEntries([
     ...cols.map(k=>[k,item[k]]),['tracking',item.tracking]])));
  }
  trace=cleaned;assessment=result;index=0;playing=false;origin=null;
  elapsedMs=0;loads++;state='paused';lastError=null;
  return Object.freeze({ok:true,engineeringStatus:result.status,total:trace.length,
    clinicalValidated:false,deviceToScopeCoupled:false});
 }
 function seek(target){
  if(!trace)return snapshot();
  if(!Number.isSafeInteger(target))return snapshot();
  index=clamp(target,0,trace.length-1);
  elapsedMs=trace[index].tMs-trace[0].tMs;
  origin=null;playing=false;state='paused';return snapshot();
 }
 function step(direction){return seek(index+(direction>0?1:-1))}
 function play(now){
  if(!trace||!finite(now)||now<0)return false;
  if(index>=trace.length-1)seek(0);
  origin=now-elapsedMs;playing=true;state='playing';return true;
 }
 function tick(now){
  if(!trace||!playing||!finite(now)||now<0||!finite(origin))return snapshot();
  const final=trace[trace.length-1].tMs-trace[0].tMs;
  elapsedMs=clamp(now-origin,0,final);
  const target=trace[0].tMs+elapsedMs;
  // Binary search: a 30,000-sample session never makes every XR frame
  // iterate through every earlier sample.
  let lo=0,hi=trace.length-1;
  while(lo<hi){const m=(lo+hi+1)>>1;
   if(trace[m].tMs<=target)lo=m;else hi=m-1;
  }
  index=lo;
  if(index>=trace.length-1){playing=false;state='complete'}
  return snapshot();
 }
 function pause(now){
  if(playing)tick(now);
  playing=false;origin=null;
  if(trace)state='paused';
  return snapshot();
 }
 return Object.freeze({load,clear,seek,step,play,pause,tick,snapshot});
}
root.PulmoXRInstrumentReplay=Object.freeze({create});
})(typeof window!=='undefined'?window:globalThis);
