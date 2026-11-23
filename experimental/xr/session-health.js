/* PulmoSIMU XR-09 — local, bounded headset session observations.
 * Callback measurements are CPU-side; they are NOT GPU timings, clinical
 * validation, display reprojection metrics, or patient-data telemetry.
 * Nothing is uploaded or persisted. Export is explicit and user-initiated.
 */
(function(global){
  'use strict';
  const KINDS=['viewerPoseLost','handTrackingLost','controllerConnected',
    'controllerDisconnected','monitorUnavailable'];
  const finite=Number.isFinite;
  const quantile=(values,p)=>{
    if(!values.length)return null;
    const v=values.slice().sort((a,b)=>a-b);
    const k=(v.length-1)*p,lower=Math.floor(k),upper=Math.ceil(k);
    return +(v[lower]+(v[upper]-v[lower])*(k-lower)).toFixed(3);
  };
  const stats=(values)=>Object.freeze({
    samples:values.length,p50Ms:quantile(values,.50),p95Ms:quantile(values,.95),
    maxMs:values.length?+Math.max(...values).toFixed(3):null
  });
  function createSessionHealth({clock=()=>performance.now(),capacity=600}={}){
    if(typeof clock!=='function'||!Number.isInteger(capacity)||capacity<10||capacity>3600)
      throw Error('Invalid local session-health configuration');
    let active=false,startedAt=null,endedAt=null,reportedHz=null,previousXR=null;
    let pending=null,totalFrames=0,poseMissingFrames=0,slowIntervals=0;
    let samples=[],events=Object.fromEntries(KINDS.map(k=>[k,0]));
    function begin(frameRateHz=null){
      active=true;startedAt=clock();endedAt=null;
      reportedHz=finite(frameRateHz)&&frameRateHz>=30&&frameRateHz<=144?frameRateHz:null;
      previousXR=null;pending=null;totalFrames=0;poseMissingFrames=0;slowIntervals=0;
      samples=[];events=Object.fromEntries(KINDS.map(k=>[k,0]));
    }
    function startFrame(xrTimestamp){
      if(!active||!finite(xrTimestamp))return false;
      if(pending)endFrame({});
      const intervalMs=previousXR===null?null:xrTimestamp-previousXR;
      previousXR=xrTimestamp;
      pending={started:clock(),intervalMs:finite(intervalMs)&&intervalMs>=0&&intervalMs<1000?intervalMs:null};
      return true;
    }
    function endFrame({renderMs=0,monitorMs=0,poseAvailable=true}={}){
      if(!active||!pending)return false;
      const callbackMs=Math.max(0,clock()-pending.started);
      const frame={intervalMs:pending.intervalMs,
        callbackMs:finite(callbackMs)?callbackMs:null,
        renderMs:finite(renderMs)&&renderMs>=0?renderMs:null,
        monitorMs:finite(monitorMs)&&monitorMs>=0?monitorMs:null};
      if(poseAvailable===false)poseMissingFrames++;
      if(reportedHz&&frame.intervalMs!==null&&frame.intervalMs>1500/reportedHz)
        slowIntervals++;
      samples.push(frame);if(samples.length>capacity)samples.shift();
      totalFrames++;pending=null;return true;
    }
    function event(kind){if(!Object.hasOwn(events,kind))return false;events[kind]++;return true}
    function end(){if(active){if(pending)endFrame({});endedAt=clock();active=false}return snapshot()}
    function snapshot(){
      const measure=k=>stats(samples.map(x=>x[k]).filter(x=>x!==null));
      const interval=measure('intervalMs');
      const durationMs=startedAt===null?null:Math.max(0,(active?clock():endedAt)-startedAt);
      return Object.freeze({
        schema:'pulmosimu.xr.session-health.v1',
        classification:'local CPU/WebXR callback observation; not GPU or clinical validation',
        dataPolicy:'no images, patient identifiers, input poses, trajectories, geometry or network upload',
        started:startedAt!==null,active,sessionDurationSeconds:durationMs===null?null:+(durationMs/1000).toFixed(2),
        reportedHeadsetFrameRateHz:reportedHz,
        observedCallbackRateHz:interval.p50Ms?+(1000/interval.p50Ms).toFixed(2):null,
        frameBudgetMs:reportedHz?+(1000/reportedHz).toFixed(3):null,
        framesObserved:totalFrames,framesRetained:samples.length,
        callbackIntervals:interval,callbackCpu:measure('callbackMs'),
        stereoRenderCallCpu:measure('renderMs'),
        endoscopeCaptureCpu:measure('monitorMs'),
        intervalsOver1Point5Budgets:reportedHz?slowIntervals:null,
        viewerPoseMissingFrames:poseMissingFrames,
        trackingAndMonitorEvents:Object.freeze({...events})
      });
    }
    return Object.freeze({begin,startFrame,endFrame,event,end,snapshot});
  }
  global.PulmoXRSessionHealth=Object.freeze({createSessionHealth});
})(typeof window!=='undefined'?window:globalThis);
