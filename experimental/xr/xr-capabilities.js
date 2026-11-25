/* PulmoSIMU XR-17A — capability-negotiated runtime.
 * Presentation capability, optional session grants and scientific validity
 * are separate questions. No new anatomy, hardware tracking or room mesh.
 * Browser WebXR APIs must be queried, never inferred from product branding.
 */
(function(root){
'use strict';
const MODES=Object.freeze({
 vr:Object.freeze({sessionMode:'immersive-vr',presentation:'vr',
   optionalFeatures:Object.freeze(['local-floor','bounded-floor','hand-tracking'])}),
 mr:Object.freeze({sessionMode:'immersive-ar',presentation:'mr',
   optionalFeatures:Object.freeze(['local-floor','bounded-floor','hand-tracking',
     'hit-test','anchors'])})
});
const FEATURES=Object.freeze(['local-floor','bounded-floor','hand-tracking',
  'hit-test','anchors','layers','depth-sensing']);
const PRESENTATIONS=Object.freeze(['desktop','vr','mr']);
const validMode=m=>m==='ar'?'mr':m;
function create({xr=null}={}){
 const support={vr:'checking',mr:'checking'};
 let probeGeneration=0,probed=false,activeSession=null,sessionMode='desktop';
 let lastError=null,requestedMode=null,sessionFeatures={},featureReporting='no-session';
 let requestedFeatures=[],lastSession=null;
 const serializableFeatureReport=()=>Object.freeze({...sessionFeatures});
 function snapshot(){
  return Object.freeze({
   version:'XR17A_CAPABILITY_REGISTRY_V1',
   desktop:'available',
   webxrAvailable:Boolean(xr&&typeof xr.requestSession==='function'),
   sessions:Object.freeze({...support}),
   probed,activeMode:sessionMode,requestedMode,lastError,
   requestedFeatures:Object.freeze(requestedFeatures.slice()),
   actualFeatures:serializableFeatureReport(),featureReporting,
   spatialRegistration:'NOT_IMPLEMENTED',
   physicalOcclusion:'NOT_IMPLEMENTED',
   worldAnchor:'NOT_IMPLEMENTED',
   patientSpecificBronchoscopy:'NOT_VALIDATED',
   sessionActive:Boolean(activeSession),
   lastSession:lastSession?Object.freeze({...lastSession}):null
  });
 }
 async function probe(){
  const gen=++probeGeneration;
  if(!xr||typeof xr.requestSession!=='function'||
     typeof xr.isSessionSupported!=='function'){
   if(gen===probeGeneration){
    support.vr='unavailable';support.mr='unavailable';probed=true;
   }
   return snapshot();
  }
  const results=await Promise.allSettled(
   [xr.isSessionSupported('immersive-vr'),xr.isSessionSupported('immersive-ar')]
  );
  if(gen!==probeGeneration)return snapshot();
  for(const [i,mode] of ['vr','mr'].entries()){
   const v=results[i];
   support[mode]=v.status==='fulfilled'
     ?(v.value===true?'supported':'unsupported')
     :'error';
  }
  probed=true;return snapshot();
 }
 function request(mode){
  mode=validMode(mode);
  const spec=MODES[mode];
  if(!spec)throw Error('UNKNOWN_XR_PRESENTATION');
  if(!xr||typeof xr.requestSession!=='function')throw Error('WEBXR_NOT_AVAILABLE');
  if(support[mode]==='unsupported'||support[mode]==='unavailable')
   throw Error(mode==='mr'?'IMMERSIVE_AR_UNSUPPORTED':'IMMERSIVE_VR_UNSUPPORTED');
  // Deliberately invoke requestSession synchronously in the user's gesture.
  // A preliminary await can consume transient activation on Quest Browser.
  requestedMode=mode;
  requestedFeatures=spec.optionalFeatures.slice();
  lastError=null;
  let outcome;
  try{outcome=xr.requestSession(spec.sessionMode,{
    optionalFeatures:requestedFeatures.slice()
  })}catch(error){
   lastError=String(error?.message||error).slice(0,180);
   throw error;
  }
  return Promise.resolve(outcome).then(session=>{
   if(!session)throw Error('EMPTY_XR_SESSION');
   // A session object is not proof that requested optional features
   // were granted; bind separately on THREE's sessionstart event.
   return session;
  },error=>{
   lastError=String(error?.message||error).slice(0,180);
   throw error;
  });
 }
 function bind(session,mode=requestedMode){
  mode=validMode(mode);
  if(!MODES[mode]||!session)throw Error('UNRECOGNIZED_XR_SESSION');
  activeSession=session;sessionMode=mode;
  requestedMode=mode;lastError=null;
  const enabled=session.enabledFeatures;
  let granted=null;
  if(enabled&&typeof enabled[Symbol.iterator]==='function'){
   try{granted=new Set(Array.from(enabled,el=>String(el)))}catch{granted=null}
  }
  featureReporting=granted?'reported-by-runtime':'runtime-did-not-report-enabled-features';
  sessionFeatures=Object.fromEntries(FEATURES.map(feature=>[
   feature,!requestedFeatures.includes(feature)?'not-requested':
     granted?(granted.has(feature)?'enabled':'not-granted'):'not-reported'
  ]));
  return snapshot();
 }
 function end(reason='session-ended'){
  if(activeSession){
   lastSession={mode:sessionMode,featureReporting,endReason:String(reason).slice(0,90)};
  }
  activeSession=null;sessionMode='desktop';
  sessionFeatures={};featureReporting='no-session';
  requestedMode=null;requestedFeatures=[];
  return snapshot();
 }
 function failure(error){
  lastError=String(error?.message||error||'UNKNOWN_XR_ERROR').slice(0,180);
  if(!activeSession){requestedMode=null;requestedFeatures=[]}
  return snapshot();
 }
 return Object.freeze({probe,request,bind,end,failure,snapshot});
}
root.PulmoXRCapabilities=Object.freeze({create,MODES,FEATURES,PRESENTATIONS});
})(typeof window!=='undefined'?window:globalThis);
