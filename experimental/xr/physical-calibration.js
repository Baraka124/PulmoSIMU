/* PulmoSIMU — XR18 physical bronchoscope signal calibration laboratory.
 * Converts read-only, independently measured PULMOSIMU-BENCH/1 telemetry to
 * reference physical units. Never commands physical or virtual instruments.
 * A provisional linear three-point bench mapping is NOT certification.
 */
(function(root){
'use strict';
const FIELDS=Object.freeze({
 insertionMm:Object.freeze({unit:'mm',min:0,max:2000,target:5}),
 shaftRollDeg:Object.freeze({unit:'deg',min:-180,max:180,target:12}),
 tipPitchDeg:Object.freeze({unit:'deg',min:-180,max:180,target:8}),
 tipYawDeg:Object.freeze({unit:'deg',min:-180,max:180,target:8})
});
const finite=x=>typeof x==='number'&&Number.isFinite(x);
const stable=x=>x&&typeof x==='object'&&!Array.isArray(x);
function create({parseFrame,maxAgeMs=250}={}){
 if(typeof parseFrame!=='function')throw Error('FRAME_PARSER_REQUIRED');
 if(!finite(maxAgeMs)||maxAgeMs<50||maxAgeMs>1000)throw Error('INVALID_AGE_LIMIT');
 let model=null,latest=null,lastAt=-Infinity,lastSeq=-1,lastError=null;
 let accepted=0,rejected=0,safetyStops=0,paused=true;
 function stop(reason){
  if(!paused)safetyStops++;
  paused=true;lastError=reason;return snapshot();
 }
 function configure(config){
  // Reset all prior evidence before attempting a replacement.
  model=null;latest=null;lastAt=-Infinity;lastSeq=-1;
  paused=true;lastError=null;
  if(!stable(config)||config.protocol!=='PULMOSIMU-CALIBRATION/1'||
     typeof config.calibrationId!=='string'||!/^([A-Za-z0-9_-]){6,64}$/.test(config.calibrationId)||
     !stable(config.axes))throw Error('INVALID_CALIBRATION_ID_OR_FORMAT');
  const axes={};
  for(const [key,spec] of Object.entries(FIELDS)){
   const a=config.axes[key];
   if(!stable(a)||a.unit!==spec.unit||!Array.isArray(a.points)||a.points.length!==3)
    throw Error('AXIS_THREE_REFERENCES_REQUIRED_'+key);
   const [lower,upper,check]=a.points;
   for(const point of a.points){
    if(!stable(point)||!finite(point.reading)||!finite(point.reference)||
       point.reading<spec.min||point.reading>spec.max||
       point.reference<spec.min||point.reference>spec.max)
      throw Error('REFERENCE_OUT_OF_RANGE_'+key);
   }
   if(!(upper.reading>lower.reading+0.0001)||
      !(check.reading>lower.reading&&check.reading<upper.reading))
    throw Error('NONINVERTIBLE_OR_INVALID_CHECKPOINT_'+key);
   const scale=(upper.reference-lower.reference)/(upper.reading-lower.reading);
   // Mirrored or zero-slope axes are not accepted without a dedicated,
   // independently signed coordinate-frame conversion procedure.
   if(!finite(scale)||scale<=0.01||scale>50)throw Error('IMPLAUSIBLE_SCALE_'+key);
   const offset=lower.reference-scale*lower.reading;
   const residual=Math.abs(check.reference-(scale*check.reading+offset));
   const target=spec.target;
   if(residual>target)throw Error('VERIFICATION_RESIDUAL_EXCEEDS_PROVISIONAL_TARGET_'+key);
   axes[key]=Object.freeze({
    scale,offset,residual,unit:spec.unit,
    readingMin:lower.reading,readingMax:upper.reading,
    referenceMin:lower.reference,referenceMax:upper.reference,
    provisionalTarget:target
   });
  }
  model=Object.freeze({calibrationId:config.calibrationId,axes:Object.freeze(axes),
   basis:'THREE_OPERATOR_ENTERED_BENCH_POINTS_PER_AXIS',
   physicalMetrologyValidated:false,clinicalApproval:false});
  return snapshot();
 }
 function ingest(line,nowMs){
  if(!model){rejected++;return Object.freeze({accepted:false,reason:'CALIBRATION_REQUIRED'})}
  if(!finite(nowMs)||nowMs<0){rejected++;return Object.freeze({accepted:false,reason:'INVALID_TIME'})}
  let frame;
  try{frame=parseFrame(typeof line==='string'?line:JSON.stringify(line))}
  catch(err){rejected++;stop('FRAME_REJECTED');return Object.freeze({accepted:false,reason:String(err.message)})}
  if(frame.seq<=lastSeq){rejected++;stop('REPLAYED_SAMPLE');return Object.freeze({accepted:false,reason:'REPLAYED_SAMPLE'})}
  const corrected={};
  for(const [key,cal] of Object.entries(model.axes)){
   if(frame[key]<cal.readingMin||frame[key]>cal.readingMax){
    rejected++;stop('OUTSIDE_CALIBRATED_RANGE_'+key);
    return Object.freeze({accepted:false,reason:'OUTSIDE_CALIBRATED_RANGE_'+key});
   }
   const mapped=cal.scale*frame[key]+cal.offset;
   if(!finite(mapped)||mapped<cal.referenceMin-.001||mapped>cal.referenceMax+.001){
    rejected++;stop('MAPPING_RANGE_FAULT_'+key);
    return Object.freeze({accepted:false,reason:'MAPPING_RANGE_FAULT_'+key});
   }
   corrected[key]=Number(mapped.toFixed(4));
  }
  lastSeq=frame.seq;lastAt=nowMs;latest=Object.freeze(corrected);paused=false;
  accepted++;lastError=null;
  return Object.freeze({accepted:true,seq:frame.seq,corrected:latest,
   drivesVirtualScope:false,clinicalApproval:false});
 }
 function snapshot(nowMs){
  const fresh=finite(nowMs)&&latest!==null&&!paused&&
    nowMs>=lastAt&&nowMs-lastAt<=maxAgeMs;
  const state=!model?'NO_CALIBRATION':
    !latest?'AWAITING_INPUT':paused?'SAFETY_STOP':
    !fresh?'STALE_SAMPLE':'BENCH_MAPPING_ONLY';
  return Object.freeze({
   version:'PULMOSIMU_CALIBRATION_MAPPING_V1',state,
   calibrationId:model?.calibrationId||null,
   axes:model?.axes||null,corrected:fresh?latest:null,
   accepted,rejected,safetyStops,lastError,
   maxSampleAgeMs:maxAgeMs,
   // No command bus. No patient connection. Never a clinical credential.
   source:'LOCAL_PHYSICAL_BENCH_READINGS_ONLY',deviceToScopeCoupled:false,
   liveDeviceValidation:false,clinicalApproval:false,
   measurementAccuracyCertified:false,
   requires:'INDEPENDENT_METROLOGY_SIGNOFF_AND_CLINICIAN_REVIEW'
  });
 }
 function clear(){model=null;latest=null;lastAt=-Infinity;lastSeq=-1;
  lastError=null;accepted=0;rejected=0;safetyStops=0;paused=true;
  return snapshot()}
 return Object.freeze({configure,ingest,stop,snapshot,clear});
}
root.PulmoXRPhysicalCalibration=Object.freeze({create,FIELDS});
})(typeof window!=='undefined'?window:globalThis);
