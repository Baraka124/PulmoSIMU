/* PulmoSIMU E4 · conservative performance/evidence assessor.
 * Browser CPU callback metrics cannot prove GPU render time, optical comfort,
 * motion-to-photon latency or 90/72 Hz headset display delivery.
 */
(function(root){
'use strict';
function assess(report={}){
 const s=report.session||{},samples=s.callbackIntervals||s.callbackCpu||null;
 const poseLoss=Number(s?.events?.viewerPoseLost||0),monitor=report.endoscopeMonitor||{};
 const span=report.xrGeometryAtExit?.modelSpanM;
 const source=report.scienceIntegrity?.source||'NOT_RECORDED';
 const findings=[];
 if(!Number.isFinite(span))findings.push('No physical XR model-span observation');
 else if(span>.92)findings.push('XR model span exceeds conservative 0.92 m comfort bound');
 if(monitor.status==='unavailable')findings.push('Endoscopic source monitor was unavailable');
 if(poseLoss>0)findings.push('Viewer pose tracking was lost during session');
 if(!samples)findings.push('No callback CPU/frame interval diagnostic supplied');
 if(source==='UNVERIFIED_IMPORT')findings.push('Imported geometry has no confirmed source identity');
 return Object.freeze({
   softwareReportPresent:Boolean(s&&Object.keys(s).length),
   observedIssues:findings,
   headsetPhysicalAcceptance:'PENDING_EXPLICIT_QUEST_2_RUN',
   scientificValidation:'REQUIRES_INDEPENDENT_REVIEW',
   gpuFrameTime:'NOT_MEASURED',
   motionToPhotonLatency:'NOT_MEASURED',
   regulatoryDeviceStatus:'NOT_CLAIMED',
   researchFreezeEligible:false
 });
}
const gates=Object.freeze([
 {id:'E1',evidence:'Quest room stability, two-hand, orientation lock and recovery',kind:'PHYSICAL_QUEST_REQUIRED'},
 {id:'E2',evidence:'Verified CT voxel/GLB hash, affine, MPR cross-sections',kind:'SOURCE_AND_BROWSER_TESTS'},
 {id:'E3',evidence:'Airway/lobe/vessel masks and physics/instrument validation',kind:'INDEPENDENT_SCIENTIFIC_REVIEW'},
 {id:'E4',evidence:'10-minute Quest 2 performance, input, stereo and comfort record',kind:'PHYSICAL_QUEST_REQUIRED'}
]);
root.PulmoXRAcceptance=Object.freeze({assess,gates});
})(typeof window!=='undefined'?window:globalThis);
