/* XR27 — bounded, local reference-procedure observation.
 * Observes the original simulation; never controls an instrument.
 * No raw poses, images, patient identifiers, uploads or competency scores.
 */
(function(root){
'use strict';
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
function create({sampleHz=20,bins=20,maxGapMs=250}={}){
 if(!Number.isInteger(bins)||bins<5||bins>100||!Number.isFinite(sampleHz)||sampleHz<1||sampleHz>60)throw Error('INVALID_OBSERVATION_CONFIG');
 const interval=1000/sampleHz;
 let allowed=new Map(),visited=new Map(),last=null,prevSpeed=null,previousClock=null,counts={},activeMs=0,travel=0,speedSum=0,speedN=0,accelSum=0,accelN=0,peakSpeed=0,frames=0,discontinuities=0;
 function reset(segments=[]){allowed=new Map(segments.filter(s=>Number.isFinite(s.id)).map(s=>[s.id,String(s.name||s.id)]));visited=new Map();last=null;prevSpeed=null;previousClock=null;counts={clear:0,interference:0,bend:0,unknown:0};activeMs=travel=speedSum=speedN=accelSum=accelN=peakSpeed=frames=discontinuities=0;}
 function suspend(){last=null;prevSpeed=null;previousClock=null;}
 function observe({timeMs,procedure,reference,tracking,segmentId,t,positionMm,shaftState}){
  if(!procedure||!reference||!tracking||!allowed.has(segmentId)||!Number.isFinite(t)||!Number.isFinite(timeMs)||!Array.isArray(positionMm)||positionMm.length!==3||!positionMm.every(Number.isFinite)){suspend();return false;}
  if(previousClock!==null&&timeMs<=previousClock){discontinuities++;suspend();return false;}
  if(previousClock!==null&&timeMs-previousClock<interval)return false;
  const gap=last?timeMs-last.timeMs:0;
  if(gap>maxGapMs){discontinuities++;last=null;prevSpeed=null;}
  const bin=Math.min(bins-1,Math.floor(clamp(t,0,1)*bins));
  if(!visited.has(segmentId))visited.set(segmentId,new Set());
  visited.get(segmentId).add(bin);frames++;
  const state=shaftState==='GEOMETRIC_INTERFERENCE'?'interference':shaftState==='BENDING_LIMIT_EXCEEDED'?'bend':shaftState==='REFERENCE_ENVELOPE_CLEAR'?'clear':'unknown';counts[state]++;
  if(last&&gap>0&&gap<=maxGapMs){
   activeMs+=gap;
   // Do not bridge branch changes or discrete navigation jumps with invented trajectories.
   if(last.segmentId===segmentId){
    const d=Math.hypot(...positionMm.map((v,i)=>v-last.positionMm[i]));
    if(d<=20){
     travel+=d;const speed=d/(gap/1000);speedSum+=speed;speedN++;peakSpeed=Math.max(peakSpeed,speed);
     if(prevSpeed!==null){accelSum+=Math.abs(speed-prevSpeed)/(gap/1000);accelN++}prevSpeed=speed;
    }else{discontinuities++;prevSpeed=null;}
   }else{prevSpeed=null;}
  }
  previousClock=timeMs;last={timeMs,segmentId,positionMm:positionMm.slice()};return true;
 }
 function snapshot(){
  const rows=Array.from(allowed,([id,name])=>({id,name,observedBins:visited.get(id)?.size||0,totalBins:bins}));
  const observedBins=rows.reduce((n,r)=>n+r.observedBins,0);
  return {schema:'pulmosimu-reference-procedure/1',sampleHz,binsPerSegment:bins,samples:frames,activeSeconds:activeMs/1000,visitedSegments:visited.size,totalSegments:allowed.size,observedBins,totalBins:allowed.size*bins,routeBinCoverage:allowed.size?observedBins/(allowed.size*bins):0,segments:rows,withinSegmentTravelMm:travel,meanSpeedMmS:speedN?speedSum/speedN:null,peakSpeedMmS:speedN?peakSpeed:null,meanAbsoluteSpeedChangeMmS2:accelN?accelSum/accelN:null,shaftSampleCounts:{...counts},discontinuities,actualVisualCoverage:false,measuredContactForce:false,competencyValidated:false,rawTrajectoryExported:false};
 }
 reset();return {reset,suspend,observe,snapshot};
}
root.PulmoXRProcedureMetrics=Object.freeze({create});
})(typeof window!=='undefined'?window:globalThis);
