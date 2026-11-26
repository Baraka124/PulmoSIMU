/* Offline synchronization QA of independently timestamped raw four-axis observations.
 * No interpolation, calibration, transport or hardware commands. */
import {readFileSync,writeFileSync} from 'node:fs';import {pathToFileURL} from 'node:url';
const axes=['insertionMm','rotationDeg','tipPitchDeg','tipYawDeg'];
export function assessCapture(channels,{maxSkewMs=10,maxGapMs=100}={}){
 if(!Number.isFinite(maxSkewMs)||maxSkewMs<=0||!Number.isFinite(maxGapMs)||maxGapMs<=maxSkewMs)throw Error('INVALID_LIMITS');
 for(const axis of axes){const a=channels?.[axis];if(!Array.isArray(a)||a.length<2||a.length>100000||!a.every((r,i)=>Number.isFinite(r.tMs)&&Number.isFinite(r.raw)&&(i===0||r.tMs>a[i-1].tMs)))throw Error('INVALID_CHANNEL:'+axis);}
 const cursors=axes.map(()=>0),primary=channels[axes[0]],gaps={};let matched=0,unmatched=0,maxObservedSkewMs=0;
 for(const a of axes){gaps[a]=channels[a].slice(1).filter((r,i)=>r.tMs-channels[a][i].tMs>maxGapMs).length;}
 for(const p of primary){const times=[p.tMs];let ok=true;for(let i=1;i<axes.length;i++){const a=channels[axes[i]];let j=cursors[i];while(j+1<a.length&&Math.abs(a[j+1].tMs-p.tMs)<Math.abs(a[j].tMs-p.tMs))j++;cursors[i]=j;const r=a[j];if(!r||Math.abs(r.tMs-p.tMs)>maxSkewMs){ok=false;break;}times.push(r.tMs);}
 const skew=Math.max(...times)-Math.min(...times);if(ok&&skew<=maxSkewMs){matched++;maxObservedSkewMs=Math.max(maxObservedSkewMs,skew);}else unmatched++;}
 return {schema:'pulmosimu-offline-capture-qa/1',referenceChannel:axes[0],matched,unmatched,gaps,maxObservedSkewMs:matched?maxObservedSkewMs:null,maxSkewMs,maxGapMs,status:unmatched===0&&Object.values(gaps).every(v=>v===0)?'PROVISIONAL_CAPTURE_TARGET_MET':'REVIEW_REQUIRED',sameClockRequired:true,clockSynchronizationMeasured:false,physicalAcceptance:false,hardwareCommand:false,benchProtocolExported:false};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){const [,,input,output]=process.argv;if(!input||!output)throw Error('Usage: node capture-qa.mjs input.json report.json');const v=JSON.parse(readFileSync(input,'utf8'));writeFileSync(output,JSON.stringify(assessCapture(v.channels,v.limits),null,2)+'\n',{flag:'wx'});}
