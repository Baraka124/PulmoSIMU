(function sessionReview(root){
'use strict';
const finite=Number.isFinite,vec=(p,max)=>Array.isArray(p)&&p.length===3&&p.every(v=>finite(v)&&Math.abs(v)<=max),round=v=>Math.round(v*1000)/1000;
const cap=6000,version='pulmosimu.reference-motion/1',source='PULMOSIMU_REFERENCE_TREE_U03',hz=10,maxTime=600000;
function check(x,m){if(!x)throw Error(m);}
function hand(h){if(h===null||h===undefined)return null;check(['HAND_WRIST','CONTROLLER_GRIP'].includes(h.kind)&&vec(h.relative_m,2.5),'INVALID_HAND');return {kind:h.kind,relative_m:h.relative_m.map(round),grip:h.grip===true};}
function sample(s){
 check(s&&Number.isInteger(s.t_ms)&&s.t_ms>=0&&s.t_ms<=maxTime&&Number.isInteger(s.segment)&&s.segment>=0&&s.segment<=1000&&finite(s.t)&&s.t>=0&&s.t<=1&&vec(s.tip_mm,2000)&&finite(s.roll_rad)&&Math.abs(s.roll_rad)<=Math.PI+.001&&Array.isArray(s.deflection)&&s.deflection.length===2&&s.deflection.every(v=>finite(v)&&Math.abs(v)<=1.001)&&['off','radial','convex'].includes(s.ebus)&&finite(s.probe_mm)&&s.probe_mm>=0&&s.probe_mm<=12,'INVALID_SAMPLE');
 check(s.risk===null||finite(s.risk)&&s.risk>=0&&s.risk<=1,'INVALID_RISK');
 return {t_ms:s.t_ms,segment:s.segment,t:Math.round(s.t*100000)/100000,tip_mm:s.tip_mm.map(round),roll_rad:round(s.roll_rad),deflection:s.deflection.map(round),left:hand(s.left),right:hand(s.right),risk:s.risk===null?null:round(s.risk),ebus:s.ebus,probe_mm:round(s.probe_mm),break_before:s.break_before===true};
}
function sanitize(doc){
 check(doc?.schema===version&&doc.source===source&&doc.reference_only===true&&doc.clinical_navigation_allowed===false&&Array.isArray(doc.samples)&&doc.samples.length>=1&&doc.samples.length<=cap,'INVALID_REFERENCE_RECORD');
 const samples=doc.samples.map(sample);for(let i=1;i<samples.length;i++)check(samples[i].t_ms>samples[i-1].t_ms&&samples[i].t_ms-samples[i-1].t_ms>=95,'INVALID_TIME_ORDER');
 samples[0].break_before=true;
 for(let i=1;i<samples.length;i++){const a=samples[i-1],b=samples[i];if(b.t_ms-a.t_ms>250||a.segment!==b.segment||distance(a.tip_mm,b.tip_mm)>20)b.break_before=true;}
 return {schema:version,source,reference_only:true,clinical_navigation_allowed:false,sample_hz:hz,samples};
}
function distance(a,b){return Math.hypot(...a.map((v,i)=>v-b[i]));}
function metrics(samples){
 let tip=0,active=0,breaks=0,riskN=0,high=0,left=0,right=0,leftN=0,rightN=0,peak=0,roll=0,probe=0,gripChanges=0,seen=new Set();
 for(let i=0;i<samples.length;i++){const b=samples[i],a=samples[i-1];seen.add(b.segment);if(b.risk!==null){riskN++;if(b.risk>=.5)high++;}if(!a)continue;if(b.break_before){breaks++;continue;}const dt=b.t_ms-a.t_ms;if(dt<=0||dt>250||a.segment!==b.segment||distance(a.tip_mm,b.tip_mm)>20){breaks++;continue;}const d=distance(a.tip_mm,b.tip_mm);active+=dt;tip+=d;peak=Math.max(peak,d*1000/dt);roll+=Math.abs(((b.roll_rad-a.roll_rad+3*Math.PI)%(2*Math.PI))-Math.PI);if(a.ebus===b.ebus&&b.ebus==='radial')probe+=Math.abs(b.probe_mm-a.probe_mm);
 for(const side of ['left','right']){const u=a[side],v=b[side];if(u&&v&&u.kind===v.kind){const delta=distance(u.relative_m,v.relative_m);if(delta<=.15){if(side==='left'){left+=delta;leftN++;}else{right+=delta;rightN++;}}if(u.grip!==v.grip)gripChanges++;}}}
 return {sampleCount:samples.length,elapsedS:samples.length>1?(samples.at(-1).t_ms-samples[0].t_ms)/1000:0,activeS:active/1000,segments:seen.size,withinSegmentTipTravelMm:round(tip),peakSampledTipSpeedMmS:round(peak),headRelativeLeftTravelMm:leftN?round(left*1000):null,headRelativeRightTravelMm:rightN?round(right*1000):null,rollTravelDeg:round(roll*180/Math.PI),radialProbeTravelMm:round(probe),observedGripTransitions:gripChanges,highRiskSampleFraction:riskN?high/riskN:null,discontinuities:breaks,competencyValidated:false,forceMeasured:false};
}
function create(){
 let rows=[],running=false,startAt=0,last=-Infinity,pendingBreak=true,reason='IDLE';
 function start(now){check(finite(now),'INVALID_CLOCK');rows=[];running=true;startAt=now;last=-Infinity;pendingBreak=true;reason='RECORDING';}
 function stop(why='STOPPED'){running=false;reason=why;pendingBreak=true;}
 function suspend(){pendingBreak=true;}
 function observe(now,input){if(!running)return null;if(!finite(now)||now<startAt||now<last){stop('CLOCK_ERROR');return null;}if(now-startAt>maxTime||rows.length>=cap){stop('RECORDING_LIMIT');return null;}if(!input){suspend();return null;}if(now-last<100)return null;
 try{const prev=rows.at(-1),s=sample({...input,t_ms:Math.round(now-startAt),break_before:pendingBreak||!prev||now-last>250||prev.segment!==input.segment||distance(prev.tip_mm,input.tip_mm)>20});rows.push(s);last=now;pendingBreak=false;return s;}catch{pendingBreak=true;return null;}}
 function packet(){return rows.length?sanitize({schema:version,source,reference_only:true,clinical_navigation_allowed:false,samples:rows}):null;}
 return {start,stop,suspend,observe,packet,snapshot:()=>({running,reason,sampleCount:rows.length,durationS:rows.length?rows.at(-1).t_ms/1000:0,referenceOnly:true,networkUpload:false}),clear(){stop('CLEARED');rows=[];}};
}
function draw(canvas,rows,index=rows.length-1,yaw=.35,pitch=.25){
 const c=canvas.getContext('2d'),w=canvas.width,h=canvas.height;c.fillStyle='#101a21';c.fillRect(0,0,w,h);if(!rows.length)return;
 index=Math.max(0,Math.min(index,rows.length-1));const b=rows[index];const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
 const project=p=>[p[0]*cy+p[2]*sy,p[1]*cp-(-p[0]*sy+p[2]*cy)*sp];
 const pts=rows.map(r=>project(r.tip_mm));let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;for(const p of pts){minX=Math.min(minX,p[0]);maxX=Math.max(maxX,p[0]);minY=Math.min(minY,p[1]);maxY=Math.max(maxY,p[1]);}
 const span=Math.max(maxX-minX,maxY-minY,20),scale=Math.min(w*.55,h*.76)/span,screen=p=>[w*.31+(p[0]-(minX+maxX)/2)*scale,h*.52-(p[1]-(minY+maxY)/2)*scale];
 c.lineWidth=1.5;c.strokeStyle='#84a8b6';c.beginPath();for(let i=0;i<=index;i++){const p=screen(pts[i]);if(i===0||rows[i].break_before)c.moveTo(...p);else c.lineTo(...p);}c.stroke();const t=screen(pts[index]);c.fillStyle='#d7b181';c.beginPath();c.arc(...t,4,0,Math.PI*2);c.fill();
 c.strokeStyle='#354751';c.beginPath();c.moveTo(w*.62,10);c.lineTo(w*.62,h-10);c.stroke();c.fillStyle='#aabec6';c.font='12px system-ui';c.fillText('REFERENCE TIP',12,18);c.fillText('HEAD-RELATIVE HANDS',w*.64,18);
 const hx=w*.81,hy=h*.40,k=Math.min(w*.15,h*.34)/.65;c.strokeStyle='#526772';c.beginPath();c.arc(hx,hy,7,0,Math.PI*2);c.stroke();for(const [side,color] of [['left','#8baebc'],['right','#d2b17e']]){c.strokeStyle=color;c.beginPath();let prev=null;for(let i=Math.max(0,index-49);i<=index;i++){const r=rows[i],v=r[side];if(!v){prev=null;continue;}const p=[hx+v.relative_m[0]*k,hy-v.relative_m[1]*k];if(!prev||r.break_before||prev.kind!==v.kind)c.moveTo(...p);else c.lineTo(...p);prev=v;}c.stroke();const v=b[side];if(v){c.fillStyle=color;c.beginPath();c.arc(hx+v.relative_m[0]*k,hy-v.relative_m[1]*k,4,0,Math.PI*2);c.fill();}}
 c.fillStyle='#aabec6';c.fillText((b.t_ms/1000).toFixed(1)+' s · SEG '+b.segment,12,h-12);
}
root.PulmoXRSessionReview=Object.freeze({create,sanitize,sample,metrics,draw,version,source});
})(typeof window!=='undefined'?window:globalThis);
