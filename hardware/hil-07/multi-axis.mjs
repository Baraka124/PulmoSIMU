/* Offline measured input analysis; no live transport or hardware commands. */
import {readFileSync,writeFileSync} from 'node:fs';
import {pathToFileURL} from 'node:url';
const axes=['insertionMm','rotationDeg','tipPitchDeg','tipYawDeg'];
const finite=Number.isFinite;
export function assess(rows,limits){
 if(!Array.isArray(rows)||!limits||!rows.every(r=>axes.includes(r.axis)&&['fit','check'].includes(r.role)&&['up','down'].includes(r.direction)&&finite(r.raw)&&finite(r.reference)))throw Error('INVALID_MEASURED_ROWS');
 const results={};
 for(const axis of axes){
  const data=rows.filter(r=>r.axis===axis),fit=data.filter(r=>r.role==='fit'),check=data.filter(r=>r.role==='check'),limit=limits[axis];
  if(!(finite(limit)&&limit>0)||fit.length<6||check.length<3||!['up','down'].every(d=>fit.some(r=>r.direction===d)&&check.some(r=>r.direction===d)))throw Error('MISSING_AXIS_OR_INDEPENDENT_CHECK:'+axis);
  const avg=(a,k)=>a.reduce((s,r)=>s+r[k],0)/a.length, mx=avg(fit,'raw'),my=avg(fit,'reference');
  const variance=fit.reduce((s,r)=>s+(r.raw-mx)**2,0);
  if(variance<1e-12||new Set(fit.map(r=>r.reference)).size<3)throw Error('INSUFFICIENT_SPAN:'+axis);
  const slope=fit.reduce((s,r)=>s+(r.raw-mx)*(r.reference-my),0)/variance,intercept=my-slope*mx;
  if(!finite(slope)||Math.abs(slope)<1e-12)throw Error('DEGENERATE_MAPPING:'+axis);
  const lo=Math.min(...fit.map(r=>r.raw)),hi=Math.max(...fit.map(r=>r.raw));
  const residual=check.map(r=>slope*r.raw+intercept-r.reference);
  const extrapolation=check.some(r=>r.raw<lo||r.raw>hi);
  const max=Math.max(...residual.map(Math.abs)),rmse=Math.sqrt(residual.reduce((s,v)=>s+v*v,0)/residual.length);
  const pairs=[];for(const u of check.filter(r=>r.direction==='up'))for(const d of check.filter(r=>r.direction==='down'&&r.reference===u.reference))pairs.push(Math.abs(slope*(u.raw-d.raw)));
  const hysteresis=pairs.length?Math.max(...pairs):null;
  results[axis]={slope,intercept,rawRange:[lo,hi],checkCount:check.length,checkRmse:rmse,checkMax:max,hysteresis,extrapolation,limit,status:!extrapolation&&max<=limit&&hysteresis!==null&&hysteresis<=limit?'PROVISIONAL_TARGET_MET':'TARGET_NOT_MET_OR_INSUFFICIENT_PAIRS'};
 }
 return {schema:'pulmosimu-offline-multiaxis/1',axes:results,status:Object.values(results).every(r=>r.status==='PROVISIONAL_TARGET_MET')?'PROVISIONAL_TARGETS_MET':'REVIEW_REQUIRED',physicalAcceptance:false,hardwareCommand:false,benchProtocolExported:false};
}
if(process.argv[1]&&import.meta.url===pathToFileURL(process.argv[1]).href){const [,,input,output]=process.argv;if(!input||!output)throw Error('Usage: node multi-axis.mjs measured.json report.json');const v=JSON.parse(readFileSync(input,'utf8'));writeFileSync(output,JSON.stringify(assess(v.rows,v.limits),null,2)+'\n',{flag:'wx'});}
