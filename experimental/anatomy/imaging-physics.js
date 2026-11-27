(function physics(root){
'use strict';
const check=(v,m)=>{if(!v)throw Error(m)},finite=Number.isFinite,range=(x,a,b)=>finite(x)&&x>=a&&x<=b,dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0),norm=a=>Math.hypot(...a),unit=a=>{const n=norm(a);check(n>1e-9,'Invalid direction');return a.map(x=>x/n);},clamp=(x,a,b)=>Math.min(b,Math.max(a,x)),apply=(m,p,translate=true)=>m.slice(0,3).map(r=>dot(r.slice(0,3),p)+(translate?r[3]:0));
function bound(study,seg,route){check(route?.schema==='pulmosimu.airway-route/1'&&route.ct_sha256===study?.source?.sha256&&route.mask_sha256===seg?.digest&&seg?.structure==='airway_lumen'&&seg.patch().ct_sha256===route.ct_sha256&&seg.patch().edits.length===0&&route.clinical_navigation_allowed!==true,'Exact unchanged source route required');}
function radiance({distance,cosine,reflectance=.55,roughness=.5,power=1,ambient=.015,exposure=1,vignette=1}){
 check(range(distance,0,100)&&range(cosine,0,1)&&range(reflectance,0,1)&&range(roughness,.1,1)&&range(power,.1,5)&&range(ambient,0,.1)&&range(exposure,.25,3)&&range(vignette,0,1),'Invalid optical parameters');
 const diffuse=reflectance/Math.PI*cosine,exponent=Math.max(1,2/(roughness*roughness)-2),specular=.04*(exponent+2)/(2*Math.PI)*Math.pow(cosine,exponent)*cosine;
 const irradiance=power/(1+(distance/5)**2),linear=exposure*vignette*(irradiance*(diffuse+specular)+ambient*reflectance);
 return {linear,encoded:Math.pow(1-Math.exp(-linear),1/2.2)};
}
function optical({study,seg,route,index=0,roll=0,fov=110,exposure=1,power=2,reflectance=.55,roughness=.5,width=160,height=120}){
 bound(study,seg,route);check(Number.isInteger(width)&&Number.isInteger(height)&&width>=16&&height>=16&&width*height<=32768&&range(fov,60,120)&&range(roll,-180,180),'Invalid optical camera');
 const U=root.PulmoOptics,cam=U.camera(route,index,roll),trace=U.tracer(study,seg),tan=Math.tan(fov*Math.PI/360),rgba=new Uint8ClampedArray(width*height*4),counts={WALL:0,RANGE:0,VOLUME_EXIT:0,BUDGET:0,ORIGIN_OUTSIDE:0};let sum=0;
 for(let y=0;y<height;y++)for(let x=0;x<width;x++){
  const u=(2*(x+.5)/width-1)*tan,v=(1-2*(y+.5)/height)*tan*height/width,d=unit(cam.forward.map((a,i)=>a+u*cam.right[i]+v*cam.up[i])),hit=trace(cam.origin,d,50),j=(x+y*width)*4;counts[hit.kind]++;
  if(hit.kind==='WALL'){const c=clamp(-dot(hit.normal,d),0,1),vignette=Math.pow(Math.max(0,dot(d,cam.forward)),4),r=radiance({distance:hit.distance,cosine:c,reflectance,roughness,power,exposure,vignette});sum+=r.linear;
   [1,.74,.68].forEach((color,k)=>rgba[j+k]=Math.round(255*Math.pow(1-Math.exp(-r.linear*color),1/2.2)));
  }else{rgba[j]=8;rgba[j+1]=13;rgba[j+2]=17;}rgba[j+3]=255;
 }
 check(!counts.ORIGIN_OUTSIDE,'Camera outside source lumen');
 return {rgba,width,height,report:{schema:'pulmosimu.optical-physics/1',ct_sha256:route.ct_sha256,mask_sha256:route.mask_sha256,index,roll_deg:roll,fov_deg:fov,counts,mean_wall_linear_response:counts.WALL?sum/counts.WALL:null,assumptions:{relative_light_power:power,diffuse_reflectance:reflectance,roughness,exposure,near_field_softening_mm:5,gamma:2.2,ambient:.015},model:'COAXIAL_DIFFUSE_SPECULAR_RELATIVE_EXPOSURE',absolute_radiometry:false,measured_tissue_optics:false,clinical_navigation_allowed:false,review:'PENDING'}};
}
function interfacePower(z1,z2){check(range(z1,.0001,10)&&range(z2,.0001,10),'Invalid acoustic impedance');const r=(z2-z1)/(z2+z1);return {reflection:r*r,transmission:1-r*r};}
function attenuationDb(alpha,f,distanceMm){check(range(alpha,0,30)&&range(f,1,30)&&range(distanceMm,0,100),'Invalid attenuation inputs');return 2*alpha*f*distanceMm/10;}
function rayInterfaces(study,seg,origin,direction,maxMm){
 const inv=study.spatial.ras_mm_to_voxel,d=study.source.dimensions,o=apply(inv,origin),v=apply(inv,direction,false),cell=o.map(x=>Math.floor(x+.5)),inside=p=>p.every((x,i)=>x>=0&&x<d[i]);
 check(inside(cell)&&seg.at(cell)===1,'Probe outside source lumen');const step=v.map(Math.sign),delta=v.map(x=>x?Math.abs(1/x):Infinity),next=v.map((x,i)=>x?(cell[i]+.5*step[i]-o[i])/x:Infinity),events=[];let material=1;
 for(let n=0;n<2048;n++){const t=Math.min(...next);if(!finite(t)||t>maxMm)return {events,termination:'RANGE'};for(let i=0;i<3;i++)if(Math.abs(next[i]-t)<1e-9){cell[i]+=step[i];next[i]+=delta[i];}
  if(!inside(cell))return {events,termination:'VOLUME_EXIT'};const m=seg.at(cell)?1:0;
  if(m!==material){events.push({distanceMm:Math.max(0,t),from:material,to:m});material=m;if(events.length>=12)return {events,termination:'INTERFACE_LIMIT'};}
 }return {events,termination:'TRAVERSAL_LIMIT'};
}
function echoes(events,{coupling,frequency,alpha,tissueZ,tissueSpeed}){
 const lumen=coupling==='air'?{z:.0004,c:343,alpha:1.5}:{z:1.48,c:1480,alpha:.002},tissue={z:tissueZ,c:tissueSpeed,alpha},out=[];
 let last=0,oneWayUs=0,lossDb=0,transmitted=1;
 for(const e of events){const a=e.from?lumen:tissue,b=e.to?lumen:tissue,span=e.distanceMm-last;check(span>=0,'Nonmonotone ray');oneWayUs+=span*1000/a.c;lossDb+=attenuationDb(a.alpha,frequency,span);const p=interfacePower(a.z,b.z);
  // Power transmission at each prior interface is traversed twice; use amplitude sqrt of power.
  const amplitude=transmitted*Math.sqrt(p.reflection)*Math.pow(10,-lossDb/20);
  out.push({physical_distance_mm:e.distanceMm,roundtrip_us:2*oneWayUs,display_depth_mm:oneWayUs*1.54,amplitude,reflection_power:p.reflection,from:e.from,to:e.to});
  transmitted*=p.transmission;last=e.distanceMm;
 }
 return out;
}
function acoustic({study,seg,route,index=0,roll=0,mode='radial',coupling='air',frequency=12,alpha=.5,tissueZ=1.63,tissueSpeed=1540,rangeMm=25,dynamicDb=55,tgcDb=8}){
 bound(study,seg,route);check(['radial','convex'].includes(mode)&&['air','ideal-fluid'].includes(coupling)&&range(roll,-180,180)&&range(frequency,3,20)&&range(alpha,.05,3)&&range(tissueZ,1.1,2.2)&&range(tissueSpeed,1400,1800)&&range(rangeMm,5,40)&&range(dynamicDb,30,80)&&range(tgcDb,0,30),'Invalid acoustic settings');
 const cam=root.PulmoOptics.camera(route,index,roll),rays=mode==='radial'?120:81,sector=Math.PI/3,lines=[],terms={RANGE:0,VOLUME_EXIT:0,INTERFACE_LIMIT:0,TRAVERSAL_LIMIT:0};let total=0,max=0;
 const settings={coupling,frequency,alpha,tissueZ,tissueSpeed};
 for(let i=0;i<rays;i++){const angle=mode==='radial'?2*Math.PI*i/rays:-sector/2+sector*i/(rays-1),direction=unit((mode==='radial'?cam.right:cam.forward).map((v,j)=>v*Math.cos(angle)+(mode==='radial'?cam.up:cam.right)[j]*Math.sin(angle)));
 // Convex is a point-sector approximation in a plane lateral to the shaft, not a forward looking transducer.
 const dir=mode==='convex'?unit(cam.right.map((v,j)=>v*Math.cos(angle)+cam.forward[j]*Math.sin(angle))):direction;
 const q=rayInterfaces(study,seg,cam.origin,dir,rangeMm);terms[q.termination]++;const e=echoes(q.events,settings);total+=e.length;for(const x of e)max=Math.max(max,x.amplitude);lines.push({angle_rad:angle,echoes:e});
 }
 // Scan conversion uses displayed depth with the conventional 1540 m/s assumption.
 // Pulse FWHM is an assumed two-cycle envelope, not a measured bandwidth.
 const size=256,rgba=new Uint8ClampedArray(size*size*4),cx=size/2,cy=mode==='radial'?size/2:14,R=mode==='radial'?size*.46:size*.88,fwhm=1540/1000/frequency,sigma=fwhm/2.355;
 for(let y=0;y<size;y++)for(let x=0;x<size;x++){const dx=x+.5-cx,dy=y+.5-cy,r=Math.hypot(dx,dy)/R*rangeMm,j=(x+y*size)*4;let angle=mode==='radial'?Math.atan2(dy,dx):Math.atan2(dx,dy),g=0;
 if(r<=rangeMm&&(mode==='radial'||Math.abs(angle)<=sector/2&&dy>=0)){
  if(mode==='radial'&&angle<0)angle+=2*Math.PI;
  const i=mode==='radial'?Math.round(angle/(2*Math.PI)*rays)%rays:clamp(Math.round((angle+sector/2)/sector*(rays-1)),0,rays-1);
  let amp=0;for(const e of lines[i].echoes)amp+=e.amplitude*Math.exp(-.5*((r-e.display_depth_mm)/sigma)**2);
  const db=20*Math.log10(Math.max(1e-12,amp))+tgcDb*r/rangeMm;g=Math.round(255*clamp((db+dynamicDb)/dynamicDb,0,1));
 }rgba[j]=rgba[j+1]=rgba[j+2]=g;rgba[j+3]=255;
 }
 return {rgba,width:size,height:size,report:{schema:'pulmosimu.acoustic-interface/1',ct_sha256:route.ct_sha256,mask_sha256:route.mask_sha256,index,roll_deg:roll,mode,range_mm:rangeMm,display_speed_m_s:1540,ray_count:rays,echo_count:total,max_relative_echo_amplitude:max,termination_counts:terms,assumptions:{coupling,frequency_MHz:frequency,tissue_impedance_MRayl:tissueZ,tissue_speed_m_s:tissueSpeed,tissue_attenuation_dB_cm_MHz:alpha,dynamic_range_dB:dynamicDb,time_gain_end_dB:tgcDb,pulse_fwhm_display_mm:fwhm},lines,model:mode==='radial'?'RADIAL_POINT_RAYS':'LATERAL_POINT_SECTOR_60_DEG',tissue_map_measured:false,refraction_solved:false,multiple_scattering_solved:false,transducer_calibrated:false,clinical_navigation_allowed:false,review:'PENDING'}};
}
root.PulmoImagingPhysics=Object.freeze({radiance,optical,interfacePower,attenuationDb,rayInterfaces,echoes,acoustic});
})(typeof window!=='undefined'?window:globalThis);
