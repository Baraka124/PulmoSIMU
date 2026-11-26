(function optics(root){
'use strict';
const check=(v,m)=>{if(!v)throw Error(m)};
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0),sub=(a,b)=>a.map((x,i)=>x-b[i]),norm=a=>Math.hypot(...a),unit=a=>{const n=norm(a);check(n>1e-9,'Degenerate camera axis');return a.map(x=>x/n);};
const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
const apply=(m,p)=>m.slice(0,3).map(r=>dot(r.slice(0,3),p)+r[3]),linear=(m,p)=>m.slice(0,3).map(r=>dot(r.slice(0,3),p));
function camera(route,k,roll=0){
 const p=route.points_ras_mm;
 check(Array.isArray(p)&&p.length>=2&&p.length<=12000&&Number.isInteger(k)&&k>=0&&k<p.length&&Number.isFinite(roll),'Invalid route frame');
 let up,forward,right;
 for(let i=0;i<=k;i++){
  forward=unit(i===p.length-1?sub(p[i],p[i-1]):sub(p[i+1],p[i]));
  const seed=up??(Math.abs(forward[2])<.9?[0,0,1]:[0,1,0]);
  let projected=seed.map((x,j)=>x-dot(seed,forward)*forward[j]);
  if(norm(projected)<1e-6){const fallback=Math.abs(forward[2])<.9?[0,0,1]:[0,1,0];projected=fallback.map((x,j)=>x-dot(fallback,forward)*forward[j]);}
  up=unit(projected);right=unit(cross(forward,up));up=unit(cross(right,forward));
 }
 const r=roll*Math.PI/180,c=Math.cos(r),s=Math.sin(r);
 return {origin:p[k].slice(),forward,right:right.map((x,i)=>c*x+s*up[i]),up:up.map((x,i)=>c*x-s*right[i])};
}
function tracer(study,seg){
 check(study?.schema==='pulmosimu.study/1'&&seg?.structure==='airway_lumen'&&seg.patch().ct_sha256===study.source.sha256&&seg.patch().edits.length===0,'Unchanged exact airway mask required');
 const inv=study.spatial.ras_mm_to_voxel,d=study.source.dimensions;
 const inside=p=>p.every((x,i)=>x>=0&&x<d[i]);
 return (origin,dir,maxMm)=>{
  const o=apply(inv,origin),v=linear(inv,dir),cell=o.map(x=>Math.floor(x+.5));
  if(!inside(cell)||!seg.at(cell))return {kind:'ORIGIN_OUTSIDE'};
  const step=v.map(Math.sign),delta=v.map(x=>x?Math.abs(1/x):Infinity),next=v.map((x,i)=>x?(cell[i]+.5*step[i]-o[i])/x:Infinity);
  for(let n=0;n<4096;n++){
   const t=Math.min(...next);if(!Number.isFinite(t)||t>maxMm)return {kind:'RANGE'};
   const axes=[0,1,2].filter(i=>Math.abs(next[i]-t)<1e-9);
   for(const ax of axes){cell[ax]+=step[ax];next[ax]+=delta[ax];}
   if(!inside(cell))return {kind:'VOLUME_EXIT'};
   if(!seg.at(cell)){
    const normal=unit([0,1,2].map(j=>axes.reduce((s,i)=>s-step[i]*inv[i][j],0)));
    return {kind:'WALL',distance:Math.max(0,t),normal,position:origin.map((x,i)=>x+t*dir[i])};
   }
  }return {kind:'BUDGET'};
 };
}
function render({study,seg,route,index=0,width=160,height=120,fov=110,roll=0,exposure=1,maxMm=50}){
 check(route?.schema==='pulmosimu.airway-route/1'&&route.ct_sha256===study?.source?.sha256&&route.mask_sha256===seg?.digest,'Exact J route required');
 check(Number.isInteger(width)&&Number.isInteger(height)&&width>=16&&height>=16&&width*height<=32768&&Number.isFinite(fov)&&fov>=60&&fov<=120&&Number.isFinite(exposure)&&exposure>=.25&&exposure<=3&&Number.isFinite(maxMm)&&maxMm>=5&&maxMm<=60,'Invalid optical settings');
 const cam=camera(route,index,roll),trace=tracer(study,seg),tan=Math.tan(fov*Math.PI/360),rgba=new Uint8ClampedArray(width*height*4),depth=new Float32Array(width*height).fill(-1);
 const counts={WALL:0,RANGE:0,VOLUME_EXIT:0,BUDGET:0,ORIGIN_OUTSIDE:0};
 for(let y=0;y<height;y++)for(let x=0;x<width;x++){
  const u=(2*(x+.5)/width-1)*tan,v=(1-2*(y+.5)/height)*tan*height/width,dir=unit(cam.forward.map((a,i)=>a+u*cam.right[i]+v*cam.up[i]));
  const h=trace(cam.origin,dir,maxMm),i=x+y*width,j=i*4;counts[h.kind]++;
  if(h.kind==='WALL'){
   depth[i]=h.distance;const c=Math.max(0,-dot(h.normal,dir)),light=Math.min(1,exposure*(.08+.92*c/(1+(h.distance/9)**2))),vignette=1-.22*Math.min(1,(u*u+v*v)/(tan*tan));
   // Assumed diffuse material; no CT-derived mucosal color.
   const intensity=Math.pow(light*vignette,1/2.2);
   rgba[j]=Math.round(194*intensity);rgba[j+1]=Math.round(135*intensity);rgba[j+2]=Math.round(122*intensity);
  }else{rgba[j]=8;rgba[j+1]=13;rgba[j+2]=17;}
  rgba[j+3]=255;
 }
 check(counts.ORIGIN_OUTSIDE===0,'Camera outside lumen');
 return {width,height,rgba,depth,counts,camera:cam,fov,index,exposure,maxMm,ct_sha256:route.ct_sha256,mask_sha256:route.mask_sha256,root_node:route.root_node,target_node:route.target_node,material:'ASSUMED_DIFFUSE',clinical_navigation_allowed:false};
}
function splatImport(study,seg,doc){
 check(doc?.schema==='pulmosimu.optical-points/1'&&doc.ct_sha256===study.source.sha256&&doc.mask_sha256===seg.digest&&doc.frame==='RAS_mm'&&doc.review==='PENDING','Optical point identity/frame mismatch');
 check(typeof doc.source_sha256==='string'&&/^[a-f0-9]{64}$/.test(doc.source_sha256)&&Array.isArray(doc.points)&&doc.points.length<=5000,'Invalid optical point provenance or count');
 const inv=study.spatial.ras_mm_to_voxel,d=study.source.dimensions;
 return doc.points.map(p=>{
  check(Array.isArray(p)&&p.length===7&&p.every(Number.isFinite)&&p[3]>=.05&&p[3]<=2&&p.slice(4).every(v=>v>=0&&v<=255),'Invalid optical point');
  const q=apply(inv,p.slice(0,3));check(q.every((x,i)=>x>=-.5&&x<=d[i]-.5),'Optical point outside CT volume');
  return p.slice();
 });
}
function overlay(frame,points){
 const {width:w,height:h,camera:c,depth}=frame,tan=Math.tan(frame.fov*Math.PI/360),out=new Uint8ClampedArray(frame.rgba);let accepted=0;
 const ordered=points.map(p=>({p,v:sub(p.slice(0,3),c.origin)})).map(x=>({...x,z:dot(x.v,c.forward)})).filter(x=>x.z>.1).sort((a,b)=>b.z-a.z);
 for(const {p,v,z} of ordered){
  const x=(dot(v,c.right)/(z*tan)+1)*w/2-.5,y=(1-dot(v,c.up)/(z*tan*h/w))*h/2-.5,range=norm(v);
  const radius=Math.min(8,Math.max(.6,p[3]*w/(2*z*tan)));
  if(x<-radius*2||x>w+radius*2||y<-radius*2||y>h+radius*2)continue;
  let drawn=false;
  for(let j=Math.max(0,Math.floor(y-2*radius));j<=Math.min(h-1,Math.ceil(y+2*radius));j++)for(let i=Math.max(0,Math.floor(x-2*radius));i<=Math.min(w-1,Math.ceil(x+2*radius));i++){
   const k=i+w*j;if(depth[k]<0||Math.abs(depth[k]-range)>Math.max(1,2*p[3]))continue;
   const a=.45*Math.exp(-((i-x)**2+(j-y)**2)/(2*radius*radius));for(let c=0;c<3;c++)out[k*4+c]=Math.round(out[k*4+c]*(1-a)+p[4+c]*a);drawn=true;
  }if(drawn)accepted++;
 }return {rgba:out,accepted};
}
function compare(frame,ref){
 check(ref.length===frame.rgba.length,'Image size mismatch');let n=0,sum=0,abs=0;
 for(let i=0;i<frame.depth.length;i++){if(frame.depth[i]<0||ref[i*4+3]!==255)continue;for(let c=0;c<3;c++){const d=frame.rgba[4*i+c]-ref[4*i+c];sum+=d*d;abs+=Math.abs(d);n++;}}
 check(n>0,'No comparable opaque wall pixels');const mse=sum/n;return {wall_pixels:n/3,mae:abs/n,rmse:Math.sqrt(mse),psnr_db:mse===0?null:10*Math.log10(255**2/mse),identical:mse===0,registration_verified:false};
}
root.PulmoOptics=Object.freeze({camera,tracer,render,splatImport,overlay,compare});
})(typeof window!=='undefined'?window:globalThis);
