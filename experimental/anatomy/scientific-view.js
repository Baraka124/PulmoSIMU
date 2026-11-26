(function scientific(root){
'use strict';
const finite=Number.isFinite,v3=p=>Array.isArray(p)&&p.length===3&&p.every(finite),axes={axial:[0,1,2],coronal:[0,2,1],sagittal:[1,2,0]};
function projection(plane,dims,w,h,cross){
 if(!axes[plane]||!v3(dims)||dims.some(n=>n<2)||!v3(cross)||!finite(w)||!finite(h)||w<2||h<2)throw Error('Invalid display grid');
 const [x,y,z]=axes[plane],flip=plane!=='axial',center=cross[z];
 function pixel(p){return [p[x]+.5,(flip?(1-p[y]/(dims[y]-1))*(h-1):p[y])+.5];}
 function voxel(u,v){const p=cross.slice();p[x]=u;p[y]=flip?(1-v/(h-1))*(dims[y]-1):v;return p;}
 function clip(a,b){
  if(!v3(a)||!v3(b))return null;
  // Clip to the displayed single voxel slab, then to actual in-plane support.
  let lo=0,hi=1;
  for(let k=0;k<3;k++){const min=k===z?center-.5:-.5,max=k===z?center+.5:dims[k]-.5,d=b[k]-a[k];
   if(Math.abs(d)<1e-12){if(a[k]<min||a[k]>max)return null;continue;}
   const m=(min-a[k])/d,n=(max-a[k])/d;lo=Math.max(lo,Math.min(m,n));hi=Math.min(hi,Math.max(m,n));if(lo>hi)return null;}
  const at=t=>pixel(a.map((v,i)=>v+(b[i]-v)*t));return [at(lo),at(hi)];
 }
 return {pixel,voxel,clip,axis:z,horizontalAxis:x,center,plane};
}
function affine(m,p){if(!v3(p)||!Array.isArray(m)||m.length!==4||m.some(r=>!Array.isArray(r)||r.length!==4||!r.every(finite)))throw Error('Invalid transform');return m.slice(0,3).map(r=>r[0]*p[0]+r[1]*p[1]+r[2]*p[2]+r[3]);}
function path(p,points){const out=[];if(!Array.isArray(points)||points.length>12000)return out;for(let i=1;i<points.length;i++){const line=p.clip(points[i-1],points[i]);if(line)out.push(line);}return out;}
function sampleField({project,w,h,volume,inverse,field,phase,grid=7}){
 if(!field||!finite(phase)||phase<0||phase>1||!Number.isInteger(grid)||grid<2||grid>10)return [];
 const out=[];for(let j=1;j<=grid;j++)for(let i=1;i<=grid;i++){
  const v=project.voxel((w-1)*i/(grid+1),(h-1)*j/(grid+1)),ras=volume.ras(...v);
  try{const q=field.map(ras,phase);if(!q.in_moving_volume)continue;const target=affine(inverse,q.position_ras_mm),a=project.pixel(v),b=project.pixel(target);if(![...a,...b,q.jacobian,...q.displacement_mm].every(finite)||q.jacobian<=0)continue;out.push({a,b,jacobian:q.jacobian,magnitudeMm:Math.hypot(...q.displacement_mm),outOfPlaneVoxel:target[project.axis]-v[project.axis]});}catch{}
 }return out;
}
function scaleBar(spacing,w){if(!finite(spacing)||spacing<=0||!finite(w)||w<20)return null;const max=w*.23*spacing,exponent=Math.pow(10,Math.floor(Math.log10(max))),mm=[5,2,1].map(x=>x*exponent).find(x=>x<=max)||exponent;return {mm,pixels:mm/spacing};}
function draw(canvas,{plane,volume,cross,route=null,shaft=null,field=null,phase=0,inverse=null,mode='off',scale=true,routeVisible=true,shaftVisible=true}){
 const c=canvas.getContext('2d'),w=canvas.width,h=canvas.height;c.clearRect(0,0,w,h);if(!volume)return {samples:0};
 const p=projection(plane,volume.dims,w,h,cross),lw=Math.max(1,w/420),radius=Math.max(2,w/180);
 function lines(points,color,dashed){c.save();c.strokeStyle=color;c.fillStyle=color;c.lineWidth=lw*1.8;c.setLineDash(dashed?[lw*4,lw*2]:[]);for(const [a,b] of path(p,points)){if(Math.hypot(a[0]-b[0],a[1]-b[1])<.8){c.beginPath();c.arc(...a,radius,0,2*Math.PI);c.fill();}else{c.beginPath();c.moveTo(...a);c.lineTo(...b);c.stroke();}}c.restore();}
 const samples=mode!=='off'&&inverse?sampleField({project:p,w,h,volume,inverse,field,phase}):[];
 c.save();c.lineWidth=lw;c.globalAlpha=.87;
 for(const q of samples){
  if(mode==='jacobian'){const log=Math.log(q.jacobian),v=Math.min(1,Math.abs(log)/Math.log(2));c.fillStyle=Math.abs(log)<1e-6?'#a9b8c0':log<0?'#9bc7e2':'#efb986';c.globalAlpha=.24+.66*v;c.fillRect(q.a[0]-radius*1.4,q.a[1]-radius*1.4,radius*2.8,radius*2.8);continue;}
  c.strokeStyle='#d2b0e2';c.fillStyle='#d2b0e2';const dx=q.b[0]-q.a[0],dy=q.b[1]-q.a[1],l=Math.hypot(dx,dy);
  if(l<1){c.beginPath();c.arc(...q.a,radius*.65,0,2*Math.PI);c.fill();continue;}
  const factor=Math.min(1,w*.18/l),b=[q.a[0]+dx*factor,q.a[1]+dy*factor],angle=Math.atan2(dy,dx),s=Math.max(3,lw*3);
  c.beginPath();c.moveTo(...q.a);c.lineTo(...b);c.lineTo(b[0]-s*Math.cos(angle-.5),b[1]-s*Math.sin(angle-.5));c.moveTo(...b);c.lineTo(b[0]-s*Math.cos(angle+.5),b[1]-s*Math.sin(angle+.5));c.stroke();
 }c.restore();
 if(routeVisible&&route)lines(route,'#77d1e3',false);
 if(shaftVisible&&shaft)lines(shaft,'#f0d29a',true);
 if(scale){const b=scaleBar(volume.spacing[p.horizontalAxis],w);if(b){const x=w*.06,y=h-Math.max(10,h*.07);c.save();c.fillStyle='rgba(5,9,12,.82)';c.fillRect(x-4,y-17,b.pixels+15,25);c.strokeStyle='#e7edf0';c.lineWidth=lw*1.5;c.beginPath();c.moveTo(x,y-3);c.lineTo(x,y+3);c.moveTo(x,y);c.lineTo(x+b.pixels,y);c.lineTo(x+b.pixels,y-3);c.lineTo(x+b.pixels,y+3);c.stroke();c.fillStyle='#e7edf0';c.font=Math.max(9,w/48)+'px system-ui';c.fillText(String(Number(b.mm.toPrecision(3)))+' mm',x,y-5);c.restore();}}
 return {samples:samples.length};
}
root.PulmoScientificView=Object.freeze({projection,affine,path,sampleField,scaleBar,draw});
})(typeof window!=='undefined'?window:globalThis);
