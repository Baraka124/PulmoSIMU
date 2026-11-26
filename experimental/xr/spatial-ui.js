/* UI01: world-space canvas typography and chrome; rendering only, no input state. */
(function(root){'use strict';
const C=Object.freeze({surface:'#171b1e',ink:'#edf0f2',muted:'#b1bac0',fine:'#85939c',edge:'#394148',accent:'#c1d6e0',hover:'#303f48',selected:'#26333c',warn:'#e5ba9b'});
function rounded(c,x,y,w,h,r=12){r=Math.min(r,w/2,h/2);c.beginPath();c.moveTo(x+r,y);c.arcTo(x+w,y,x+w,y+h,r);c.arcTo(x+w,y+h,x,y+h,r);c.arcTo(x,y+h,x,y,r);c.arcTo(x,y,x+w,y,r);c.closePath();}
function frame(c,w,h){c.clearRect(0,0,w,h);rounded(c,2,2,w-4,h-4,17);c.fillStyle=C.surface;c.fill();c.strokeStyle=C.edge;c.lineWidth=2;c.stroke();}
function line(c,x,y,w){c.fillStyle=C.edge;c.fillRect(x,y,w,1.5);}
function text(c,value,x,y,maxWidth,{size=28,min=21,weight=450,color=C.ink,align='left'}={}){let s=String(value??''),n=size;c.textAlign=align;c.textBaseline='alphabetic';c.fillStyle=color;const set=()=>c.font=weight+' '+n+'px system-ui, sans-serif';set();while(c.measureText(s).width>maxWidth&&n>min){n--;set();}if(c.measureText(s).width>maxWidth){while(s.length&&c.measureText(s+'…').width>maxWidth)s=s.slice(0,-1);s+='…';}c.fillText(s,x,y);c.textAlign='left';return s;}
const preserve=new Set(['VR','MR','CT','MPR','R3','XR','MM','FOV','QA','L','R','H','Y','X','ID','FPS','PBR']);
function label(s){return String(s).split(/(\s+|[\/·-])/).map(v=>preserve.has(v)?v==='MM'?'mm':v:/^[A-Z]+$/.test(v)?v.toLowerCase():v).join('').replace(/^./,a=>a.toUpperCase());}
function control(c,{x,y,w,h,title,hovered=false,selected=false,available=true,action='',physical=false}){const color=available?C.ink:C.fine;
 if(physical){c.fillStyle=available?'#242c31':'#1d2226';c.fillRect(x,y,w,h);}
 if(hovered&&available||selected){rounded(c,x+2,y+3,w-4,h-6,8);c.fillStyle=hovered?C.hover:C.selected;c.fill();}
 if(hovered&&available){c.fillStyle=C.accent;c.fillRect(x+3,y+11,3,h-22);}
 const navigation=/^(back|more|.*-open|view|precision|guide|tool)$/.test(action);
 const glyph=navigation?'›':action==='exit'?'↗':selected?'•':'';
 text(c,label(title),x+16,y+h*.62,w-(glyph?55:30),{size:h>100?44:h<55?25:29,min:h>100?32:21,weight:hovered||selected?600:500,color});
 if(glyph)text(c,glyph,x+w-15,y+h*.63,32,{size:30,min:25,weight:500,color:available?C.accent:C.fine,align:'right'});
 // Visual bounds and hit target are exactly the same normalized rectangle.
}
function header(c,w,kicker,title){text(c,kicker,42,45,w-84,{size:23,weight:560,color:C.muted});text(c,title,42,111,w-84,{size:49,min:38,weight:520});}
root.PulmoXRSpatialUI=Object.freeze({C,frame,line,text,label,control,header});
})(typeof window!=='undefined'?window:globalThis);
