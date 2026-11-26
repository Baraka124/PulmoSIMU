/* UI01: world-space canvas typography and chrome; rendering only, no input state. */
(function(root){'use strict';
const C=Object.freeze({surface:'#171b1e',ink:'#edf0f2',muted:'#b1bac0',fine:'#85939c',edge:'#394148',accent:'#c1d6e0',hover:'#303f48',selected:'#26333c',warn:'#e5ba9b'});
function rounded(c,x,y,w,h,r=12){r=Math.min(r,w/2,h/2);c.beginPath();c.moveTo(x+r,y);c.arcTo(x+w,y,x+w,y+h,r);c.arcTo(x+w,y+h,x,y+h,r);c.arcTo(x,y+h,x,y,r);c.arcTo(x,y,x+w,y,r);c.closePath();}
function frame(c,w,h){c.clearRect(0,0,w,h);rounded(c,2,2,w-4,h-4,17);c.fillStyle=C.surface;c.fill();c.strokeStyle=C.edge;c.lineWidth=2;c.stroke();}
function line(c,x,y,w){c.fillStyle=C.edge;c.fillRect(x,y,w,1.5);}
function text(c,value,x,y,maxWidth,{size=28,min=21,weight=450,color=C.ink,align='left'}={}){let s=String(value??''),n=size;c.textAlign=align;c.textBaseline='alphabetic';c.fillStyle=color;const set=()=>c.font=weight+' '+n+'px system-ui, sans-serif';set();while(c.measureText(s).width>maxWidth&&n>min){n--;set();}if(c.measureText(s).width>maxWidth){while(s.length&&c.measureText(s+'…').width>maxWidth)s=s.slice(0,-1);s+='…';}c.fillText(s,x,y);c.textAlign='left';return s;}
const preserve=new Set(['VR','MR','CT','MPR','R3','XR','MM','FOV','QA','L','R','H','Y','X','ID','FPS','PBR']);
function label(s){return String(s).split(/(\s+|[\/·-])/).map(v=>preserve.has(v)?v==='MM'?'mm':v:/^[A-Z]+$/.test(v)?v.toLowerCase():v).join('').replace(/^./,a=>a.toUpperCase());}
const icons={
 controls:'M3 6 H21 M3 12 H21 M3 18 H21 M8 3 V9 M16 9 V15 M10 15 V21',
 anatomy:'M12 3 V10 M12 7 L7 12 M12 7 L17 12 M7 5 C3 6 2 12 3 17 Q4 21 10 20 V11 M17 5 C21 6 22 12 21 17 Q20 21 14 20 V11',
 scope:'M5 3 H13 V7 H5 Z M9 7 V13 Q9 20 16 20 Q21 20 21 15 M18 15 H23',
 guide:'M4 3 H20 V21 H4 Z M8 8 H16 M8 12 H16 M8 16 H13',
 focus:'M3 9 V3 H9 M15 3 H21 V9 M21 15 V21 H15 M9 21 H3 V15 M12 9 V15 M9 12 H15',
 channel:'M4 4 L15 15 M12 18 L18 12 M15 15 L21 21 M3 7 L7 3',
 exit:'M10 4 H4 V20 H10 M9 12 H21 M17 8 L21 12 L17 16',
 minimize:'M5 12 H19',reading:'M4 18 L10 5 L16 18 M6 14 H14 M18 9 H23 M20.5 6.5 V11.5',
 settings:'M12 3 L14 6 L18 5 L19 9 L22 12 L19 15 L18 19 L14 18 L12 21 L10 18 L6 19 L5 15 L2 12 L5 9 L6 5 L10 6 Z M12 9 A3 3 0 1 0 12 15 A3 3 0 1 0 12 9',
 back:'M20 12 H4 M10 6 L4 12 L10 18',
 section:'M4 4 H20 V20 H4 Z M4 12 H20 M12 4 V20',
 position:'M12 2 V22 M2 12 H22 M9 5 L12 2 L15 5 M9 19 L12 22 L15 19 M5 9 L2 12 L5 15 M19 9 L22 12 L19 15',
 image:'M3 5 H21 V19 H3 Z M6 15 L10 10 L14 15 L17 12 L21 16',
 move:'M5 12 H19 M15 8 L19 12 L15 16',
 tool:'M4 20 L13 11 M13 11 L12 6 L16 3 L15 8 L19 9 L22 6 L21 11 L17 14 Z'
};
function iconKind(action,title=''){if(action==='dock-mode')return /anatomy/i.test(title)?'anatomy':'scope';if(/exit/.test(action))return 'exit';if(/collapse|guide-hide/.test(action))return 'minimize';if(/reading/.test(action))return 'reading';if(/recenter|reset|focus|^view$/.test(action))return 'focus';if(/guide|branch/.test(action))return 'guide';if(/channel/.test(action))return 'channel';if(/anatomy|inspect/.test(action))return 'anatomy';if(/procedure|scope|device/.test(action))return 'scope';if(/back/.test(action))return 'back';if(/section|ct-/.test(action))return 'section';if(/workspace|position|panel-/.test(action))return 'position';if(/radial|imaging|optics/.test(action))return 'image';if(/tool|instrument/.test(action))return 'tool';if(/station|settings/.test(action))return 'settings';if(/dock-open|more|precision/.test(action))return 'controls';return null;}
function symbol(c,kind,x,y,size,color){if(!icons[kind]||typeof Path2D==='undefined')return false;c.save();c.translate(x,y);c.scale(size/24,size/24);c.strokeStyle=color;c.lineWidth=1.65;c.lineCap='round';c.lineJoin='round';c.stroke(new Path2D(icons[kind]));c.restore();return true;}
function control(c,{x,y,w,h,title,hovered=false,selected=false,available=true,action='',physical=false}){const color=available?C.ink:C.fine;
 if(physical){c.fillStyle=available?'#242c31':'#1d2226';c.fillRect(x,y,w,h);}
 if(hovered&&available||selected){rounded(c,x+2,y+3,w-4,h-6,8);c.fillStyle=hovered?C.hover:C.selected;c.fill();}
 if(hovered&&available){c.fillStyle=C.accent;c.fillRect(x+3,y+11,3,h-22);}
 const navigation=/^(back|more|.*-open|view|precision|guide|tool)$/.test(action);
 const glyph=navigation?'›':action==='exit'?'↗':selected?'•':'';
 const kind=iconKind(action,title),s=Math.min(29,h*.49),showIcon=Boolean(kind&&w>=190&&h>=47);const left=showIcon&&symbol(c,kind,x+15,y+(h-s)/2,s,available?C.accent:C.fine)?s+26:16;
 text(c,label(title),x+left,y+h*.62,w-left-(glyph?39:14),{size:h>100?44:h<45?24:h<55?25:29,min:h>100?32:21,weight:hovered||selected?600:500,color});
 if(glyph)text(c,glyph,x+w-15,y+h*.63,32,{size:30,min:25,weight:500,color:available?C.accent:C.fine,align:'right'});
 // Visual bounds and hit target are exactly the same normalized rectangle.
}
function header(c,w,kicker,title){text(c,kicker,42,45,w*.60,{size:23,weight:560,color:C.muted});text(c,title,42,111,w-84,{size:49,min:38,weight:520});}
root.PulmoXRSpatialUI=Object.freeze({C,frame,line,text,label,control,header,iconKind,symbol});
})(typeof window!=='undefined'?window:globalThis);
