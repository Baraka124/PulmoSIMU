// PulmoSIMU HIL-01 preliminary nonclinical training handle.
// Millimetres. Parametric concept, not manufacturing-approved.
// Render each half separately with part = "left" / "right".
// No powered movement; commercial tube and real hardware selected later.
part="assembly"; // [assembly,left,right,lever,coupon]
$fn=48;
L=166; W=62; D=54; wall=3;
screw_d=3.4; port_d=8; pivot_d=4.2;
assert(wall>=2 && L>100 && W>2*wall && D>2*wall);
module outline(inset=0){
 hull(){
  for(z=[-L/2+20,L/2-20])
   translate([0,0,z]) scale([(W/2-inset)/(20-inset),(D/2-inset)/(20-inset),1])
    sphere(r=20-inset);
 }
}
module shell(){
 difference(){
  outline();
  outline(wall);
  translate([0,0,-L/2-8]) cylinder(d=port_d,h=30);
  translate([0,0,L/2-18]) cylinder(d=12,h=30);
  translate([-7,-D/2-2,25]) cube([14,15,32]);
 }
}
module half(side){
 difference(){
  intersection(){
   union(){
    shell();
    // Trim transverse screw bosses to outer contour so they join the shell.
    intersection(){
     outline();
     union(){for(z=[-50,50],y=[-15,15])
      translate([0,y,z]) rotate([0,90,0]) cylinder(d=9,h=W+4,center=true);}
    }
   }
   translate([side<0?-W/2-2:0,-D/2-4,-L/2-4]) cube([W/2+2,D+8,L+8]);
  }
  for(z=[-50,50],y=[-15,15])
   translate([0,y,z]) rotate([0,90,0]) cylinder(d=screw_d,h=W+10,center=true);
 }
}
module lever(){
 difference(){
  hull(){cylinder(d=12,h=6,center=true);translate([0,24,0])cylinder(d=17,h=6,center=true);}
  cylinder(d=pivot_d,h=10,center=true);
 }
}
module coupon(){
 difference(){cube([42,22,5]);for(i=[0:2])translate([9+i*12,11,-1])cylinder(d=[3.4,4.2,8][i],h=8);}
}
if(part=="left") half(-1);
else if(part=="right") half(1);
else if(part=="lever") lever();
else if(part=="coupon") coupon();
else {color([.20,.26,.30])half(-1);color([.36,.43,.47])half(1);}
