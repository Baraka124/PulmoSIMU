// HIL-02 passive sensor carrier concept, millimetres, no actuator.
// Separate draft parts, hardware envelopes are assumptions until measured.
part="tray"; // [tray,lever_sensor,coupon,shaft_insert]
$fn=48;
module tray(){
 difference(){
  union(){
   translate([-11,-17,0]) cube([22,34,3]);
   for(x=[-8,8],y=[-13,13])translate([x,y,3])cylinder(d=6,h=4);
  }
  for(x=[-8,8],y=[-13,13])translate([x,y,-1])cylinder(d=2.7,h=10);
  translate([-5,-8,-1])cube([10,16,5]);
 }
}
module lever_sensor(){
 difference(){
  union(){
   translate([-10,-12,0])cube([20,24,3]);
   translate([-10,-12,3])cube([4,24,13]);
   translate([6,-12,3])cube([4,24,13]);
  }
  for(x=[-8,8])translate([x,0,11])rotate([0,90,0])cylinder(d=4.2,h=6,center=true);
  for(y=[-8,8])translate([0,y,-1])cylinder(d=3.4,h=5);
 }
}
module shaft_insert(){
 difference(){
  union(){cylinder(d=7.8,h=12);cylinder(d=12,h=2);}
  translate([0,0,-1])cylinder(d=4.4,h=15);
 }
}
module coupon(){
 difference(){cube([48,20,5]);for(i=[0:3])translate([7+i*11,10,-1])cylinder(d=[2.7,3.4,4.2,4.4][i],h=7);}
}
if(part=="tray")tray();else if(part=="lever_sensor")lever_sensor();else if(part=="shaft_insert")shaft_insert();else if(part=="coupon")coupon();
