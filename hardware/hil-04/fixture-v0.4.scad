// HIL04 draft passive desktop fixture. No motors, no powered brake.
// Component envelopes only; verify actual bearings, rollers, load cell,
// manual release, guards and retention before fabrication.
part="base"; // [base,guide,guard_coupon]
$fn=48;
module base(){
 difference(){
  cube([140,85,8],center=true);
  for(x=[-55,55],y=[-30,30])translate([x,y,-6])cylinder(d=5.5,h=12);
  for(x=[-35,0,35],y=[-18,18])translate([x,y,-6])cylinder(d=3.4,h=12);
 }
}
module guide(){
 difference(){
  union(){cube([24,32,5],center=true);translate([0,0,10])cube([24,12,20],center=true);}
  translate([0,0,14])rotate([0,90,0])cylinder(d=8,h=30,center=true);
  for(y=[-11,11])translate([0,y,-4])cylinder(d=3.4,h=10);
 }
}
module guard_coupon(){
 difference(){cube([40,26,4],center=true);for(x=[-14,14])translate([x,0,-3])cylinder(d=3.4,h=6);}
}
if(part=="base")base();else if(part=="guide")guide();else if(part=="guard_coupon")guard_coupon();
