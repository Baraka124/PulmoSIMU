// HIL03 draft passive angle reference jig. Not metrology certified.
// Independent calibrated angle reference required, not just these printed marks.
$fn=72;
difference(){
 union(){cube([100,60,5],center=true);translate([0,0,2.5])cylinder(d=10,h=12);}
 translate([0,0,-4])cylinder(d=4.2,h=22);
 for(a=[-60:15:60])rotate([0,0,a])translate([0,24,-4])cylinder(d=2.2,h=8);
}
