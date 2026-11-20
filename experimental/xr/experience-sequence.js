/**
 * PulmoSIMU XR — first-use comfort sequence.
 * No automated headset locomotion, clinical actions or invasive animation.
 * Simple deterministic state that can be browser- and unit-tested.
 */
export const STEPS=Object.freeze([
 Object.freeze({title:'Welcome to the respiratory lab',text:'You are seated in a spatial laboratory. The model is in front of you; your head stays free. Use either controller to point at NEXT.',hint:'START HERE  ·  NO WALKING REQUIRED'}),
 Object.freeze({title:'Hold the anatomy',text:'Squeeze and hold either SIDE GRIP (the button under your middle finger). Gently move your hand: the lungs follow. Release to leave them there.',hint:'SIDE GRIP  ·  PICK UP  ·  RELEASE'}),
 Object.freeze({title:'Explore from every angle',text:'Squeeze both SIDE GRIPS and move your hands apart to enlarge the reference lungs. Rotate your hands to turn them. Release both when comfortable.',hint:'BOTH GRIPS  ·  ROTATE  ·  RESIZE'}),
 Object.freeze({title:'Discover the airways',text:'Aim at a lung region with a controller and press the INDEX TRIGGER to select it. Choose ISOLATE LOBE for a clear view of that structure.',hint:'INDEX TRIGGER  ·  SELECT'}),
 Object.freeze({title:'Follow a bronchial route',text:'Choose AIRWAY GUIDE. The LEFT THUMBSTICK moves a highlighted reference-route marker; your viewpoint stays still. RECENTER brings everything back.',hint:'AIRWAY GUIDE  ·  LEFT STICK  ·  RECENTER'})
]);
export function createWelcomeSequence(){
 let index=0,active=true;
 return Object.freeze({
  get:()=>Object.freeze({active,index,total:STEPS.length,...(active?STEPS[index]:{})}),
  next:()=>{if(!active)return false; if(index>=STEPS.length-1){active=false;return false}index++;return true},
  previous:()=>{if(!active)return false;index=Math.max(0,index-1);return true},
  skip:()=>{active=false},
  restart:()=>{index=0;active=true}
 });
}
