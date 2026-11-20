/**
 * PulmoSIMU XR-06 · Anatomical inspection state.
 *
 * References only the canonical SEGS-derived airway graph. No anatomical
 * dimensions are inferred from the procedural reference scene coordinates.
 * All section fractions express *display-local* position, not millimetres.
 */
const LOBES=new Set(['RUL','RML','RLL','LUL','LLL']);
const sectionClamp=x=>Math.max(-1,Math.min(1,x));
export function createAnatomicalWorkbench(graph,{initialRegion='RUL'}={}){
 if(!graph||!Array.isArray(graph.nodes)||typeof graph.regionOf!=='function'||
   typeof graph.get!=='function'||typeof graph.chain!=='function')throw Error('Invalid reference airway graph');
 const nodes=graph.nodes.filter(n=>n&&Number.isInteger(n.id));
 let region=LOBES.has(initialRegion)?initialRegion:'RUL';
 let branchId=null,sectionActive=false,sectionFraction=0;
 const regional=id=>nodes.filter(n=>graph.regionOf(n.id)===id);
 const bestBranch=id=>regional(id).filter(n=>n.children.length===0).sort((a,b)=>b.generation-a.generation||a.id-b.id)[0]?.id??regional(id)[0]?.id??null;
 branchId=bestBranch(region);
 function snapshot(){
   const branch=branchId===null?null:graph.get(branchId);
   return Object.freeze({
    regionId:region,branchId,branchName:branch?.name??null,
    generation:branch?.generation??null,
    source:'pulmosimu-airway-reference-r2',units:'uncalibrated-procedural-scene',
    calibrated:false,measurementsAllowed:false,
    branchCount:regional(region).length,
    pathIds:Object.freeze(branchId===null?[]:graph.chain(branchId).map(n=>n.id)),
    sectionActive,sectionFraction
   });
 }
 function selectRegion(id){
   if(!LOBES.has(id))return false;
   if(id===region)return true;
   region=id;branchId=bestBranch(id);return true;
 }
 function selectBranch(id){
   if(!Number.isInteger(id)||!graph.get(id))return false;
   const lobe=graph.regionOf(id);
   if(!LOBES.has(lobe))return false;
   region=lobe;branchId=id;return true;
 }
 function nextBranch(step=1){
   if(!Number.isInteger(step)||step===0)throw Error('Branch step must be nonzero integer');
   const list=regional(region);
   if(!list.length)return null;
   const index=list.findIndex(n=>n.id===branchId);
   branchId=list[((index<0?0:index)+Math.sign(step)+list.length)%list.length].id;
   return branchId;
 }
 function setSection(active){sectionActive=Boolean(active);return snapshot()}
 function moveSection(delta){
   if(!Number.isFinite(delta))throw Error('Invalid section movement');
   sectionFraction=sectionClamp(sectionFraction+delta);return sectionFraction;
 }
 function setSectionFraction(value){
   if(!Number.isFinite(value))throw Error('Invalid section fraction');
   sectionFraction=sectionClamp(value);return sectionFraction;
 }
 function resetSection(){sectionActive=false;sectionFraction=0;return snapshot()}
 return Object.freeze({snapshot,selectRegion,selectBranch,nextBranch,setSection,moveSection,setSectionFraction,resetSection});
}
export function sectionalLocalPoint({center,depth,fraction}){
 if(!Array.isArray(center)||center.length!==3||!center.every(Number.isFinite)||
   !Number.isFinite(depth)||depth<=0||!Number.isFinite(fraction))throw Error('Invalid source-local clipping coordinates');
 return [center[0],center[1],center[2]+sectionClamp(fraction)*depth/2];
}
