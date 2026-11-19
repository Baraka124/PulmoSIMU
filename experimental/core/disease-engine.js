/**
 * PulmoSIMU R4 — spatial disease mechanisms.
 * Distinct disease mechanisms are mapped onto the canonical R1 regional
 * state and R3 network. This module contains NO clinical prediction model.
 * All coefficients and anatomical heterogeneity are research placeholders.
 */
import {REGIONS,createState,withRegionModifier,withTumor,withTherapy,advance,calculate} from './mechanistic-lung.js';

const bound=x=>Math.max(0,Math.min(1,Number.isFinite(x)?x:0));
const mechanisms=['bronchoconstriction','inflammation','mucus','remodeling','emphysema','infection'];
export const DISEASE_LIBRARY=Object.freeze({
 healthy:{label:'Reference',description:'Illustrative resting respiratory state',distribution:'diffuse',dominant:['bronchoconstriction']},
 asthma:{label:'Asthma',description:'Inflammation, airway smooth muscle constriction, mucus and remodeling',distribution:'diffuse',dominant:['inflammation','bronchoconstriction','mucus']},
 copd:{label:'COPD',description:'Emphysema, airway remodeling and chronic bronchitic burden',distribution:'basal',dominant:['emphysema','remodeling','mucus']},
 bronchiectasis:{label:'Bronchiectasis',description:'Regional mucus retention, infection and inflammatory injury',distribution:'focal',dominant:['mucus','infection','inflammation']},
 lung_cancer:{label:'Lung cancer',description:'Local airway occupancy with obstruction-dependent regional effects',distribution:'focal',dominant:['tumor']}
});
/** Pure nonclinical spatial weighting. Lobar weights are arbitrary—not prevalence maps. */
const shape={
 diffuse:{RUL:0.15,RML:0.10,RLL:0.18,LUL:0.12,LLL:0.16},
 basal:{RUL:0.05,RML:0.10,RLL:0.34,LUL:0.05,LLL:0.29},
 focal:{RUL:0,RML:0,RLL:0,LUL:0,LLL:0}
};
export function createDiseaseExperiment({
 disease='healthy',severity=1,focusRegion='RUL',distribution,
 modifiers={},therapy='none',days=0
}={}){
 if(!DISEASE_LIBRARY[disease])throw Error('Unknown disease');
 if(!REGIONS.some(r=>r.id===focusRegion))throw Error('Unknown focus region');
 if(!Number.isFinite(severity)||severity<0||severity>1)throw Error('Severity must be 0..1');
 if(!Number.isFinite(days)||days<0)throw Error('Days must be nonnegative');
 if(!['none','diffuse','basal','focal',undefined].includes(distribution))throw Error('Unknown distribution');
 for(const [name,value] of Object.entries(modifiers)){
  if(!mechanisms.includes(name))throw Error('Invalid mechanism '+name);
  if(!Number.isFinite(value)||value<0||value>1)throw Error('Mechanism adjustment must be 0..1');
 }
 const diseaseMode=DISEASE_LIBRARY[disease];
 const pattern=distribution&&distribution!=='none'?distribution:diseaseMode.distribution;
 let state=createState(disease,severity);
 // Focal tumor in createState() defaults to RUL; relocate rather than duplicate.
 if(disease==='lung_cancer'){
  state=withTumor(state,'RUL',0);
  state=withTumor(state,focusRegion,0.7*severity);
 }
 // Disease spatial patterns modify the pre-existing mechanism strengths.
 for(const region of REGIONS){
  const regionFactor=pattern==='focal'?(region.id===focusRegion?1:0):shape[pattern][region.id];
  for(const mechanism of mechanisms){
   const baseline=state.baseline[mechanism]||0;
   // Differential burden is expressed through a positive regional increment;
   // focal non-tumor disease is attenuated outside the focus via a global-zero base.
   if(pattern==='focal'&&disease==='bronchiectasis'){
    const local=region.id===focusRegion?baseline:baseline*0.12;
    const target=bound(local+(modifiers[mechanism]||0)*(region.id===focusRegion?1:0));
    state={...state,baseline:{...state.baseline,[mechanism]:0}};
    state=withRegionModifier(state,region.id,mechanism,target);
   }else{
    const localIncrement=bound(baseline*regionFactor*0.35+
      (modifiers[mechanism]||0)*(region.id===focusRegion?1:0));
    state=withRegionModifier(state,region.id,mechanism,localIncrement);
   }
  }
 }
 state=withTherapy(state,therapy);
 state=advance(state,days);
 return {...state,diseaseExperiment:{
  version:'r4.0',distribution:pattern,focusRegion,
  modifiers:{...modifiers},mechanisms:[...diseaseMode.dominant],
  validation:'Illustrative spatial mechanism distributions only'
 }};
}
export function diseaseReadout(state){
 const result=calculate(state);
 const total=(key)=>result.regions.reduce((sum,r)=>sum+r.weight*r.mechanisms[key],0);
 const mechanismBurden=Object.fromEntries(
  [...mechanisms,'tumor'].map(k=>[k,total(k)])
 );
 const location=result.regions.find(r=>r.id===state.diseaseExperiment?.focusRegion);
 return {disease:state.scenario,focusRegion:state.diseaseExperiment?.focusRegion,
  distribution:state.diseaseExperiment?.distribution,
  mechanismBurden,focusMechanisms:location?.mechanisms,
  regional:result.regions.map(r=>({
   id:r.id,
   obstructionIndex:1-r.radiusIndex,
   inflammation:r.mechanisms.inflammation,
   mucus:r.mechanisms.mucus,
   infection:r.mechanisms.infection,
   emphysema:r.mechanisms.emphysema,
   tumor:r.mechanisms.tumor
  })),
  validated:false};
}
