/**
 * PulmoSIMU R5 — Exploratory Therapeutic Laboratory.
 * The intervention model describes pathway-to-mechanism coupling, not real
 * efficacy, clinical eligibility, PK/PD, drug concentration or patient response.
 * Pure, deterministic; zero exposure leaves the same baseline intact.
 */
import {calculate} from './mechanistic-lung.js';
import {createDiseaseExperiment} from './disease-engine.js';

const clamp=(v)=>Math.max(0,Math.min(1,v));
const MECHANISMS=['bronchoconstriction','inflammation','mucus','remodeling','emphysema','infection','tumor'];
export const PHENOTYPES=Object.freeze({
 unspecified:{label:'Unspecified',signals:{type2:0.5,eosinophilic:0.5,allergic:0.5},description:'No phenotype evidence supplied'},
 type2_high:{label:'Type 2-high (illustrative)',signals:{type2:1,eosinophilic:0.85,allergic:0.70},description:'Demonstration of stronger type 2 signalling'},
 eosinophilic:{label:'Eosinophilic (illustrative)',signals:{type2:0.85,eosinophilic:1,allergic:0.35},description:'Demonstration of eosinophilic pathway sensitivity'},
 allergic:{label:'Allergic (illustrative)',signals:{type2:0.75,eosinophilic:0.45,allergic:1},description:'Demonstration of IgE-related pathway sensitivity'},
 type2_low:{label:'Type 2-low (illustrative)',signals:{type2:0.12,eosinophilic:0.10,allergic:0.10},description:'Demonstration of low type 2 signalling'}
});
export const INTERVENTIONS=Object.freeze({
 none:{label:'No intervention',class:'reference',pathway:'none',timeConstantDays:1,effects:{},phenotypeSignal:null},
 bronchodilator:{label:'Bronchodilator',class:'airway mechanics',pathway:'smooth muscle tone',timeConstantDays:0.025,effects:{bronchoconstriction:0.68},phenotypeSignal:null},
 ics:{label:'Inhaled corticosteroid',class:'anti-inflammatory',pathway:'inflammatory signaling',timeConstantDays:12,effects:{inflammation:0.38,mucus:0.13},phenotypeSignal:null},
 anti_ige:{label:'Anti-IgE pathway',class:'biologic',pathway:'IgE-related signaling',timeConstantDays:42,effects:{inflammation:0.30,mucus:0.08},phenotypeSignal:'allergic'},
 anti_il5:{label:'Anti-IL-5 pathway',class:'biologic',pathway:'eosinophil-associated signaling',timeConstantDays:35,effects:{inflammation:0.45},phenotypeSignal:'eosinophilic'},
 anti_il5r:{label:'Anti-IL-5R pathway',class:'biologic',pathway:'eosinophil-associated signaling',timeConstantDays:35,effects:{inflammation:0.48},phenotypeSignal:'eosinophilic'},
 anti_il4r:{label:'Anti-IL-4R pathway',class:'biologic',pathway:'IL-4/IL-13 signaling',timeConstantDays:28,effects:{inflammation:0.45,mucus:0.27},phenotypeSignal:'type2'},
 anti_tslp:{label:'Anti-TSLP pathway',class:'biologic',pathway:'upstream epithelial signaling',timeConstantDays:42,effects:{inflammation:0.32,mucus:0.09},phenotypeSignal:'type2'}
});
function checkInput({disease,focusRegion,severity,modifiers,intervention,phenotype,days,responseScale}){
 if(!Object.hasOwn(INTERVENTIONS,intervention))throw Error('Unknown intervention');
 if(!Object.hasOwn(PHENOTYPES,phenotype))throw Error('Unknown phenotype');
 if(!Number.isFinite(days)||days<0||days>3650)throw Error('Days must be 0..3650');
 if(!Number.isFinite(responseScale)||responseScale<0||responseScale>1)throw Error('Response factor must be 0..1');
}
export function responseProfile(intervention='none',phenotype='unspecified',days=0,responseScale=1){
 checkInput({intervention,phenotype,days,responseScale});
 const drug=INTERVENTIONS[intervention],signals=PHENOTYPES[phenotype].signals;
 const activation=1-Math.exp(-days/drug.timeConstantDays);
 // Relative phenotype sensitivity is strictly an assumed exploration parameter.
 const phenotypeSensitivity=drug.phenotypeSignal===null?1:0.15+0.85*signals[drug.phenotypeSignal];
 const multiplier=activation*phenotypeSensitivity*responseScale;
 return {intervention,phenotype,simulatedDays:days,activation,
   phenotypeSensitivity,responseScale,multiplier,
   mechanismReduction:Object.fromEntries(Object.entries(drug.effects).map(([k,v])=>[k,clamp(v*multiplier)])),
   pathway:drug.pathway,validated:false};
}
/** Use a no-treatment disease state as the counterfactual; avoid double-application
 * of legacy R1 THERAPIES. Copy each baseline mechanism to region modifiers after
 * attenuation, leaving immutable tumor occupancy and morphology unchanged.
 */
export function createTherapyExperiment({
 disease='asthma',severity=1,focusRegion='RUL',modifiers={},
 intervention='none',phenotype='unspecified',days=0,responseScale=1
}={}){
 checkInput({intervention,phenotype,days,responseScale});
 const untreated=createDiseaseExperiment({disease,severity,focusRegion,modifiers,therapy:'none',days:0});
 const before=calculate(untreated),response=responseProfile(intervention,phenotype,days,responseScale);
 const state={...untreated,timeDays:days,therapy:'none',
   baseline:Object.fromEntries(Object.keys(untreated.baseline).map(k=>[k,0])),
   regions:untreated.regions.map(region=>{
     const previous=before.regions.find(r=>r.id===region.id);
     const projected={};
     for(const mechanism of MECHANISMS){
       if(mechanism==='tumor')continue;
       projected[mechanism]=clamp(previous.mechanisms[mechanism]*
         (1-(response.mechanismReduction[mechanism]||0)));
     }
     return {...region,modifiers:projected};
   }),
   therapyExperiment:{intervention,phenotype,days,responseScale,pathway:response.pathway,
     model:'illustrative pathway attenuation',validated:false}
 };
 const after=calculate(state);
 return {untreated,treated:state,before,after,response,comparison:compareSnapshots(before,after),
   provenance:{validated:false,clinicalUse:false,model:'hypothesis testing only; invented exploratory parameters'}};
}
export function compareSnapshots(before,after){
 const metrics=['inflammation','bronchoconstriction','mucus','infection','tumor','emphysema'];
 const weighted=(s,k)=>s.regions.reduce((total,r)=>total+r.weight*r.mechanisms[k],0);
 const mechanisms=Object.fromEntries(metrics.map(k=>{
  const baseline=weighted(before,k),treated=weighted(after,k);
  return [k,{baseline,treated,delta:treated-baseline}];
 }));
 return {mechanisms,ventilationIndex:{baseline:before.summary.overallVentilationIndex,
    treated:after.summary.overallVentilationIndex,
    delta:after.summary.overallVentilationIndex-before.summary.overallVentilationIndex},
   resistanceIndex:{baseline:before.summary.meanResistanceIndex,
    treated:after.summary.meanResistanceIndex,
    delta:after.summary.meanResistanceIndex-before.summary.meanResistanceIndex},
   regional:before.regions.map(r=>{const t=after.regions.find(x=>x.id===r.id);
    return {id:r.id,baselineResistance:r.resistanceIndex,
      treatedResistance:t.resistanceIndex,
      deltaResistance:t.resistanceIndex-r.resistanceIndex}})
 };
}
