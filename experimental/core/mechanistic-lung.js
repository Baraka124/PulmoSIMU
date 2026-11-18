/**
 * PulmoSIMU · Mechanistic Lung Foundation R1
 * Research prototype. Units are explicit where defined; illustrative parameters
 * are not calibrated for diagnosis, drug choice or individual response prediction.
 * Pure deterministic code; no DOM or renderer dependency.
 */
export const VERSION = 'r1.0.0';
export const REGIONS = Object.freeze([
  {id:'RUL',side:'R',label:'Right upper',weight:0.18,compliance:0.20,baselineRadius:1.0},
  {id:'RML',side:'R',label:'Right middle',weight:0.10,compliance:0.20,baselineRadius:1.0},
  {id:'RLL',side:'R',label:'Right lower',weight:0.27,compliance:0.23,baselineRadius:1.0},
  {id:'LUL',side:'L',label:'Left upper',weight:0.21,compliance:0.20,baselineRadius:1.0},
  {id:'LLL',side:'L',label:'Left lower',weight:0.24,compliance:0.23,baselineRadius:1.0},
]);
const clamp=(v,min=0,max=1)=>Math.max(min,Math.min(max,Number.isFinite(v)?v:min));
const mean=(a)=>a.reduce((s,x)=>s+x,0)/a.length;
export const SCENARIOS=Object.freeze({
  healthy:{bronchoconstriction:0.04,inflammation:0.05,mucus:0.05,remodeling:0.02,emphysema:0,infection:0,tumor:0},
  asthma:{bronchoconstriction:0.65,inflammation:0.70,mucus:0.42,remodeling:0.28,emphysema:0,infection:0,tumor:0},
  copd:{bronchoconstriction:0.28,inflammation:0.35,mucus:0.42,remodeling:0.50,emphysema:0.58,infection:0,tumor:0},
  bronchiectasis:{bronchoconstriction:0.13,inflammation:0.56,mucus:0.73,remodeling:0.36,emphysema:0,infection:0.64,tumor:0},
  lung_cancer:{bronchoconstriction:0.08,inflammation:0.20,mucus:0.16,remodeling:0.08,emphysema:0,infection:0,tumor:0.70}
});
/** Time constants in simulated days. Coefficients are exploratory, not PK/PD fits. */
export const THERAPIES=Object.freeze({
  none:{label:'No intervention',targets:{},tauDays:1},
  bronchodilator:{label:'Bronchodilator (illustrative)',targets:{bronchoconstriction:0.65},tauDays:0.02},
  ics:{label:'Inhaled corticosteroid (illustrative)',targets:{inflammation:0.38,mucus:0.12},tauDays:12},
  anti_il5:{label:'Anti-IL-5 pathway (illustrative)',targets:{inflammation:0.42},tauDays:35},
  anti_il4r:{label:'Anti-IL-4R pathway (illustrative)',targets:{inflammation:0.45,mucus:0.26},tauDays:28}
});
export function createState(scenario='healthy',severity=1){
  if(!SCENARIOS[scenario]) throw Error('Unknown scenario: '+scenario);
  const s=clamp(severity);
  return {
    schema:'pulmosimu-mechanistic-r1',scenario,severity:s,timeDays:0,therapy:'none',
    baseline:Object.fromEntries(Object.entries(SCENARIOS[scenario]).map(([k,v])=>[k,clamp(v*s)])),
    regions:REGIONS.map((r)=>({id:r.id,modifiers:{},tumorFraction:0})),
    provenance:{model:'illustrative reduced-order',validated:false,patientSpecific:false}
  };
}
export function withRegionModifier(state,regionId,mechanism,value){
  if(!Object.hasOwn(state.baseline,mechanism))throw Error('Unknown mechanism');
  if(!REGIONS.some(r=>r.id===regionId))throw Error('Unknown region');
  return {...state,regions:state.regions.map(r=>r.id===regionId?
    {...r,modifiers:{...r.modifiers,[mechanism]:clamp(value)}}:r)};
}
export function withTumor(state,regionId,fraction){
  if(!REGIONS.some(r=>r.id===regionId))throw Error('Unknown region');
  return {...state,regions:state.regions.map(r=>r.id===regionId?{...r,tumorFraction:clamp(fraction)}:r)};
}
export function withTherapy(state,therapy){
  if(!THERAPIES[therapy])throw Error('Unknown therapy');
  return {...state,therapy,timeDays:0};
}
export function advance(state,deltaDays){
  if(!Number.isFinite(deltaDays)||deltaDays<0)throw Error('Time increment must be non-negative');
  return {...state,timeDays:state.timeDays+deltaDays};
}
export function calculate(state){
  const treatment=THERAPIES[state.therapy];if(!treatment)throw Error('Unknown therapy');
  const activation=1-Math.exp(-state.timeDays/treatment.tauDays);
  const rows=REGIONS.map((r)=>{
    const entry=state.regions.find(x=>x.id===r.id);
    const m={};
    for(const key of Object.keys(state.baseline)){
      const untreated=clamp(state.baseline[key]+(entry?.modifiers[key]||0));
      m[key]=clamp(untreated*(1-(treatment.targets[key]||0)*activation));
    }
    const tumor=clamp(Math.max(m.tumor,entry?.tumorFraction||0));
    // Dimensionless resistance proxy from relative caliber; not a flow solver.
    const relativeRadius=clamp(
      1-0.30*m.bronchoconstriction-0.18*m.mucus-0.17*m.remodeling-
      0.10*m.inflammation-0.32*tumor,0.18,1);
    const resistanceIndex=Math.pow(1/relativeRadius,4);
    const complianceIndex=clamp(1+0.75*m.emphysema-0.48*m.remodeling-0.18*m.inflammation,0.25,1.8);
    const conductance=(r.weight/resistanceIndex)*complianceIndex*(1-0.75*tumor);
    return {id:r.id,side:r.side,label:r.label,weight:r.weight,
      mechanisms:{...m,tumor},radiusIndex:relativeRadius,
      resistanceIndex,complianceIndex,conductance};
  });
  const sum=rows.reduce((s,r)=>s+r.conductance,0);
  const referenceSum=REGIONS.reduce((s,r)=>s+r.weight,0);
  const regions=rows.map(r=>({...r,ventilationFraction:r.conductance/sum,
    ventilationIndex:r.conductance/r.weight}));
  return {schema:state.schema,scenario:state.scenario,therapy:state.therapy,
    simulatedDays:state.timeDays,therapyActivation:activation,regions,
    summary:{overallVentilationIndex:sum/referenceSum,
      meanResistanceIndex:mean(rows.map(r=>r.resistanceIndex)),
      ventilationHeterogeneity:Math.sqrt(mean(regions.map(r=>Math.pow(r.ventilationIndex-mean(regions.map(x=>x.ventilationIndex)),2))))},
    validity:{kind:'illustrative, dimensionless surrogate',validated:false,
      note:'Not a clinical prediction, anatomical measurement, CFD result or calibrated pharmacodynamic model.'}};
}
