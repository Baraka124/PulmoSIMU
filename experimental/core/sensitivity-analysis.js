/**
 * Uncertainty/sensitivity gate for the R3 respiratory mechanics.
 * Local one-at-a-time perturbations do not represent a credible population
 * distribution, nor provide clinical confidence intervals.
 */
import {buildNetwork,simulateBreathing,R3_DEFAULTS} from './breathing-network.js';
const PARAMETERS=['sceneUnitMetres','distalResistancePaSPerM3','totalComplianceM3PerPa','gasViscosityPaS'];
export function parameterSensitivity(graph,state,{perturbationFraction=0.2,durationSeconds=15,overrides={}}={}){
 if(!Number.isFinite(perturbationFraction)||perturbationFraction<=0||perturbationFraction>=1)throw Error('Sensitivity fraction must be between zero and one');
 const baselineSettings={...R3_DEFAULTS,...overrides};
 const compute=values=>simulateBreathing(buildNetwork(graph,state,values),durationSeconds).summary.approximateTidalVolumeL;
 const baseline=compute(baselineSettings);
 const effects=PARAMETERS.map(key=>{
  const low=compute({...baselineSettings,[key]:baselineSettings[key]*(1-perturbationFraction)});
  const high=compute({...baselineSettings,[key]:baselineSettings[key]*(1+perturbationFraction)});
  return {parameter:key,assumedValue:baselineSettings[key],low,baseline,high,
   spreadL:Math.abs(high-low),relativeSpread:Math.abs(high-low)/Math.max(1e-9,baseline)};
 });
 return {baselineTidalVolumeL:baseline,perturbationFraction,
  effects:effects.sort((a,b)=>b.relativeSpread-a.relativeSpread),
  provenance:{validated:false,method:'one-at-a-time parameter perturbation',
   disclaimer:'Exploratory numerical sensitivity, not statistical uncertainty or clinically validated tolerance'}};
}
