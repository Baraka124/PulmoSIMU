/**
 * PulmoSIMU R2 — versioned airway graph adapter.
 * Consumes a faithful snapshot of the existing PNEUMO·VR SEGS graph.
 * No patient geometry or full anatomical accuracy is implied.
 */
export function loadAirwayGraph(data){
 if(data?.schema!=='pulmosimu-airway-reference-r2'||!Array.isArray(data.nodes))throw Error('Unsupported anatomy data');
 const byId=new Map();
 for(const node of data.nodes){
  if(!Number.isInteger(node.id)||byId.has(node.id))throw Error('Duplicate/invalid airway ID');
  if(!Array.isArray(node.points)||node.points.length<2||node.points.some(p=>p.length!==3||p.some(x=>!Number.isFinite(x))))throw Error('Invalid airway coordinates');
  if(!(node.radiusStart>0&&node.radiusEnd>0))throw Error('Invalid airway radii');
  byId.set(node.id,{...node,children:[...node.children],points:node.points.map(p=>[...p])});
 }
 for(const node of byId.values())for(const child of node.children)if(!byId.has(child))throw Error('Missing child airway '+child);
 if(!byId.has(0))throw Error('Missing root trachea');
 const reached=new Set(),active=new Set();
 function walk(id){if(active.has(id))throw Error('Cyclic airway graph');if(reached.has(id))throw Error('Multiple-parent graph');active.add(id);reached.add(id);for(const c of byId.get(id).children)walk(c);active.delete(id)}
 walk(0);if(reached.size!==byId.size)throw Error('Disconnected airway graph');
 const parent=new Map([[0,null]]);
 for(const node of byId.values())for(const child of node.children)parent.set(child,node.id);
 return Object.freeze({schema:data.schema,units:data.units,source:data.source,
  nodes:Object.freeze([...byId.values()]),byId,parent,
  root:byId.get(0),get:(id)=>byId.get(id),
  chain:(id)=>{if(!byId.has(id))throw Error('Unknown airway');const chain=[];while(id!==null){chain.unshift(byId.get(id));id=parent.get(id)}return chain},
  regionOf:(id)=>{if(!byId.has(id))throw Error('Unknown airway');const path=[];while(id!==null){path.unshift(id);id=parent.get(id)}
   const regionId=path.find(x=>[2,3,5,6,7].includes(x));return ({2:'LUL',3:'LLL',5:'RUL',6:'RML',7:'RLL'})[regionId]??null}
 });
}
export async function fetchReferenceAirwayGraph(url=new URL('./reference-airways.json',import.meta.url)){
 const response=await fetch(url);if(!response.ok)throw Error('Could not load airway reference');
 return loadAirwayGraph(await response.json());
}
