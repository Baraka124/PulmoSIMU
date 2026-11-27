(function fidelity(root){
'use strict';
const check=(x,m)=>{if(!x)throw Error(m)},finite=Number.isFinite,hash=x=>typeof x==='string'&&/^[a-f0-9]{64}$/.test(x),num=(x,min=0,max=1e12)=>finite(x)&&x>=min&&x<=max,int=(x,min=0,max=1e12)=>Number.isSafeInteger(x)&&num(x,min,max);
function inspect(doc,study,seg){
 check(study?.schema==='pulmosimu.study/1'&&seg?.structure==='airway_lumen'&&seg.patch().edits.length===0,'Load exact unchanged airway mask');
 check(doc?.schema==='pulmosimu.anatomy-fidelity/1'&&doc.review==='PENDING'&&doc.clinical_navigation_allowed===false&&doc.structure==='airway_lumen'&&doc.ct_sha256===study.source.sha256&&doc.mask_sha256===seg.digest&&hash(doc.mesh_sha256),'Report identity or review status mismatch');
 const s=doc.surface,u=doc.upper_airway,sections=doc.sections;
 check(int(doc.foreground_voxels,1,5000000)&&int(doc.component_count,1,doc.foreground_voxels)&&num(doc.largest_component_fraction,0,1)&&num(doc.voxel_volume_ml)&&doc.foreground_voxels===seg.metrics().edited_voxels,'Invalid mask statistics');
 check(s&&int(s.vertices,1,4500000)&&int(s.triangles,1,1500000)&&int(s.boundary_edges)&&int(s.nonmanifold_edges)&&num(s.surface_area_mm2)&&num(s.enclosed_mesh_volume_ml),'Invalid mesh statistics');
 check(Array.isArray(doc.fov_boundary_faces)&&doc.fov_boundary_faces.length<=6&&doc.fov_boundary_faces.every(x=>/^axis[012]_(min|max)$/.test(x)),'Invalid boundary flags');
 check(u&&['NOT_ASSESSED','ALL_MARKS_SAME_6_CONNECTED_COMPONENT','INCOMPLETE_OR_DISCONNECTED'].includes(u.state)&&Array.isArray(u.landmarks)&&u.landmarks.length<=5,'Invalid landmark report');
 const names=['oral_cavity','pharynx','larynx','trachea','distal_airway'],dims=study.source.dimensions,voxel=p=>Array.isArray(p)&&p.length===3&&p.every((v,i)=>int(v,0,dims[i]-1));
 check(u.landmarks.every(r=>names.includes(r.name)&&voxel(r.voxel)&&int(r.component,0,doc.component_count)&&(r.component>0)===(seg.at(r.voxel)===1))&&new Set(u.landmarks.map(r=>r.name)).size===u.landmarks.length,'Landmark mask mismatch');
 const connected=u.landmarks.length===5&&u.landmarks[0].component>0&&new Set(u.landmarks.map(r=>r.component)).size===1;
 check(connected===(u.state==='ALL_MARKS_SAME_6_CONNECTED_COMPONENT')&&u.same_component===connected,'Landmark status inconsistent');
 check(Array.isArray(sections)&&sections.length<=32&&sections.every(r=>voxel(r.center_voxel)&&seg.at(r.center_voxel)===1&&num(r.step_mm,.25,2)&&num(r.radius_mm,2,40)&&typeof r.truncated==='boolean'&&(r.truncated?r.sampled_area_mm2===null&&r.area_equivalent_diameter_mm===null:num(r.sampled_area_mm2,0,6000)&&num(r.area_equivalent_diameter_mm,0,100)&&Math.abs(r.area_equivalent_diameter_mm-2*Math.sqrt(r.sampled_area_mm2/Math.PI))<.001)),'Invalid section report');
 return {meshLinked:study.surface?.sha256===doc.mesh_sha256,componentCount:doc.component_count,largestFraction:doc.largest_component_fraction,volumeMl:doc.voxel_volume_ml,triangles:s.triangles,boundaryEdges:s.boundary_edges,nonmanifoldEdges:s.nonmanifold_edges,surfaceAreaMm2:s.surface_area_mm2,fovBoundary:doc.fov_boundary_faces.slice(),upperState:u.state,sections:sections.map(r=>({center:r.center_voxel.slice(),truncated:r.truncated,area:r.sampled_area_mm2,diameter:r.area_equivalent_diameter_mm})),review:'PENDING',reportRecomputed:false,clinicalNavigationAllowed:false};
}
root.PulmoAnatomyFidelity=Object.freeze({inspect});
})(typeof window!=='undefined'?window:globalThis);
