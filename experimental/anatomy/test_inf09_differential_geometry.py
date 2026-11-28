import tempfile
import unittest
from pathlib import Path
import numpy as np
import nibabel as nib
from differential_geometry import curve
from volumetric_geometry import fields,skeleton_distance
from branch_geometry_workflow import run as branches
from differential_geometry_workflow import run

class DifferentialTests(unittest.TestCase):
    def test_line(self):
        p=np.c_[np.arange(30.),np.zeros((30,2))]
        r=curve(p,step_mm=.5,smooth_mm=1.)
        self.assertEqual(r["status"],"ESTIMATED")
        self.assertLess(max(x["curvature_per_mm"] for x in r["samples"]),1e-8)
        self.assertTrue(all(x["torsion_per_mm"] is None for x in r["samples"]))
    def test_circle(self):
        t=np.linspace(0,2.5,501);p=np.c_[10*np.cos(t),10*np.sin(t),np.zeros_like(t)]
        r=curve(p,step_mm=.25,smooth_mm=.5)
        self.assertAlmostEqual(np.median([x["curvature_per_mm"] for x in r["samples"]]),.1,delta=.008)
        self.assertLess(max(abs(x["torsion_per_mm"]) for x in r["samples"] if x["torsion_per_mm"] is not None),1e-8)
    def test_sphere(self):
        q=np.indices((61,61,61),dtype=float)*.5-15
        m=np.sum(q*q,axis=0)<=64
        f,r=fields(m,np.diag([.5,.5,.5,1.]),sigma_mm=1.,band_mm=.5)
        v=f["curvature-valid-mask"]>0
        self.assertGreater(v.sum(),0)
        self.assertAlmostEqual(np.median(f["mean-curvature"][v]),.125,delta=.04)
        self.assertAlmostEqual(np.median(f["gaussian-curvature"][v]),1/64,delta=.015)
        self.assertLess(f["signed-distance-mm"][30,30,30],0)
        self.assertGreater(f["signed-distance-mm"][0,0,0],0)
        self.assertFalse(v[0].any())
    def test_physical_skeleton_distance(self):
        m=np.zeros((9,9,9),dtype=np.uint8);m[4,4,4]=1
        d,sk=skeleton_distance(m,np.diag([1.,2.,3.,1.]),{"nodes_voxel_ijk":[[4,4,4]]})
        self.assertEqual(d[4,4,5],3)
        self.assertEqual(sk.sum(),1)
    def test_source_bound_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);x,y,z=np.indices((41,41,41))
            m=(((x-20)**2+(y-20)**2<=25)&(z>=5)&(z<=35)).astype(np.uint8)
            a=np.diag([.5,.5,.5,1.])
            for name,data in (("ct",np.full(m.shape,-800,dtype=np.int16)),("mask",m)):
                im=nib.Nifti1Image(data,a);im.header.set_xyzt_units("mm");im.set_sform(a,code=1)
                nib.save(im,str(root/(name+".nii")))
            ct=root/"ct.nii";mask=root/"mask.nii"
            branches(ct,mask,root/"branches",structure="airway_lumen",section_parameters={"radius_mm":4.})
            r=run(ct,mask,root/"branches",root/"out")
            self.assertEqual(len(r["assets"]),11)
            self.assertGreater(r["curve_samples"],0)
            self.assertFalse(r["clinical_navigation_allowed"])
            p=root/"branches"/"branch-graph.json";p.write_text(p.read_text()+" ")
            with self.assertRaisesRegex(ValueError,"Graph integrity"):
                run(ct,mask,root/"branches",root/"out2")
