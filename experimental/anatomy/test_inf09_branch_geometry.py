import tempfile
import unittest
from pathlib import Path
import numpy as np
import nibabel as nib
from anatomical_branch_graph import decompose,build
from branch_morphometry import section,measure
from branch_geometry_workflow import run

class GraphTests(unittest.TestCase):
    def test_chain_partition(self):
        p=np.array([[0,0,0],[1,0,0],[2,0,0],[3,1,0],[3,-1,0]])
        edges=[(0,1),(1,2),(2,3),(2,4)]
        b,c,d=decompose(p,edges,p.astype(float),np.ones(len(p)))
        self.assertEqual(len(b),3)
        self.assertEqual(c[0]["cycle_rank"],0)
        self.assertAlmostEqual(sum(x["length_mm"] for x in b),2+2*np.sqrt(2))
        self.assertEqual(d,[1,2,3,1,1])
    def test_closed_cycle_and_isolated_node(self):
        p=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[9,9,9]])
        b,c,d=decompose(p,[(0,1),(0,3),(1,2),(2,3)],p.astype(float),np.ones(5))
        self.assertEqual(len(b),1)
        self.assertTrue(b[0]["closed"])
        self.assertIsNone(b[0]["tortuosity"])
        self.assertEqual(c[0]["cycle_rank"],1)
        self.assertEqual(len(c),2)
        self.assertEqual(d[-1],0)
    def test_diagonal_components_remain_separate(self):
        m=np.zeros((7,7,7),dtype=np.uint8);m[2,2,2]=1;m[3,3,3]=1
        g=build(m,np.eye(4))
        self.assertEqual(g["mask_components_6"],2)
        self.assertEqual(len(g["graph_components"]),2)
        self.assertEqual(len(g["edges"]),0)

class MorphometryTests(unittest.TestCase):
    def tube(self):
        x,y,z=np.indices((41,41,41))
        # Physical 3 mm radius, 0.5 mm isotropic spacing.
        m=(((x-20)**2+(y-20)**2<=36)&(z>=5)&(z<=35)).astype(np.uint8)
        return m,np.diag([.5,.5,.5,1.])
    def test_physical_section(self):
        m,a=self.tube()
        r=section(m,a,[10,10,10],[0,0,1],radius_mm=5.,step_mm=.2)
        self.assertEqual(r["status"],"ESTIMATED")
        self.assertLess(abs(r["area_mm2"]-np.pi*9),3)
        self.assertLess(abs(r["equivalent_diameter_mm"]-6),.5)
        self.assertLess(r["eccentricity"],.2)
        self.assertGreater(r["hydraulic_diameter_mm"],0)
    def test_truncated_section(self):
        m,a=self.tube()
        r=section(m,a,[10,10,10],[0,0,1],radius_mm=2.,step_mm=.2)
        self.assertEqual(r["status"],"TRUNCATED_OR_OUTSIDE_FOV")
        self.assertIsNone(r["area_mm2"])
    def test_integrated_workflow(self):
        m,a=self.tube()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name,data in (("ct",np.full(m.shape,-800,dtype=np.int16)),("mask",m)):
                im=nib.Nifti1Image(data,a);im.header.set_xyzt_units("mm");im.set_sform(a,code=1)
                nib.save(im,str(root/(name+".nii")))
            r=run(root/"ct.nii",root/"mask.nii",root/"out",structure="airway_lumen",section_parameters={"radius_mm":5.,"interval_mm":2.})
            self.assertGreater(r["branch_count"],0)
            self.assertGreater(r["estimated_section_count"],0)
            self.assertEqual(len(r["assets"]),2)
            self.assertFalse(r["clinical_navigation_allowed"])
            with self.assertRaises(ValueError):
                run(root/"ct.nii",root/"mask.nii",root/"out",structure="airway_lumen")
