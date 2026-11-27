import unittest
import numpy as np
from inverse_consistency import evaluate
from mass_conservation_refinement import refine

class InverseTests(unittest.TestCase):
    def test_translation_with_anisotropy(self):
        u=np.zeros((8,8,8,3),dtype=np.float32);u[...,0]=2
        v=-u
        a=np.diag([2.,1.,1.,1.])
        fields,r=evaluate(u,a,v,a,landmarks=[{"fixed_RAS_mm":[4,3,3],"moving_RAS_mm":[6,3,3]}])
        self.assertEqual(r["fixed_to_moving_to_fixed"]["support_fraction"],.875)
        self.assertEqual(r["fixed_to_moving_to_fixed"]["roundtrip"]["max_mm"],0)
        self.assertEqual(r["landmarks"]["forward_TRE"]["max_mm"],0)
        self.assertFalse(fields["fixed-roundtrip-support"][-1].any())
    def test_inconsistent_reverse(self):
        u=np.zeros((5,5,5,3),dtype=np.float32);u[...,0]=1
        fields,r=evaluate(u,np.eye(4),np.zeros_like(u),np.eye(4))
        self.assertEqual(r["fixed_to_moving_to_fixed"]["roundtrip"]["mean_mm"],1)
    def test_invalid_landmarks(self):
        u=np.zeros((5,5,5,3),dtype=np.float32)
        with self.assertRaises(ValueError):
            evaluate(u,np.eye(4),u,np.eye(4),landmarks=[{"fixed_RAS_mm":[10,0,0],"moving_RAS_mm":[0,0,0]}])

class ConservationTests(unittest.TestCase):
    def fixture(self):
        x=np.full((12,12,12),-800,dtype=np.float32);y=np.full_like(x,-780)
        mi=np.zeros_like(x);mi[3:9,3:9,3:9]=1
        me=np.ones_like(x);u=np.zeros(x.shape+(3,),dtype=np.float32)
        return x,y,mi,me,np.eye(4),np.eye(4),u
    def test_conservation_correction(self):
        u,j,m,r0,r1,r=refine(*self.fixture(),regularization=.00001,max_iterations=20)
        self.assertTrue(r["accepted"])
        self.assertLess(r["refined_mass_mse"],r["baseline_mass_mse"])
        self.assertGreater(j.min(),0)
        self.assertEqual(m.sum(),216)
        self.assertLess(np.mean(r1[m>0]**2),np.mean(r0[m>0]**2))
    def test_exact_baseline_preserved(self):
        x,y,mi,me,a,b,u=self.fixture()
        result,j,m,r0,r1,r=refine(x,x,mi,me,a,b,u,max_iterations=2)
        self.assertFalse(r["accepted"])
        np.testing.assert_array_equal(result,u)
    def test_folding_rejected(self):
        x,y,mi,me,a,b,u=self.fixture()
        u[...,0]=-2*np.indices(x.shape)[0]
        with self.assertRaisesRegex(ValueError,"folding"):refine(x,y,mi,me,a,b,u)
