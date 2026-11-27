import unittest
import numpy as np
from deterministic_segmentation import segment

class GeodesicTests(unittest.TestCase):
    def run_case(self,data=None,**kwargs):
        if data is None: data=np.full((9,9,9),-800,dtype=np.float32)
        fields={}
        parameters={"max_cost_mm":2.1,"gradient_scale":80.,"edge_weight":0.,"smooth_mm":0.,"iterations":0,"max_volume_ml":10.}
        parameters.update(kwargs.pop("refinement",{}))
        args={"foreground":[[4,4,4]],"lower_hu":-900,"upper_hu":-700}
        args.update(kwargs)
        m,d,r=segment(data,[1,1,2],method="geodesic_discrete_contour",refinement=parameters,intermediates=fields,**args)
        return m,d,r,fields
    def test_physical_cost(self):
        m,d,r,f=self.run_case()
        self.assertEqual(f["geodesic-cost.nii"][5,4,4],1)
        self.assertEqual(f["geodesic-cost.nii"][4,4,5],2)
        self.assertEqual(m[4,4,6],0)
        self.assertEqual(m[4,4,5],1)
        self.assertTrue(np.all(d[m>0]<0))
    def test_barrier(self):
        a=np.full((9,9,9),-800,dtype=np.float32);a[5,:,:]=0
        m,_,_,f=self.run_case(a,refinement={"max_cost_mm":30})
        self.assertFalse(m[6:].any())
        self.assertEqual(f["geodesic-cost.nii"][6,4,4],-1)
    def test_background_constraint(self):
        m,_,_,_=self.run_case(background=[[4,4,5]],refinement={"max_cost_mm":4,"smooth_mm":.5,"iterations":8})
        self.assertEqual(m[4,4,5],0)
        self.assertEqual(m[4,4,4],1)
    def test_energy_descent(self):
        m,_,r,_=self.run_case(refinement={"max_cost_mm":3.5,"smooth_mm":.5,"iterations":8})
        e=r["refinement"]["energy_history_before_connectivity"]
        self.assertTrue(all(b<=a+1e-5 for a,b in zip(e,e[1:])))
        self.assertEqual(m[4,4,4],1)
    def test_volume_guard(self):
        with self.assertRaisesRegex(ValueError,"volume ceiling"):
            self.run_case(refinement={"max_volume_ml":.0001})
    def test_refinement_parameters_required(self):
        with self.assertRaises(ValueError):
            segment(np.zeros((5,5,5)),[1,1,1],foreground=[[2,2,2]],lower_hu=-1,upper_hu=1,method="geodesic_discrete_contour")
