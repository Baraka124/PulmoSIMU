import tempfile
import json
import unittest
from pathlib import Path
import numpy as np
from deterministic_segmentation import segment
from computational_assets import file_hash
from pulmonary_protocols import compile_protocol

class LevelsetTests(unittest.TestCase):
    def case(self,**kw):
        a=np.full((11,11,11),-800,dtype=np.float32)
        a[6,:,:]=0
        p={"seed_radius_mm":1.1,"inside_hu":-800,"outside_hu":0,"max_volume_ml":10.,"curvature_mm":.25,"iterations":12}
        p.update(kw)
        fields={}
        return (*segment(a,[1,1,2],foreground=[[4,5,5]],background=[[4,6,5]],lower_hu=-900,upper_hu=-700,method="region_levelset",refinement=p,intermediates=fields),fields)
    def test_constraints_and_fields(self):
        m,d,r,f=self.case()
        self.assertEqual(m[4,5,5],1)
        self.assertEqual(m[4,6,5],0)
        self.assertFalse(m[6:].any())
        self.assertTrue(np.array_equal(f["levelset-final.nii"]>0,m.astype(bool)))
        self.assertTrue(np.isfinite(f["levelset-final.nii"]).all())
        self.assertEqual(len(r["refinement"]["voxel_history"]),13)
    def test_ceiling(self):
        with self.assertRaisesRegex(ValueError,"volume ceiling"):
            self.case(max_volume_ml=.0001)
    def test_reproducible(self):
        a,_,_,f=self.case()
        b,_,_,g=self.case()
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(f["levelset-final.nii"],g["levelset-final.nii"])

class ProtocolTests(unittest.TestCase):
    def test_source_and_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/"source.nii";source.write_bytes(b"fixture")
            (root/"intake.json").write_text(json.dumps({"schema":"pulmo.imaging-intake/1","study_id":"12345678-1234-4234-8234-123456789abc","modality":"CT","quantity":"HU","normalized":{"sha256":file_hash(source),"shape":[10,10,10]}}))
            req={"structure":"airway_lumen","roi":[0,8,0,8,0,8],"enhancement":{"iterations":2,"kappa":40,"scales_mm":[1,2],"bright":False},"segmentation":{"foreground":[[3,3,3]],"lower_hu":-950,"upper_hu":-600,"method":"region_growing"}}
            a=compile_protocol(root,[req])
            self.assertEqual(a,compile_protocol(root,[req]))
            with self.assertRaises(ValueError): compile_protocol(root,[req,req])
            source.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError,"Source changed"): compile_protocol(root,[req])
