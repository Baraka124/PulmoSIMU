import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import nibabel as nib
from ct_enhancement import diffuse,vesselness
from imaging_intake import normalize
from ct_feature_workflow import execute

SID="12345678-1234-4234-8234-123456789abc"

class FeatureTests(unittest.TestCase):
    def test_constant_field(self):
        a=np.full((9,9,9),-700,dtype=np.float32)
        b,q=diffuse(a,[.7,.7,2])
        np.testing.assert_array_equal(a,b)
        self.assertEqual(q["rmse_change_HU"],0)
        v,s,_=vesselness(a,[.7,.7,2])
        np.testing.assert_array_equal(v,0)
        np.testing.assert_array_equal(s,0)
    def test_diffusion_conservation(self):
        a=np.zeros((9,9,9),dtype=np.float32);a[4,4,4]=20
        b,_=diffuse(a,[1,1,2],iterations=4,kappa=40)
        self.assertLess(float(b.max()),20)
        self.assertAlmostEqual(float(b.sum()),20,places=4)
        self.assertEqual(float(a.max()),20)
    def test_unstable_step(self):
        with self.assertRaises(ValueError): diffuse(np.zeros((5,5,5)),[1,1,1],step=1)
    def test_tube_response(self):
        x,y,z=np.indices((25,25,25))
        a=(300*np.exp(-((x-12)**2+(y-12)**2)/8)).astype(np.float32)
        v,s,_=vesselness(a,[1,1,1],scales_mm=(1,2))
        self.assertGreater(v[12,12,12],v[2,2,12])
        self.assertIn(s[12,12,12],(1,2))
        self.assertTrue(np.all((v>=0)&(v<=1)))
    def test_workflow_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            a=np.arange(12**3,dtype=np.int16).reshape((12,12,12))-1000
            affine=np.diag([1.,1.,2.,1.]);affine[:3,3]=[10,20,30]
            im=nib.Nifti1Image(a,affine);im.header.set_xyzt_units("mm")
            im.set_sform(affine,code=1)
            nib.save(im,str(root/"source.nii"))
            normalize(root/"source.nii",root/"intake",SID,modality="CT",quantity="HU")
            kw={"roi":[2,10,1,9,3,11],"iterations":2}
            execute(root/"intake",root/"run1",root/"cache",**kw,scales_mm=(1.,))
            m=execute(root/"intake",root/"run2",root/"cache",**kw,scales_mm=(2.,))
            r=json.loads((root/"run2"/"execution.json").read_text())
            self.assertTrue(r["diffusion_cache_hit"])
            self.assertFalse(r["vesselness_cache_hit"])
            self.assertEqual(len(m["assets"]),8)
            crop=nib.load(str(root/"run2"/"roi.nii"))
            np.testing.assert_allclose(crop.affine[:3,3],[12,21,36])
            execute(root/"intake",root/"run3",root/"cache",**kw,scales_mm=(2.,))
            r=json.loads((root/"run3"/"execution.json").read_text())
            self.assertTrue(r["diffusion_cache_hit"])
            self.assertTrue(r["vesselness_cache_hit"])
            victim=next((root/"cache").glob("*/enhanced.nii"))
            victim.write_bytes(b"modified")
            with self.assertRaises(ValueError):
                execute(root/"intake",root/"run4",root/"cache",**kw,scales_mm=(2.,))
