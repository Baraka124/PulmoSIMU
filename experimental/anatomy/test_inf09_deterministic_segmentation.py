import tempfile
import unittest
from pathlib import Path
import numpy as np
import nibabel as nib
from deterministic_segmentation import segment
from imaging_intake import normalize
from ct_feature_workflow import execute as features
from segmentation_workflow import execute

SID="12345678-1234-4234-8234-123456789abc"
class SegmentationTests(unittest.TestCase):
    def test_disconnected_regions(self):
        a=np.zeros((9,9,9),dtype=np.float32)
        a[2:4,2:4,2:4]=-800;a[6:8,6:8,6:8]=-800
        mask,d,r=segment(a,[1,1,2],foreground=[[2,2,2]],lower_hu=-900,upper_hu=-700)
        self.assertEqual(mask.sum(),8)
        self.assertEqual(r["components"],1)
        self.assertAlmostEqual(r["volume_ml"],.016)
        self.assertEqual(d[2,2,2],-1)
        self.assertEqual(d[1,2,2],1)
        self.assertFalse(r["touches_roi_boundary"])
    def test_watershed_markers(self):
        a=np.full((9,9,9),-700,dtype=np.float32)
        m,_,r=segment(a,[1,1,1],foreground=[[2,4,4]],background=[[6,4,4]],lower_hu=-800,upper_hu=-600,method="watershed")
        self.assertEqual(m[2,4,4],1)
        self.assertEqual(m[6,4,4],0)
        self.assertGreater(m.sum(),0)
        self.assertLess(m.sum(),a.size)
    def test_feature_constraint(self):
        a=np.full((7,7,7),-700);f=np.zeros_like(a);f[2:5,3,3]=1
        m,_,_=segment(a,[1,1,1],foreground=[[3,3,3]],lower_hu=-800,upper_hu=-600,feature=f,min_feature=.5)
        self.assertEqual(m.sum(),3)
    def test_bad_seed(self):
        with self.assertRaises(ValueError):
            segment(np.zeros((5,5,5)),[1,1,1],foreground=[[1,1,1]],lower_hu=-900,upper_hu=-700)
    def test_source_grid_embedding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            a=np.full((12,12,12),-800,dtype=np.int16)
            affine=np.diag([1.,1.,2.,1.]);affine[:3,3]=[10,20,30]
            im=nib.Nifti1Image(a,affine);im.header.set_xyzt_units("mm");im.set_sform(affine,code=1)
            nib.save(im,str(root/"original.nii"))
            normalize(root/"original.nii",root/"intake",SID,modality="CT",quantity="HU")
            features(root/"intake",root/"features",root/"cache",roi=[2,8,3,9,1,7],iterations=1,scales_mm=(1.,))
            m=execute(root/"features",root/"seg",structure="airway_lumen",foreground=[[1,1,1]],lower_hu=-900,upper_hu=-700)
            self.assertEqual(len(m["assets"]),12)
            out=nib.load(str(root/"seg"/"candidate-source-grid.nii.gz"))
            np.testing.assert_allclose(out.affine,affine)
            mask=np.asarray(out.dataobj)
            self.assertEqual(mask.sum(),216)
            self.assertEqual(mask[2,3,1],1)
            self.assertEqual(mask[1,3,1],0)
            self.assertTrue((root/"seg"/"signed-distance.nii").exists())
