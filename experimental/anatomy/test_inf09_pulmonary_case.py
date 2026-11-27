import json
import struct
import tempfile
import unittest
from pathlib import Path
import numpy as np
import nibabel as nib
from imaging_intake import normalize
from pulmonary_case_runner import run,intersection_report

SID="12345678-1234-4234-8234-123456789abc"

class CaseTests(unittest.TestCase):
    def test_common_roi(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            data=np.ones((4,4,4),dtype=np.uint8)
            for name in ("a","b"):
                nib.save(nib.Nifti1Image(data,np.eye(4)),str(root/(name+".nii")))
            a={"structure":"RUL","roi":[0,4,0,4,0,4],"roi_mask":root/"a.nii"}
            b={"structure":"RML","roi":[2,6,0,4,0,4],"roi_mask":root/"b.nii"}
            r=intersection_report(a,b,.002)
            self.assertEqual(r["common_evaluated_voxels"],32)
            self.assertEqual(r["overlap_voxels"],32)
            self.assertAlmostEqual(r["overlap_ml"],.064)
            self.assertEqual(r["review"],"OVERLAP_REVIEW_REQUIRED")
            b["roi"]=[4,8,0,4,0,4]
            self.assertIsNone(intersection_report(a,b,.002)["overlap_voxels"])
    def test_case_geometry_and_feature_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            data=np.zeros((15,15,15),dtype=np.int16)
            data[4:8,4:8,4:8]=-800
            data[9:12,9:12,9:12]=-500
            affine=np.diag([1.,1.,2.,1.])
            im=nib.Nifti1Image(data,affine);im.header.set_xyzt_units("mm");im.set_sform(affine,code=1)
            nib.save(im,str(root/"scan.nii"))
            normalize(root/"scan.nii",root/"intake",SID,modality="CT",quantity="HU")
            shared={"roi":[2,14,2,14,2,14],"enhancement":{"iterations":1,"kappa":10.,"scales_mm":[1.],"bright":False}}
            requests=[{**shared,"structure":"airway_lumen","segmentation":{"foreground":[[3,3,3]],"lower_hu":-900,"upper_hu":-700,"method":"region_growing"}},
                      {**shared,"structure":"lung","segmentation":{"foreground":[[8,8,8]],"lower_hu":-600,"upper_hu":-400,"method":"region_growing"}}]
            r=run(root/"intake",root/"case",root/"cache",requests,geometry=True)
            self.assertEqual(r["feature_groups"],1)
            self.assertEqual(len(r["structures"]),2)
            self.assertEqual(r["pairwise_overlap"][0]["overlap_voxels"],0)
            self.assertFalse(r["clinical_navigation_allowed"])
            glb=root/"case"/r["geometry"]["path"]
            with glb.open("rb") as f:
                magic,version,length=struct.unpack("<4sII",f.read(12))
                n,tag=struct.unpack("<I4s",f.read(8))
                scene=json.loads(f.read(n))
            self.assertEqual(magic,b"glTF")
            self.assertEqual(version,2)
            self.assertEqual(length,glb.stat().st_size)
            self.assertEqual({n["name"] for n in scene["nodes"]},{"airway_lumen","lung"})
            for structure in ("airway_lumen","lung"):
                self.assertTrue((root/"case"/"structures"/structure/"computational-assets.json").is_file())
