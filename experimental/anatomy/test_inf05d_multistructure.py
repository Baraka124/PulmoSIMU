import json
from pathlib import Path
import tempfile
import unittest
import uuid
import struct
import numpy as np
import nibabel as nib
from multistructure_worker import build, LOBES, sha

class MultiStructureTest(unittest.TestCase):
    def test_shared_frame_and_missing_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);m=r/"masks";m.mkdir();shape=(18,18,18);affine=np.diag([1.,2.,3.,1.])
            def save(p,data):
                im=nib.Nifti1Image(data,affine);im.header.set_xyzt_units("mm");im.set_sform(affine,code=1);nib.save(im,str(p))
            save(r/"ct.nii",np.zeros(shape,dtype=np.int16))
            for i,k in enumerate(LOBES):
                data=np.zeros(shape,dtype=np.uint8);data[2+i*3:4+i*3,3:6,3:6]=1;save(m/(k+".nii"),data)
            out=r/"out";result=build(r/"ct.nii",out,uuid.uuid4(),mask_dir=m,provider="numerical fixture",version="1")
            self.assertEqual(set(result["available_structures"]),set(LOBES)|{"lung"})
            self.assertIn("airway_lumen",result["missing_structures"])
            self.assertEqual(result["lobe_overlap_excess_memberships"],0)
            self.assertFalse(result["clinical_navigation_allowed"])
            with (out/"pulmonary.glb").open("rb") as f:
                header=f.read(12);size,kind=struct.unpack("<I4s",f.read(8));doc=json.loads(f.read(size))
            self.assertEqual(header[:4],b"glTF");self.assertEqual({n["name"] for n in doc["nodes"]},set(LOBES))
            for k in result["available_structures"]:
                q=json.loads((out/"structures"/k/"surface.json").read_text())
                self.assertEqual(q["model_from_ras_mm"],result["model_from_ras_mm"])
            self.assertEqual(json.loads((out/"structures/lung/surface.json").read_text())["foreground_voxels"],90)
            for a in result["assets"]:self.assertEqual(a["sha256"],sha(out/a["path"]))
    def test_wrong_affine_no_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);m=r/"masks";m.mkdir();data=np.ones((5,5,5),dtype=np.uint8)
            for p,a in [(r/"ct.nii",np.eye(4)),(m/"RUL.nii",np.diag([2.,1.,1.,1.]))]:
                im=nib.Nifti1Image(data,a);im.header.set_xyzt_units("mm");im.set_sform(a,code=1);nib.save(im,str(p))
            with self.assertRaises(ValueError):build(r/"ct.nii",r/"out",uuid.uuid4(),mask_dir=m,provider="fixture",version="1")
            self.assertFalse((r/"out").exists())
