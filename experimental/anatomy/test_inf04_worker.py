import json
from pathlib import Path
import tempfile
import unittest
import uuid
import numpy as np
import nibabel as nib
from study_worker import build, sha

class WorkerTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.ct=self.root/"ct.nii";self.mask=self.root/"mask.nii"
        shape=(15,15,15);a=np.eye(4)
        data=np.zeros(shape,dtype=np.uint8);data[5:10,5:10,3:12]=1
        for path,arr in [(self.ct,np.full(shape,-800,dtype=np.int16)),(self.mask,data)]:
            im=nib.Nifti1Image(arr,a);im.header.set_xyzt_units("mm");im.set_sform(a,code=1);nib.save(im,str(path))
        self.id=uuid.uuid4();self.out=self.root/"result"
    def run_worker(self):
        return build(self.ct,self.out,self.id,mask=self.mask,engine="numerical fixture",engine_version="1")
    def test_package(self):
        m=self.run_worker()
        self.assertEqual(m["status"],"review_pending")
        self.assertFalse(m["clinical_navigation_allowed"])
        self.assertEqual(m["source_sha256"],sha(self.ct))
        self.assertEqual(m,json.loads((self.out/"study-package.json").read_text()))
        self.assertTrue(any(a["kind"]=="glb" for a in m["assets"]))
        self.assertTrue(any(a["kind"]=="airway_graph" for a in m["assets"]))
        for a in m["assets"]:
            p=self.out/a["path"];self.assertEqual(a["sha256"],sha(p));self.assertEqual(a["bytes"],p.stat().st_size)
            self.assertTrue(a["object_key"].startswith(str(self.id)+"/v1/"))
        with self.assertRaises(ValueError): self.run_worker()
    def test_wrong_grid_is_atomic(self):
        im=nib.load(str(self.mask));a=np.eye(4);a[0,3]=10;im.set_sform(a,code=1);nib.save(im,str(self.mask))
        with self.assertRaises(ValueError):self.run_worker()
        self.assertFalse(self.out.exists())
    def test_missing_provenance_rejected(self):
        with self.assertRaises(ValueError):build(self.ct,self.out,self.id,mask=self.mask)
        self.assertFalse(self.out.exists())
if __name__=="__main__":unittest.main()
