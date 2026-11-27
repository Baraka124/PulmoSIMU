import gzip
import tempfile
import unittest
import zipfile
from pathlib import Path
import numpy as np
import nibabel as nib
from imaging_intake import identify, normalize, extract_archive
from computational_assets import Registry

SID="12345678-1234-4234-8234-123456789abc"

class IntakeTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
    def test_compressed_volume(self):
        p=self.root/"scan.nii.gz"
        im=nib.Nifti1Image(np.full((4,5,6),-700,dtype=np.int16),np.diag([1,1,2,1]))
        im.header.set_xyzt_units("mm")
        im.set_sform(im.affine,code=1)
        nib.save(im,str(p))
        self.assertEqual(identify(p),"gzip_nifti_candidate")
        r=normalize(p,self.root/"output",SID,modality="CT",quantity="HU")
        self.assertEqual(r["normalized"]["shape"],[4,5,6])
        self.assertEqual(r["normalized"]["spacing_mm"],[1,1,2])
        self.assertEqual(r["normalized"]["intensity_range"],[-700,-700])
        self.assertFalse(r["deidentification_certified"])
    def test_archive_traversal(self):
        p=self.root/"bad.zip"
        with zipfile.ZipFile(p,"w") as z: z.writestr("../escape.dcm",b"test")
        with self.assertRaises(ValueError): extract_archive(p,self.root/"stage")
        self.assertFalse((self.root/"escape.dcm").exists())
    def test_non_ct_blocked(self):
        p=self.root/"a.nrrd";p.write_bytes(b"NRRD0005")
        self.assertEqual(identify(p),"nrrd_not_supported")
        with self.assertRaises(ValueError): normalize(p,self.root/"output",SID,modality="MR")

class AssetTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.reg=Registry(self.root,SID)
        for f in ("source.nii","filtered.nii","mask.nii"): (self.root/f).write_bytes(f.encode())
    def add(self,path,kind,parents=()):
        return self.reg.add(path,kind,quantity="test",units="1",frame_id="RAS-test",parents=parents,operation="fixture",version="1")
    def test_lineage(self):
        a=self.add("source.nii","source_volume")
        b=self.add("filtered.nii","processed_volume",[a])
        c=self.add("mask.nii","label_mask",[b])
        self.assertEqual(self.reg.descendants(a),[b,c])
        self.assertEqual(self.reg.descendants(b),[c])
        self.assertEqual(len(self.reg.export()["assets"]),3)
    def test_modified_bytes(self):
        self.add("source.nii","source_volume")
        (self.root/"source.nii").write_bytes(b"changed")
        with self.assertRaises(ValueError): self.reg.verify()
    def test_missing_parent(self):
        with self.assertRaises(ValueError): self.add("mask.nii","label_mask",["unknown"])
    def test_traversal(self):
        with self.assertRaises(ValueError): self.add("../source.nii","source_volume")
