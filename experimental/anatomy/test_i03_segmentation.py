"""Synthetic geometry tests only. Run: python -m unittest discover -s experimental/anatomy -p 'test_i03*.py'"""
import unittest
import numpy as np
from segmentation_pipeline import topology, apply_patch

class SegmentationTests(unittest.TestCase):
    def test_disconnected_components_and_volume(self):
        m=np.zeros((6,6,6),dtype=np.uint8)
        m[1:4,1,1]=1;m[5,5,5]=1
        report,graph=topology(m,np.diag([1.,2.,3.,1.]),True)
        self.assertEqual(report["component_count"],2)
        self.assertEqual(report["fragment_voxels"],1)
        self.assertAlmostEqual(report["volume_ml"],.024)
        self.assertFalse(graph["navigation_allowed"])
    def test_patch_order_and_identity(self):
        m=np.zeros((2,2,2),dtype=np.uint8)
        p={"schema":"pulmosimu.mask-patch/1","ct_sha256":"a","mask_sha256":"b","structure":"airway_lumen","review":"PENDING","edits":[{"index":1,"value":1}]}
        out=apply_patch(m,p,"a","b","airway_lumen")
        self.assertEqual(out[1,0,0],1)
        self.assertEqual(m.sum(),0)
        with self.assertRaises(ValueError): apply_patch(m,p,"wrong","b","airway_lumen")
