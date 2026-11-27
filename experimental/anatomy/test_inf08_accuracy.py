import unittest
import numpy as np
from quantitative_accuracy import compare_masks

class AccuracyTests(unittest.TestCase):
    def setUp(self):
        self.a=np.zeros((5,5,5),dtype=np.uint8)
        self.a[2,2,2]=1
    def test_identity(self):
        m=compare_masks(self.a,self.a,[1,2,3])
        self.assertEqual(m["dice"],1)
        self.assertEqual(m["hd95_mm"],0)
        self.assertAlmostEqual(m["reference_volume_ml"],.006)
    def test_anisotropic_offset(self):
        b=np.roll(self.a,1,axis=2)
        m=compare_masks(b,self.a,[1,2,3])
        self.assertEqual(m["dice"],0)
        self.assertEqual(m["hd95_mm"],3)
        self.assertEqual(m["mean_symmetric_surface_distance_mm"],3)
    def test_missing_prediction(self):
        m=compare_masks(np.zeros_like(self.a),self.a,[1,1,1])
        self.assertEqual(m["status"],"prediction_empty")
        self.assertEqual(m["sensitivity"],0)
        self.assertIsNone(m["precision"])
        self.assertIsNone(m["hd95_mm"])
    def test_empty_pair(self):
        z=np.zeros_like(self.a)
        m=compare_masks(z,z,[1,1,1])
        self.assertEqual(m["status"],"both_empty")
        self.assertIsNone(m["dice"])
    def test_nonbinary(self):
        with self.assertRaises(ValueError):
            compare_masks(self.a*2,self.a,[1,1,1])
    def test_invalid_spacing(self):
        with self.assertRaises(ValueError):
            compare_masks(self.a,self.a,[1,0,1])
    def test_partial_overlap(self):
        b=self.a.copy();b[3,2,2]=1
        m=compare_masks(b,self.a,[1,1,1])
        self.assertAlmostEqual(m["dice"],2/3)
        self.assertEqual(m["jaccard"],.5)
        self.assertEqual(m["precision"],.5)
        self.assertEqual(m["relative_volume_error"],1)
