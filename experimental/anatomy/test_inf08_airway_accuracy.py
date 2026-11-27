import unittest
import numpy as np
from airway_topology_accuracy import compare_airways

class AirwayAccuracyTests(unittest.TestCase):
    def setUp(self):
        self.mask=np.zeros((15,15,15),dtype=np.uint8)
        self.mask[7,7,3:12]=1
    def test_identity(self):
        m=compare_airways(self.mask,self.mask,[1,1,2])
        self.assertEqual(m["reference_centerline_coverage"],1)
        self.assertEqual(m["predicted_centerline_precision"],1)
        self.assertEqual(m["matched_radius_mean_absolute_error_mm"],0)
    def test_empty(self):
        m=compare_airways(np.zeros_like(self.mask),self.mask,[1,1,1])
        self.assertEqual(m["reference_centerline_coverage"],0)
        self.assertIsNone(m["predicted_centerline_precision"])
        self.assertIsNone(m["matched_radius_mean_absolute_error_mm"])
    def test_displaced(self):
        m=compare_airways(np.roll(self.mask,4,axis=0),self.mask,[1,1,1],1)
        self.assertEqual(m["reference_centerline_coverage"],0)
        self.assertEqual(m["predicted_centerline_precision"],0)
    def test_fragment(self):
        p=self.mask.copy();p[7,7,7]=0
        m=compare_airways(p,self.mask,[1,1,1])
        self.assertEqual(m["prediction"]["components"],2)
        self.assertEqual(m["reference"]["components"],1)
    def test_invalid(self):
        with self.assertRaises(ValueError):
            compare_airways(self.mask,self.mask,[1,0,1])
