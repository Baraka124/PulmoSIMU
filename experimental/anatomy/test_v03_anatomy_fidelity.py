"""Numerical fixtures only, not patient data."""
import tempfile
import unittest
import struct
import json
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage
from anatomy_fidelity import reconstruct,make_glb,landmark_evidence,section
class FidelityTests(unittest.TestCase):
    def test_metric_surface(self):
        data=np.zeros((12,12,12),dtype=np.uint8);data[3:9,3:9,3:9]=1
        a=np.diag([.5,1.,2.,1.]);v,n,f,t,q=reconstruct(data,np.zeros(3),a)
        self.assertEqual(q["boundary_edges"],0);self.assertEqual(q["nonmanifold_edges"],0)
        self.assertGreater(q["enclosed_mesh_volume_ml"],0)
        self.assertLess(abs(q["enclosed_mesh_volume_ml"]-.216),.04)
        self.assertTrue(np.allclose(np.linalg.norm(n,axis=1),1,atol=1e-5))
        glb=make_glb(v,n,f);self.assertEqual(struct.unpack("<4sII",glb[:12]),(b"glTF",2,len(glb)))
        length,kind=struct.unpack("<I4s",glb[12:20]);self.assertEqual(kind,b"JSON")
        doc=json.loads(glb[20:20+length]);self.assertEqual(doc["accessors"][2]["count"],f.size)
    def test_landmarks_and_section(self):
        data=np.zeros((30,30,30),dtype=np.uint8);data[10:20,10:20,4:26]=1
        labels,_=ndimage.label(data)
        names=("oral_cavity","pharynx","larynx","trachea","distal_airway")
        spec={"landmarks":{k:[15,15,5+i*4] for i,k in enumerate(names)}}
        self.assertTrue(landmark_evidence(spec,labels,np.zeros(3),data,np.eye(4))["same_component"])
        self.assertFalse(landmark_evidence(None,labels,np.zeros(3),data,np.eye(4))["same_component"])
        q=section(data,np.eye(4),{"center_voxel":[15,15,15],"normal_ras":[0,0,1],"radius_mm":12,"step_mm":.5})
        self.assertFalse(q["truncated"]);self.assertAlmostEqual(q["sampled_area_mm2"],100,delta=2)
        q=section(data,np.eye(4),{"center_voxel":[15,15,15],"normal_ras":[0,0,1],"radius_mm":3,"step_mm":.5})
        self.assertTrue(q["truncated"]);self.assertIsNone(q["sampled_area_mm2"])
if __name__=="__main__":unittest.main()
