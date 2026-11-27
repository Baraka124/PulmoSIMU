import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np
import nibabel as nib
from deformation_mechanics import kinematics

class KinematicsTests(unittest.TestCase):
    def test_rigid_translation(self):
        u=np.zeros((5,6,7,3));u[:]=[2,-3,4]
        f,j,e,r=kinematics(u,np.diag([1.,2.,3.,1.]))
        np.testing.assert_allclose(f,np.broadcast_to(np.eye(3),f.shape))
        np.testing.assert_allclose(j,1)
        np.testing.assert_allclose(e,0)
        self.assertEqual(r["nonpositive_jacobian_fraction"],0)
    def test_anisotropic_rotated_grid(self):
        a=np.array([[0.,-2,0,10],[1,0,0,20],[0,0,3,30],[0,0,0,1]])
        grid=np.moveaxis(np.indices((5,6,7)),0,-1)
        x=nib.affines.apply_affine(a,grid)
        target=np.array([[1.2,.1,0],[0,.8,0],[0,0,1.1]])
        u=x@(target-np.eye(3)).T
        f,j,e,_=kinematics(u,a)
        np.testing.assert_allclose(f,np.broadcast_to(target,f.shape),atol=1e-6)
        np.testing.assert_allclose(j,np.linalg.det(target),atol=1e-6)
        t=(target.T@target-np.eye(3))/2
        np.testing.assert_allclose(e,np.broadcast_to([t[0,0],t[1,1],t[2,2],t[0,1],t[0,2],t[1,2]],e.shape),atol=1e-6)
    def test_rigid_rotation_has_zero_strain(self):
        grid=np.moveaxis(np.indices((5,5,5)),0,-1)
        rot=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        _,j,e,_=kinematics(grid@(rot-np.eye(3)).T,np.eye(4))
        np.testing.assert_allclose(j,1)
        np.testing.assert_allclose(e,0)
    def test_folding_detected(self):
        u=np.zeros((5,5,5,3));u[...,0]=-2*np.indices(u.shape[:3])[0]
        _,j,_,r=kinematics(u,np.eye(4))
        np.testing.assert_allclose(j,-1)
        self.assertEqual(r["nonpositive_jacobian_fraction"],1)

@unittest.skipUnless(importlib.util.find_spec("SimpleITK"),"Optional registration runtime unavailable")
class RegistrationSmoke(unittest.TestCase):
    def test_dense_package(self):
        from respiratory_dense_registration import register
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            x,y,z=np.indices((16,16,16))
            for name,shift in (("fixed",0),("moving",1)):
                data=(-900+600*np.exp(-((x-8-shift)**2+(y-8)**2+(z-8)**2)/15)).astype(np.float32)
                im=nib.Nifti1Image(data,np.diag([1.,1.,2.,1.]));im.header.set_xyzt_units("mm");im.set_sform(im.affine,code=1)
                nib.save(im,str(root/(name+".nii")))
            args=(root/"fixed.nii",root/"moving.nii",root/"out")
            with self.assertRaises(ValueError):register(*args,factors=(2,1),iterations=(1,1))
            m=register(*args,factors=(2,1),iterations=(2,2),same_patient_attested=True)
            self.assertEqual(len(m["levels"]),2)
            self.assertEqual(len(m["assets"]),7)
            u=nib.load(str(root/"out"/"displacement-ras-mm.nii"))
            self.assertEqual(u.shape,(16,16,16,3))
            self.assertTrue(np.isfinite(u.get_fdata()).all())
            self.assertFalse(m["clinical_navigation_allowed"])
