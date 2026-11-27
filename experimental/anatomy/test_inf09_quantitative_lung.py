import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import nibabel as nib
from quantitative_lung import density,paired
from paired_lung_workflow import pull,run
from computational_assets import file_hash

class QuantitativeTests(unittest.TestCase):
    def test_native_density(self):
        x=np.array([-1000,-960,-950,-800,-400,0,100,-1020],dtype=np.float32).reshape(2,2,2)
        r=density(x,np.ones_like(x),.001,phase="inspiration")
        self.assertEqual(r["low_attenuation_fraction"],3/8)
        self.assertEqual(r["two_compartment_valid_fraction"],6/8)
        self.assertAlmostEqual(r["model_tissue_equivalent_volume_ml"]+r["model_air_equivalent_volume_ml"],.006)
    def test_prm_boundaries(self):
        x=np.array([-960,-950,-950,-810,-960,-1001,-800,-800],dtype=np.float32).reshape(2,2,2)
        y=np.array([-900,-857,-856,-700,-800,-900,-900,-900],dtype=np.float32).reshape(x.shape)
        j=np.ones_like(x);j.flat[-1]=-1
        m=np.ones_like(x)
        f,r=paired(x,y,j,m,m,m,.001)
        np.testing.assert_array_equal(f["prm-labels"].ravel(),[3,2,1,4,5,5,4,0])
        self.assertEqual(r["excluded_nonpositive_jacobian_voxels"],1)
        self.assertEqual(r["paired_valid_voxels"],7)
    def test_mass_conserved_compression(self):
        x=np.full((4,4,4),-800,dtype=np.float32)
        y=np.full_like(x,-600);j=np.full_like(x,.5);m=np.ones_like(x)
        f,r=paired(x,y,j,m,m,m,.002)
        np.testing.assert_allclose(f["tissue-volume-residual"],0,atol=1e-7)
        np.testing.assert_allclose(f["air-volume-change-per-inspiration-volume"],.5,atol=1e-7)
        self.assertAlmostEqual(r["relative_integrated_tissue_volume_error"],0)
    def test_support_and_nonconservation(self):
        x=np.full((4,4,4),-800,dtype=np.float32);m=np.ones_like(x)
        sp=m.copy();sp[0]=0
        f,r=paired(x,x,np.full_like(x,.5),m,m,sp,.001)
        self.assertAlmostEqual(r["relative_integrated_tissue_volume_error"],-.5)
        self.assertEqual(r["paired_valid_fraction"],.75)
        self.assertFalse(f["paired-valid-mask"][0].any())
    def test_pull_ras_direction(self):
        x=np.broadcast_to(np.arange(5,dtype=np.float32)[:,None,None],(5,5,5)).copy()
        u=np.zeros(x.shape+(3,),dtype=np.float32);u[...,0]=2
        a=np.diag([2.,1.,1.,1.])
        y,m,s=pull(x,np.ones_like(x,dtype=np.uint8),a,a,u)
        np.testing.assert_allclose(y[:-1],x[1:])
        self.assertFalse(s[-1].any())
        self.assertFalse(m[-1].any())

class WorkflowTests(unittest.TestCase):
    def test_unfiltered_source_bound_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);reg=root/"reg";reg.mkdir()
            shape=(5,5,5);a=np.diag([1.,1.,2.,1.])
            def write(p,data):
                im=nib.Nifti1Image(data,a);im.header.set_xyzt_units("mm");im.set_sform(a,code=1);nib.save(im,str(p))
            fixed=root/"fixed.nii";moving=root/"moving.nii";mask=root/"mask.nii"
            x=np.full(shape,-960,dtype=np.float32);x[2,2,2]=-1020
            write(fixed,x);write(moving,np.full(shape,-900,dtype=np.float32));write(mask,np.ones(shape,dtype=np.uint8))
            write(reg/"displacement-ras-mm.nii",np.zeros(shape+(3,),dtype=np.float32))
            write(reg/"jacobian.nii",np.ones(shape,dtype=np.float32))
            assets={p.name:{"sha256":file_hash(p),"bytes":p.stat().st_size} for p in reg.glob("*.nii")}
            meta={"schema":"pulmo.dense-respiratory-registration/1","direction":"fixed_RAS_to_moving_RAS_mm",
                  "fixed_phase":"inspiration","moving_phase":"expiration","same_patient_operator_attested":True,
                  "fixed_ct_sha256":file_hash(fixed),"moving_ct_sha256":file_hash(moving),"fixed_voxel_to_ras_mm":a.tolist(),
                  "moving_voxel_to_ras_mm":a.tolist(),"shape":list(shape),"assets":assets}
            (reg/"registration.json").write_text(json.dumps(meta))
            args=(fixed,moving,mask,mask,reg,root/"out")
            kw={"noncontrast_HU_attested":True,"parenchymal_masks_attested":True}
            with self.assertRaises(ValueError):run(*args)
            r=run(*args,**kw)
            self.assertEqual(r["quantification"]["prm_classes"]["indeterminate"]["voxels"],1)
            self.assertEqual(r["quantification"]["paired_valid_voxels"],125)
            self.assertEqual(len(r["assets"]),11)
            self.assertFalse(r["clinical_navigation_allowed"])
            write(reg/"jacobian.nii",np.full(shape,.5,dtype=np.float32))
            with self.assertRaisesRegex(ValueError,"asset changed"):
                run(fixed,moving,mask,mask,reg,root/"out2",**kw)
