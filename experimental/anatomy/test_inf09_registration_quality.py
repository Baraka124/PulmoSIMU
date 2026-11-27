import unittest
import numpy as np
from inverse_consistency import evaluate
from mass_conservation_refinement import refine

class InverseTests(unittest.TestCase):
    def test_translation_with_anisotropy(self):
        u=np.zeros((8,8,8,3),dtype=np.float32);u[...,0]=2
        v=-u
        a=np.diag([2.,1.,1.,1.])
        fields,r=evaluate(u,a,v,a,landmarks=[{"fixed_RAS_mm":[4,3,3],"moving_RAS_mm":[6,3,3]}])
        self.assertEqual(r["fixed_to_moving_to_fixed"]["support_fraction"],.875)
        self.assertEqual(r["fixed_to_moving_to_fixed"]["roundtrip"]["max_mm"],0)
        self.assertEqual(r["landmarks"]["forward_TRE"]["max_mm"],0)
        self.assertFalse(fields["fixed-roundtrip-support"][-1].any())
    def test_inconsistent_reverse(self):
        u=np.zeros((5,5,5,3),dtype=np.float32);u[...,0]=1
        fields,r=evaluate(u,np.eye(4),np.zeros_like(u),np.eye(4))
        self.assertEqual(r["fixed_to_moving_to_fixed"]["roundtrip"]["mean_mm"],1)
    def test_invalid_landmarks(self):
        u=np.zeros((5,5,5,3),dtype=np.float32)
        with self.assertRaises(ValueError):
            evaluate(u,np.eye(4),u,np.eye(4),landmarks=[{"fixed_RAS_mm":[10,0,0],"moving_RAS_mm":[0,0,0]}])

class ConservationTests(unittest.TestCase):
    def fixture(self):
        x=np.full((12,12,12),-800,dtype=np.float32);y=np.full_like(x,-780)
        mi=np.zeros_like(x);mi[3:9,3:9,3:9]=1
        me=np.ones_like(x);u=np.zeros(x.shape+(3,),dtype=np.float32)
        return x,y,mi,me,np.eye(4),np.eye(4),u
    def test_conservation_correction(self):
        u,j,m,r0,r1,r=refine(*self.fixture(),regularization=.00001,max_iterations=20)
        self.assertTrue(r["accepted"])
        self.assertLess(r["refined_mass_mse"],r["baseline_mass_mse"])
        self.assertGreater(j.min(),0)
        self.assertEqual(m.sum(),216)
        self.assertLess(np.mean(r1[m>0]**2),np.mean(r0[m>0]**2))
    def test_exact_baseline_preserved(self):
        x,y,mi,me,a,b,u=self.fixture()
        result,j,m,r0,r1,r=refine(x,x,mi,me,a,b,u,max_iterations=2)
        self.assertFalse(r["accepted"])
        np.testing.assert_array_equal(result,u)
    def test_folding_rejected(self):
        x,y,mi,me,a,b,u=self.fixture()
        u[...,0]=-2*np.indices(x.shape)[0]
        with self.assertRaisesRegex(ValueError,"folding"):refine(x,y,mi,me,a,b,u)

class QualityWorkflowTests(unittest.TestCase):
    def test_manifest_direction_and_output(self):
        import tempfile,json
        from pathlib import Path
        import nibabel as nib
        from computational_assets import file_hash
        from registration_quality_workflow import run
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);shape=(6,6,6);a=np.eye(4)
            def write(path,value):
                im=nib.Nifti1Image(value,a);im.header.set_xyzt_units("mm");im.set_sform(a,code=1);nib.save(im,str(path))
            x=root/"insp.nii";y=root/"exp.nii"
            write(x,np.full(shape,-800,dtype=np.float32));write(y,np.full(shape,-780,dtype=np.float32))
            for name,first,second,phase,other in (("f",x,y,"inspiration","expiration"),("r",y,x,"expiration","inspiration")):
                folder=root/name;folder.mkdir()
                p=folder/"displacement-ras-mm.nii";write(p,np.zeros(shape+(3,),dtype=np.float32))
                m={"schema":"pulmo.dense-respiratory-registration/1","direction":"fixed_RAS_to_moving_RAS_mm",
                   "fixed_phase":phase,"moving_phase":other,"same_patient_operator_attested":True,
                   "fixed_ct_sha256":file_hash(first),"moving_ct_sha256":file_hash(second),"shape":list(shape),
                   "fixed_voxel_to_ras_mm":a.tolist(),"moving_voxel_to_ras_mm":a.tolist(),
                   "assets":{p.name:{"sha256":file_hash(p),"bytes":p.stat().st_size}}}
                (folder/"registration.json").write_text(json.dumps(m))
            r=run(x,y,root/"f",root/"r",root/"out")
            self.assertEqual(len(r["assets"]),4)
            self.assertEqual(r["baseline_consistency"]["fixed_to_moving_to_fixed"]["roundtrip"]["max_mm"],0)
            self.assertFalse(r["clinical_navigation_allowed"])
            with self.assertRaisesRegex(ValueError,"phase mismatch"):
                run(x,y,root/"r",root/"f",root/"wrong")
