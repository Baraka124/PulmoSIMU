import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from cloud_job_worker import download_source

SID="12345678-1234-4234-8234-123456789abc"
DATA=b"transfer-test-only"

class Response(io.BytesIO):
    status=200
    def getheader(self,name):
        return None

class Connection:
    def __init__(self,data):
        self.data=data
    def request(self,*args,**kwargs):
        pass
    def getresponse(self):
        return Response(self.data)
    def close(self):
        pass

class TestSourceTransfer(unittest.TestCase):
    def setUp(self):
        self.record={"object_key":SID+"/input/source.nii","bytes":len(DATA),"sha256":hashlib.sha256(DATA).hexdigest()}
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.target=Path(self.tmp.name)/"source.nii"
    def transfer(self,data):
        return download_source("sb_secret_test_only",SID,self.record,self.target,lambda *a,**kw:Connection(data))
    def test_valid_source(self):
        self.assertEqual(self.transfer(DATA).read_bytes(),DATA)
    def test_digest_mismatch(self):
        with self.assertRaises(ValueError):
            self.transfer(b"x"*len(DATA))
        self.assertFalse(self.target.exists())
        self.assertEqual(list(self.target.parent.iterdir()),[])
    def test_truncated_source(self):
        with self.assertRaises(ValueError):
            self.transfer(DATA[:-1])
        self.assertFalse(self.target.exists())
    def test_oversized_response(self):
        with self.assertRaises(ValueError):
            self.transfer(DATA+b"x")
        self.assertFalse(self.target.exists())
    def test_foreign_study(self):
        self.record["object_key"]="foreign/input/source.nii"
        with self.assertRaises(ValueError):
            self.transfer(DATA)
    def test_no_overwrite(self):
        self.target.write_bytes(b"existing")
        with self.assertRaises(FileExistsError):
            self.transfer(DATA)
        self.assertEqual(self.target.read_bytes(),b"existing")
