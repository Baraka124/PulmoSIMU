import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import uuid
from publish_reference import check_package, REQUIRED

class PublisherTest(unittest.TestCase):
    def test_tampered_asset_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);sid=str(uuid.uuid4());assets=[]
            for rel,(kind,structure) in REQUIRED.items():
                p=root/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b"fixture")
                assets.append(dict(path=rel,kind=kind,structure=structure,version=1,object_key=sid+"/v1/"+rel,bytes=7,sha256=hashlib.sha256(b"fixture").hexdigest()))
            (root/"study-package.json").write_text(json.dumps(dict(schema="pulmo.study-package/1",study_id=sid,status="review_pending",clinical_navigation_allowed=False,source_sha256=assets[0]["sha256"],assets=assets)))
            (root/"source.nii").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError,"digest mismatch"):check_package(root)
