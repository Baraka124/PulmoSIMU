"""Trusted local medical image intake. No inference, clinical data upload or PHI certification."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import uuid
import zipfile

MAX_INPUT=2_000_000_000
MAX_VOLUME=750_000_000
MAX_FILES=4096
BLOCK=1024*1024

def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def bounded_copy(src,dst,limit):
    count=0
    with Path(dst).open("xb") as out:
        while True:
            b=src.read(BLOCK)
            if not b: break
            count+=len(b)
            if count>limit: raise ValueError("Decoded input exceeds budget")
            out.write(b)
    return count

def snapshot_directory(source,destination):
    destination.mkdir()
    records=[]; total=0
    for p in sorted(source.rglob("*")):
        if p.is_symlink(): raise ValueError("Source symlinks unsupported")
        if p.is_dir(): continue
        if not p.is_file(): raise ValueError("Unsupported source entry")
        total+=p.stat().st_size
        if total>MAX_INPUT or len(records)>=MAX_FILES: raise ValueError("Directory budget exceeded")
        # Flat randomized-independent names avoid copying identifying filenames into outputs.
        target=destination/("%06d.dcm"%len(records))
        before=digest(p)
        with p.open("rb") as f: bounded_copy(f,target,MAX_INPUT)
        if digest(target)!=before or digest(p)!=before: raise ValueError("Source changed during intake")
        records.append(before)
    if not records: raise ValueError("Empty directory")
    return sorted(records)

def extract_archive(source,destination):
    destination.mkdir()
    with zipfile.ZipFile(source) as z:
        items=z.infolist()
        if len(items)>MAX_FILES: raise ValueError("Archive member budget exceeded")
        seen=set(); total=0; n=0
        for i in items:
            p=PurePosixPath(i.filename)
            mode=(i.external_attr>>16)&0xffff
            if p.is_absolute() or ".." in p.parts or "\\" in i.filename or ":" in i.filename or str(p) in seen or stat.S_ISLNK(mode) or i.flag_bits&1:
                raise ValueError("Unsafe archive member")
            seen.add(str(p))
            if i.is_dir(): continue
            if stat.S_IFMT(mode) not in (0,stat.S_IFREG): raise ValueError("Unsupported archive member")
            total+=i.file_size
            if total>MAX_INPUT: raise ValueError("Archive expanded budget exceeded")
            with z.open(i) as f:
                size=bounded_copy(f,destination/("%06d.dcm"%n),i.file_size)
            if size!=i.file_size: raise ValueError("Archive size mismatch")
            n+=1
        if not n: raise ValueError("Empty archive")

def identify(source):
    p=Path(source)
    if p.is_symlink(): raise ValueError("Source symlinks unsupported")
    if p.is_dir(): return "dicom_directory_candidate"
    if not p.is_file() or not 0<p.stat().st_size<=MAX_INPUT: raise ValueError("Invalid source file")
    if zipfile.is_zipfile(p): return "dicom_zip_candidate"
    with p.open("rb") as f: h=f.read(352)
    if h[:2]==b"\x1f\x8b": return "gzip_nifti_candidate"
    if h[344:348]==b"n+1\x00" or h[4:12]==b"n+2\x00\r\n\x1a\n": return "nifti_candidate"
    if h.startswith(b"NRRD"): return "nrrd_not_supported"
    if h[128:132]==b"DICM": return "single_dicom_not_supported"
    raise ValueError("Unrecognized or unsupported input")

def normalize(source,output,study_id,*,modality,quantity=None):
    """Normalize classic CT only; for NIfTI require explicit operator HU declaration."""
    import numpy as np
    import nibabel as nib
    from dicom_ingest import convert
    from study_worker import validate_ct
    sid=str(uuid.UUID(str(study_id)))
    source=Path(source)
    kind=identify(source)
    source=source.resolve(strict=True); output=Path(output).resolve()
    if modality!="CT": raise ValueError("Only CT normalization implemented")
    if output.exists() or source==output or (source.is_dir() and source in output.parents): raise ValueError("New output outside input required")
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".intake-",dir=output.parent) as tmp:
        root=Path(tmp); stage=root/"package";stage.mkdir(); ct=stage/"source.nii"
        original=None; members=None
        if kind=="dicom_directory_candidate":
            members=snapshot_directory(source,root/"dicom")
            convert(root/"dicom",ct)
        elif kind=="dicom_zip_candidate":
            original=digest(source)
            extract_archive(source,root/"dicom")
            convert(root/"dicom",ct)
        elif kind in ("gzip_nifti_candidate","nifti_candidate"):
            if quantity!="HU": raise ValueError("Explicit operator declaration of HU required for NIfTI")
            original=digest(source)
            opener=gzip.open if kind=="gzip_nifti_candidate" else open
            with opener(source,"rb") as f: bounded_copy(f,ct,MAX_VOLUME)
        else: raise ValueError("Recognized format has no normalizer yet")
        if original is not None and digest(source)!=original: raise ValueError("Source changed during intake")
        image=validate_ct(ct)
        s,sc=image.get_sform(coded=True);q,qc=image.get_qform(coded=True)
        if qc and not np.allclose(s,q,rtol=0,atol=1e-4): raise ValueError("Conflicting NIfTI frames")
        # Read slices, avoiding a second whole-volume allocation.
        lo=math.inf; hi=-math.inf
        for k in range(image.shape[2]):
            a=np.asarray(image.dataobj[:,:,k])
            if not np.isfinite(a).all(): raise ValueError("Nonfinite calibrated intensity")
            lo=min(lo,float(a.min()));hi=max(hi,float(a.max()))
        report={"schema":"pulmo.imaging-intake/1","study_id":sid,"source_kind":kind,"original_sha256":original,
                "directory_member_sha256":members,"modality":"CT","quantity":"HU",
                "quantity_evidence":"operator_declaration" if "nifti" in kind else "DICOM_linear_rescale",
                "normalized":{"path":"source.nii","sha256":digest(ct),"bytes":ct.stat().st_size,"shape":list(image.shape),
                "voxel_to_ras_mm":s.tolist(),"spacing_mm":np.linalg.norm(s[:3,:3],axis=0).tolist(),"intensity_range":[lo,hi]},
                "deidentification_certified":False,"clinical_navigation_allowed":False,"review":"PENDING"}
        (stage/"intake.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        if output.exists(): raise ValueError("Output appeared during intake")
        stage.rename(output)
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source");p.add_argument("output");p.add_argument("--study-id",required=True)
    p.add_argument("--modality",required=True);p.add_argument("--quantity",choices=("HU",))
    a=p.parse_args();print(json.dumps(normalize(a.source,a.output,a.study_id,modality=a.modality,quantity=a.quantity),indent=2))
