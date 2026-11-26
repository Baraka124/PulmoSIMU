#!/usr/bin/env python3
"""H01: strict local classic single-frame CT DICOM to uncompressed NIfTI.
Requires numpy, pydicom and nibabel. No DICOM identifiers copied to NIfTI.
Input must already be authorized/de-identified; pixels may retain identifiers.
"""
import argparse
from pathlib import Path
import hashlib
import json
import numpy as np
import pydicom
import nibabel as nib

def convert(folder, output):
    output=Path(output)
    if output.suffix != ".nii" or output.exists():
        raise ValueError("Choose a new uncompressed .nii destination")
    sidecar=output.with_suffix(".ingest.json")
    if sidecar.exists():
        raise ValueError("Sidecar already exists")
    records=[]
    for path in sorted(Path(folder).rglob("*")):
        if not path.is_file():
            continue
        try:
            ds=pydicom.dcmread(path)
        except pydicom.errors.InvalidDicomError:
            continue
        if getattr(ds,"Modality",None)!="CT":
            continue
        if str(getattr(ds,"SOPClassUID",""))!="1.2.840.10008.5.1.4.1.1.2" or int(getattr(ds,"NumberOfFrames",1))!=1:
            raise ValueError("Only classic single-frame CT is supported")
        if str(getattr(ds,"BurnedInAnnotation","")).upper()=="YES":
            raise ValueError("Burned-in annotation flagged")
        if "LOCALIZER" in [str(x).upper() for x in getattr(ds,"ImageType",[])]:
            raise ValueError("Localizer mixed into CT input")
        records.append(ds)
    if len(records)<2:
        raise ValueError("At least two CT slices required")
    if len({(str(d.StudyInstanceUID),str(d.SeriesInstanceUID),str(d.FrameOfReferenceUID)) for d in records})!=1:
        raise ValueError("Mixed study, series or frame of reference")
    if len({str(d.SOPInstanceUID) for d in records})!=len(records):
        raise ValueError("Duplicate SOP instance")
    d=records[0]
    orient=np.asarray(d.ImageOrientationPatient,dtype=float)
    spacing=np.asarray(d.PixelSpacing,dtype=float)
    if orient.shape!=(6,) or spacing.shape!=(2,) or not np.isfinite(orient).all() or not np.isfinite(spacing).all() or np.any(spacing<=0):
        raise ValueError("Invalid orientation or spacing")
    u,v=orient[:3],orient[3:]
    if abs(np.linalg.norm(u)-1)>1e-4 or abs(np.linalg.norm(v)-1)>1e-4 or abs(u@v)>1e-4:
        raise ValueError("Non-orthonormal DICOM direction cosines")
    normal=np.cross(u,v)
    def pos(x):
        p=np.asarray(x.ImagePositionPatient,dtype=float)
        if p.shape!=(3,) or not np.isfinite(p).all(): raise ValueError("Invalid slice position")
        return p
    for x in records:
        if (x.Rows,x.Columns)!=(d.Rows,d.Columns) or not np.allclose(x.ImageOrientationPatient,orient,atol=1e-5,rtol=0) or not np.allclose(x.PixelSpacing,spacing,atol=1e-5,rtol=0):
            raise ValueError("Inconsistent slice geometry")
        if str(getattr(x,"RescaleType","HU")).upper()!="HU" or "RescaleSlope" not in x or "RescaleIntercept" not in x or "ModalityLUTSequence" in x:
            raise ValueError("Explicit linear HU scaling required")
    records.sort(key=lambda x: pos(x)@normal)
    positions=np.array([pos(x) for x in records])
    increments=np.diff(positions,axis=0)
    step=np.median(increments,axis=0)
    distance=step@normal
    if distance<=0.01 or np.any(increments@normal<=0.01) or np.max(np.linalg.norm(increments-step,axis=1))>0.05 or np.linalg.norm(step-distance*normal)>0.05:
        raise ValueError("Duplicate, nonuniform or tilted slice stack; resampling not implemented")
    if np.max(np.linalg.norm(positions-(positions[0]+np.arange(len(records))[:,None]*step),axis=1))>0.05:
        raise ValueError("Stack positions do not fit one affine")
    if int(d.Rows)*int(d.Columns)*len(records)>200_000_000:
        raise ValueError("Volume exceeds supported local size")
    arrays=[]
    for x in records:
        slope,intercept=float(x.RescaleSlope),float(x.RescaleIntercept)
        if not np.isfinite([slope,intercept]).all() or slope==0: raise ValueError("Invalid HU scaling")
        px=x.pixel_array
        if px.shape!=(d.Rows,d.Columns): raise ValueError("Unexpected pixel array")
        hu=px.astype(np.float32)*slope+intercept
        if not np.isfinite(hu).all(): raise ValueError("Nonfinite HU")
        arrays.append(hu.T)
    volume=np.stack(arrays,axis=2)
    lps=np.eye(4)
    lps[:3,0]=u*spacing[1];lps[:3,1]=v*spacing[0];lps[:3,2]=step;lps[:3,3]=positions[0]
    ras=np.diag([-1.,-1.,1.,1.])@lps
    image=nib.Nifti1Image(volume,ras)
    image.header.set_xyzt_units("mm")
    image.set_sform(ras,code=1);image.set_qform(None,code=0)
    nib.save(image,str(output))
    reloaded=nib.load(str(output))
    if reloaded.shape!=volume.shape or not np.allclose(reloaded.affine,ras,atol=1e-4,rtol=0):
        raise ValueError("Written NIfTI geometry failed roundtrip")
    with output.open("rb") as stream: digest=hashlib.file_digest(stream,"sha256").hexdigest()
    report={"schema":"pulmosimu.dicom-ingest/1","output_sha256":digest,"shape":list(volume.shape),"voxel_to_ras_mm":reloaded.affine.tolist(),"slice_count":len(records),"conversion":"classic_ct_linear_HU_no_resampling","clinical_review":"PENDING","deidentification_certified":False}
    sidecar.write_text(json.dumps(report,indent=2)+"\n")
    return report

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dicom_directory");parser.add_argument("output_nii")
    args=parser.parse_args()
    print(json.dumps(convert(args.dicom_directory,args.output_nii),indent=2))
