"""Local content-addressed computational provenance. No storage or execution service."""
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import uuid

KINDS=frozenset(("source_volume","processed_volume","scalar_field","vector_field","tensor_field","label_mask","surface_mesh","volume_mesh","centerline_graph","transform","quality_report","simulation_result"))
HASH=re.compile(r"[a-f0-9]{64}")
SPATIAL=KINDS-{"quality_report"}
def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode("utf-8")

def file_hash(path):
    with path.open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()

def asset_id(record):
    return hashlib.sha256(canonical(record)).hexdigest()

def checked_file(root,path):
    if not isinstance(path,str) or "\\" in path or ":" in path: raise ValueError("Invalid asset path")
    rel=PurePosixPath(path)
    if rel.is_absolute() or not rel.parts or ".." in rel.parts or rel.as_posix()!=path: raise ValueError("Invalid asset path")
    target=root.joinpath(*rel.parts)
    if target.resolve()!=target or not target.is_file() or target.stat().st_size<=0: raise ValueError("Missing or unsafe asset")
    return target

class Registry:
    def __init__(self,root,study_id):
        self.root=Path(root).resolve(strict=True)
        if not self.root.is_dir(): raise ValueError("Asset directory required")
        self.study_id=str(uuid.UUID(str(study_id)))
        self.records={}
    def add(self,path,kind,*,quantity,units,frame_id,parents=(),operation,version,parameters=None):
        if kind not in KINDS: raise ValueError("Unknown asset kind")
        for value in (quantity,units,operation,version):
            if not isinstance(value,str) or not 0<len(value)<=160: raise ValueError("Explicit asset semantics required")
        if kind in SPATIAL and (not isinstance(frame_id,str) or not 0<len(frame_id)<=160):
            raise ValueError("Spatial asset requires explicit frame identifier")
        if frame_id is not None and (not isinstance(frame_id,str) or not 0<len(frame_id)<=160): raise ValueError("Invalid frame")
        parents=list(parents)
        if len(set(parents))!=len(parents) or any(p not in self.records for p in parents):
            raise ValueError("Parents must be registered, unique assets")
        if kind!="source_volume" and not parents: raise ValueError("Derived asset requires parents")
        if kind=="source_volume" and parents: raise ValueError("Source must be a root")
        if parameters is not None and not isinstance(parameters,dict): raise ValueError("Parameters must be an object")
        file=checked_file(self.root,path)
        record={"study_id":self.study_id,"path":path,"kind":kind,"quantity":quantity,"units":units,
                "frame_id":frame_id,"parents":parents,"operation":operation,"version":version,
                "parameters":json.loads(canonical(parameters or {})),"sha256":file_hash(file),"bytes":file.stat().st_size}
        key=asset_id(record)
        self.records[key]=record
        return key
    def descendants(self,key):
        if key not in self.records: raise KeyError(key)
        affected={key}
        for k,r in self.records.items():
            if any(p in affected for p in r["parents"]): affected.add(k)
        return [k for k in self.records if k in affected and k!=key]
    def verify(self):
        seen=set()
        for key,r in self.records.items():
            if asset_id(r)!=key or r["study_id"]!=self.study_id or not set(r["parents"])<=seen:
                raise ValueError("Invalid dependency identity")
            path=checked_file(self.root,r["path"])
            if path.stat().st_size!=r["bytes"] or file_hash(path)!=r["sha256"]:
                raise ValueError("Asset changed after registration")
            seen.add(key)
    def export(self,filename="computational-assets.json"):
        if filename!="computational-assets.json": raise ValueError("Fixed manifest filename required")
        self.verify()
        manifest={"schema":"pulmo.computational-assets/1","study_id":self.study_id,
                  "assets":[{"asset_id":k,**r} for k,r in self.records.items()],
                  "review":"PENDING","clinical_navigation_allowed":False}
        with (self.root/filename).open("x") as f: json.dump(manifest,f,indent=2,allow_nan=False)
        return manifest
