-- INF07D: lease-bound publication of a first multi-structure package; existing RPC unchanged.
begin;
create function public.inf07_publish_job(p_job uuid,p_token uuid,p_assets jsonb,p_provenance jsonb)
returns void language plpgsql security invoker set search_path='' as $$
declare p_study uuid; source_row public.study_assets%rowtype; a jsonb; prefix text; n integer; k text; expected_kind text; expected_structure text; paths text[]; total bigint;
begin
 select study_id into p_study from public.processing_jobs
 where id=p_job and lease_token=p_token and status='running' and kind='full_pipeline'
 and stage='publish' and lease_expires_at>clock_timestamp() for update;
 if not found then raise exception 'Active publishing lease required';end if;
 prefix:=p_study::text||'/v1/';
 if jsonb_typeof(p_assets) is distinct from 'array' or jsonb_array_length(p_assets) not between 9 and 50 then raise exception 'Invalid multi inventory';end if;
 if jsonb_typeof(p_provenance) is distinct from 'object' or coalesce(p_provenance->>'source_url','') not like 'https://%' or length(coalesce(p_provenance->>'license','')) not between 1 and 200 or p_provenance->>'redistribution_confirmed' is distinct from 'true' then raise exception 'Reference provenance required';end if;
 perform 1 from public.studies where id=p_study and source_kind='public_reference' and status='ingested' for update;
 if not found then raise exception 'Ingested reference study required';end if;
 select * into strict source_row from public.study_assets where study_id=p_study and kind='ct_nifti' for update;
 if exists(select 1 from public.study_assets where study_id=p_study and id<>source_row.id) then raise exception 'Study already has derived assets';end if;
 select array_agg(value->>'path'),count(distinct value->>'path'),sum((value->>'bytes')::bigint) into paths,n,total from jsonb_array_elements(p_assets);
 if n<>jsonb_array_length(p_assets) or total>200262144 or not array['source.nii','pulmonary.glb','reconstruction.json','study-package.json']<@paths then raise exception 'Invalid common inventory';end if;
 foreach k in array array['airway_lumen','pulmonary_artery','pulmonary_vein','RUL','RML','RLL','LUL','LLL','lung'] loop
  select count(*) into n from unnest(paths) p where p=any(array['structures/'||k||'/mask/mask.raw','structures/'||k||'/mask/mask.json','structures/'||k||'/mask/topology.json','structures/'||k||'/surface.glb','structures/'||k||'/surface.json']);
  if n not in (0,5) then raise exception 'Incomplete structure';end if;
 end loop;
 if not exists(select 1 from unnest(paths) p where p like 'structures/%/surface.glb') then raise exception 'No structures';end if;
 if 'structures/airway_lumen/mask/airway-skeleton.json'=any(paths) and not 'structures/airway_lumen/surface.glb'=any(paths) then raise exception 'Orphan graph';end if;
 for a in select value from jsonb_array_elements(p_assets) loop
  select v.kind,v.structure into expected_kind,expected_structure from (values ('source.nii','ct_nifti',null::text),
('pulmonary.glb','glb',null::text),
('reconstruction.json','manifest',null::text),
('study-package.json','manifest',null::text),
('structures/airway_lumen/mask/mask.raw','mask','airway_lumen'::text),
('structures/airway_lumen/mask/mask.json','manifest','airway_lumen'::text),
('structures/airway_lumen/mask/topology.json','quality_report','airway_lumen'::text),
('structures/airway_lumen/surface.glb','glb','airway_lumen'::text),
('structures/airway_lumen/surface.json','manifest','airway_lumen'::text),
('structures/pulmonary_artery/mask/mask.raw','mask','pulmonary_artery'::text),
('structures/pulmonary_artery/mask/mask.json','manifest','pulmonary_artery'::text),
('structures/pulmonary_artery/mask/topology.json','quality_report','pulmonary_artery'::text),
('structures/pulmonary_artery/surface.glb','glb','pulmonary_artery'::text),
('structures/pulmonary_artery/surface.json','manifest','pulmonary_artery'::text),
('structures/pulmonary_vein/mask/mask.raw','mask','pulmonary_vein'::text),
('structures/pulmonary_vein/mask/mask.json','manifest','pulmonary_vein'::text),
('structures/pulmonary_vein/mask/topology.json','quality_report','pulmonary_vein'::text),
('structures/pulmonary_vein/surface.glb','glb','pulmonary_vein'::text),
('structures/pulmonary_vein/surface.json','manifest','pulmonary_vein'::text),
('structures/RUL/mask/mask.raw','mask','RUL'::text),
('structures/RUL/mask/mask.json','manifest','RUL'::text),
('structures/RUL/mask/topology.json','quality_report','RUL'::text),
('structures/RUL/surface.glb','glb','RUL'::text),
('structures/RUL/surface.json','manifest','RUL'::text),
('structures/RML/mask/mask.raw','mask','RML'::text),
('structures/RML/mask/mask.json','manifest','RML'::text),
('structures/RML/mask/topology.json','quality_report','RML'::text),
('structures/RML/surface.glb','glb','RML'::text),
('structures/RML/surface.json','manifest','RML'::text),
('structures/RLL/mask/mask.raw','mask','RLL'::text),
('structures/RLL/mask/mask.json','manifest','RLL'::text),
('structures/RLL/mask/topology.json','quality_report','RLL'::text),
('structures/RLL/surface.glb','glb','RLL'::text),
('structures/RLL/surface.json','manifest','RLL'::text),
('structures/LUL/mask/mask.raw','mask','LUL'::text),
('structures/LUL/mask/mask.json','manifest','LUL'::text),
('structures/LUL/mask/topology.json','quality_report','LUL'::text),
('structures/LUL/surface.glb','glb','LUL'::text),
('structures/LUL/surface.json','manifest','LUL'::text),
('structures/LLL/mask/mask.raw','mask','LLL'::text),
('structures/LLL/mask/mask.json','manifest','LLL'::text),
('structures/LLL/mask/topology.json','quality_report','LLL'::text),
('structures/LLL/surface.glb','glb','LLL'::text),
('structures/LLL/surface.json','manifest','LLL'::text),
('structures/lung/mask/mask.raw','mask','lung'::text),
('structures/lung/mask/mask.json','manifest','lung'::text),
('structures/lung/mask/topology.json','quality_report','lung'::text),
('structures/lung/surface.glb','glb','lung'::text),
('structures/lung/surface.json','manifest','lung'::text),
('structures/airway_lumen/mask/airway-skeleton.json','airway_graph','airway_lumen'::text)) v(path,kind,structure) where v.path=a->>'path';
  if not found or a->>'kind' is distinct from expected_kind or a->>'structure' is distinct from expected_structure or a->>'object_key' is distinct from prefix||(a->>'path') or a->>'version' is distinct from '1' or coalesce((a->>'bytes')::bigint,0) not between 1 and 50000000 or coalesce(a->>'sha256','') !~ '^[0-9a-f]{64}$' then raise exception 'Invalid asset metadata';end if;
  if not exists(select 1 from storage.objects o where o.bucket_id='pulmo-study-assets' and o.name=a->>'object_key' and (o.metadata->>'size')::bigint=(a->>'bytes')::bigint) then raise exception 'Storage object missing or size mismatch';end if;
  if a->>'path'='source.nii' then
   if source_row.sha256 is distinct from a->>'sha256' or source_row.bytes is distinct from (a->>'bytes')::bigint then raise exception 'Registered CT mismatch';end if;
   -- The package requires the canonical source path; never create a second catalogue row.
   if source_row.object_key is distinct from a->>'object_key' then raise exception 'Canonical source path required';end if;
   continue;
  end if;
  insert into public.study_assets(study_id,kind,structure,version,sha256,bytes,object_key) values(p_study,expected_kind,expected_structure,1,a->>'sha256',(a->>'bytes')::bigint,a->>'object_key');
 end loop;
 update public.studies set status='review_pending' where id=p_study;
 update public.processing_jobs set status='succeeded',stage='review_pending',progress=100,
 finished_at=now(),lease_token=null,lease_expires_at=null,error_code=null
 where id=p_job and lease_token=p_token and lease_expires_at>clock_timestamp();
 if not found then raise exception 'Lease expired during publication';end if;
 insert into public.audit_events(study_id,event_type,detail) values(p_study,'multi_reference_job_published',p_provenance||jsonb_build_object('job_id',p_job,'source_sha256',source_row.sha256));
end $$;
revoke all on function public.inf07_publish_job(uuid,uuid,jsonb,jsonb) from public,anon,authenticated;
grant execute on function public.inf07_publish_job(uuid,uuid,jsonb,jsonb) to service_role;
commit;
