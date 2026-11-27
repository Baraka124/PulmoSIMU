-- INF05A: owner-only reference retrieval; publisher is server-only.
begin;
create policy pulmo_reference_asset_read on storage.objects for select to authenticated
using (bucket_id='pulmo-study-assets' and exists (
 select 1 from public.study_assets a join public.studies s on s.id=a.study_id
 where a.object_key=storage.objects.name and s.owner_id=(select auth.uid())
 and s.source_kind='public_reference' and s.status in ('review_pending','research_ready')
));
create function public.publish_reference_package(p_study uuid,p_assets jsonb,p_provenance jsonb)
returns void language plpgsql security invoker set search_path='' as $$
declare a jsonb; prefix text := p_study::text||'/v1/'; n integer;
begin
 if jsonb_typeof(p_assets) is distinct from 'array' or jsonb_array_length(p_assets) not between 9 and 10 then raise exception 'Invalid asset inventory'; end if;
 if jsonb_typeof(p_provenance) is distinct from 'object' or coalesce(p_provenance->>'source_url','') not like 'https://%' or length(coalesce(p_provenance->>'license','')) not between 1 and 200 or p_provenance->>'redistribution_confirmed' is distinct from 'true' then raise exception 'Reference provenance required'; end if;
 perform 1 from public.studies where id=p_study and source_kind='public_reference' and status='draft' for update;
 if not found then raise exception 'Draft reference study required'; end if;
 if exists(select 1 from public.study_assets where study_id=p_study) then raise exception 'Study already has assets'; end if;
 select count(distinct value->>'path') into n from jsonb_array_elements(p_assets);
 if n<>jsonb_array_length(p_assets) then raise exception 'Duplicate asset paths'; end if;
 if exists(select 1 from unnest(array['source.nii','airway-mask/mask.raw','airway-mask/mask.json','airway-mask/topology.json','airway-mask/airway-skeleton.json','airway-surface/airway.glb','airway-surface/reconstruction.json','airway-surface/fidelity.json','study-package.json']) as r(path)
 where not exists(select 1 from jsonb_array_elements(p_assets) v where v->>'path'=r.path)) then raise exception 'Required package assets missing'; end if;
 for a in select value from jsonb_array_elements(p_assets) loop
  if coalesce(a->>'path','') not in ('source.nii','source.ingest.json','airway-mask/mask.raw','airway-mask/mask.json','airway-mask/topology.json','airway-mask/airway-skeleton.json','airway-surface/airway.glb','airway-surface/reconstruction.json','airway-surface/fidelity.json','study-package.json') or a->>'object_key' is distinct from prefix||(a->>'path') then raise exception 'Invalid asset key'; end if;
  if not exists(select 1 from storage.objects o where o.bucket_id='pulmo-study-assets' and o.name=a->>'object_key' and (o.metadata->>'size')::bigint=(a->>'bytes')::bigint) then raise exception 'Storage object missing or size mismatch'; end if;
  insert into public.study_assets(study_id,kind,structure,version,sha256,bytes,object_key)
  values(p_study,a->>'kind',a->>'structure',1,a->>'sha256',(a->>'bytes')::bigint,a->>'object_key');
 end loop;
 update public.studies set status='review_pending' where id=p_study;
 insert into public.audit_events(study_id,event_type,detail) values(p_study,'reference_package_published',p_provenance);
end $$;
revoke all on function public.publish_reference_package(uuid,jsonb,jsonb) from public,anon,authenticated;
grant execute on function public.publish_reference_package(uuid,jsonb,jsonb) to service_role;
commit;
