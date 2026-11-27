begin;
create function public.inf07_claim_approved_job(p_study uuid)
returns setof public.processing_jobs
language sql security invoker set search_path='' as $$
 with candidate as (
 select j.id from public.processing_jobs j join public.studies s on s.id=j.study_id
 where s.id=p_study and s.source_kind='public_reference' and s.status='ingested'
 and j.kind='full_pipeline' and j.status='queued'
 and (select count(*) from public.study_assets a where a.study_id=s.id)=1
 and exists(select 1 from public.study_assets a where a.study_id=s.id and a.kind='ct_nifti'
 and a.object_key=s.id::text||'/v1/source.nii' and a.bytes<=50000000)
 order by j.created_at,j.id for update of j skip locked limit 1
 )
 update public.processing_jobs j set status='running',stage='claimed',attempts=j.attempts+1,
 started_at=now(),lease_token=gen_random_uuid(),lease_expires_at=now()+interval '15 minutes'
 from candidate c where j.id=c.id returning j.*;
$$;
revoke all on function public.inf07_claim_approved_job(uuid) from public,anon,authenticated;
grant execute on function public.inf07_claim_approved_job(uuid) to service_role;
commit;
