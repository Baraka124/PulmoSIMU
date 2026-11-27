-- INF07A: additive queue primitives, to be reviewed and tested before deployment.
begin;
alter table public.processing_jobs
 add column if not exists lease_token uuid,
 add column if not exists lease_expires_at timestamptz;
-- Existing duplicate active jobs must be resolved before applying this migration.
create unique index if not exists inf07_one_active_full_pipeline
 on public.processing_jobs(study_id)
 where kind='full_pipeline' and status in ('queued','running');

create function public.inf07_claim_job()
returns setof public.processing_jobs
language sql security invoker set search_path = '' as $$
 with candidate as (
 select j.id from public.processing_jobs j
 join public.studies s on s.id=j.study_id
 where j.kind='full_pipeline' and j.status='queued'
 and s.source_kind='public_reference' and s.status='ingested'
 and exists (select 1 from public.study_assets a where a.study_id=s.id and a.kind='ct_nifti')
 order by j.created_at,j.id
 for update of j skip locked limit 1
 )
 update public.processing_jobs j set status='running',stage='claimed',
 attempts=j.attempts+1,started_at=now(),lease_token=gen_random_uuid(),
 lease_expires_at=now()+interval '15 minutes'
 from candidate c where j.id=c.id returning j.*;
$$;

create function public.inf07_heartbeat(p_job uuid,p_token uuid,p_stage text,p_progress integer)
returns boolean language sql security invoker set search_path='' as $$
 with changed as (
 update public.processing_jobs set stage=p_stage,progress=p_progress,
 lease_expires_at=now()+interval '15 minutes'
 where id=p_job and lease_token=p_token and status='running'
 and lease_expires_at>now() and p_progress between progress and 99
 and p_stage in ('claimed','download','validate','segment','reconstruct','qa','publish')
 returning id
 ) select exists(select 1 from changed);
$$;

-- Expiration fails closed: no automatic retry while publication outcome is uncertain.
create function public.inf07_expire_jobs()
returns bigint language sql security invoker set search_path='' as $$
 with changed as (
 update public.processing_jobs set status='failed',stage='lease_expired',
 error_code='LEASE_EXPIRED_REVIEW_REQUIRED',finished_at=now(),
 lease_token=null,lease_expires_at=null
 where kind='full_pipeline' and status='running' and lease_expires_at<=now()
 returning id
 ) select count(*) from changed;
$$;

revoke all on function public.inf07_claim_job(),public.inf07_heartbeat(uuid,uuid,text,integer),public.inf07_expire_jobs() from public,anon,authenticated;
grant execute on function public.inf07_claim_job(),public.inf07_heartbeat(uuid,uuid,text,integer),public.inf07_expire_jobs() to service_role;
-- Lease tokens must not be readable through the existing owner-facing table grant.
revoke select on public.processing_jobs from authenticated;
grant select (id,study_id,kind,status,stage,progress,attempts,error_code,created_at,started_at,finished_at) on public.processing_jobs to authenticated;
commit;
