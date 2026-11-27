-- Apply after INF07A. Failure does not delete objects or auto-retry.
begin;
create function public.inf08_fail_job(p_job uuid,p_token uuid,p_code text)
returns boolean language plpgsql security invoker set search_path='' as $$
declare sid uuid;
begin
 if p_code is null or p_code not in ('COMPUTATION_FAILED','COMPUTATION_TIMEOUT','PUBLICATION_REVIEW_REQUIRED','WORKER_FAILED') then
 raise exception 'Invalid failure code';end if;
 update public.processing_jobs set status='failed',stage='failed',
 error_code=p_code,finished_at=now(),lease_token=null,lease_expires_at=null
 where id=p_job and lease_token=p_token and status='running'
 and lease_expires_at>clock_timestamp() and kind='full_pipeline'
 returning study_id into sid;
 if not found then return false;end if;
 insert into public.audit_events(study_id,event_type,detail)
 values(sid,'computational_job_failed',jsonb_build_object('job_id',p_job,'error_code',p_code));
 return true;
end $$;
revoke all on function public.inf08_fail_job(uuid,uuid,text) from public,anon,authenticated;
grant execute on function public.inf08_fail_job(uuid,uuid,text) to service_role;
commit;
