-- INF02: shared research study infrastructure; no clinical-data upload enabled.
-- All client access is read-only and scoped to the study owner.
begin;
create table public.studies (
 id uuid primary key default gen_random_uuid(),
 owner_id uuid not null references auth.users(id) on delete restrict,
 title text not null check (char_length(title) between 1 and 120),
 source_kind text not null check (source_kind in ('public_reference','authorized_research')),
 modality text not null default 'CT' check (modality = 'CT'),
 status text not null default 'draft' check (status in ('draft','ingested','processing','review_pending','research_ready','failed','archived')),
 created_at timestamptz not null default now()
);
create index studies_owner_idx on public.studies(owner_id);
create table public.study_assets (
 id uuid primary key default gen_random_uuid(),
 study_id uuid not null references public.studies(id) on delete restrict,
 kind text not null check (kind in ('dicom_zip','ct_nifti','mask','glb','manifest','airway_graph','field','quality_report','other')),
 structure text,
 version integer not null default 1 check (version > 0),
 sha256 text not null check (sha256 ~ '^[0-9a-f]{64}$'),
 bytes bigint not null check (bytes > 0),
 object_key text not null unique check (char_length(object_key) <= 500 and object_key not like '%..%'),
 created_at timestamptz not null default now(),
 check (object_key like study_id::text || '/%')
);
create index study_assets_study_idx on public.study_assets(study_id);
create table public.processing_jobs (
 id uuid primary key default gen_random_uuid(),
 study_id uuid not null references public.studies(id) on delete restrict,
 kind text not null check (kind in ('ingest','segment','reconstruct','topology','qa','full_pipeline')),
 status text not null default 'queued' check (status in ('queued','running','succeeded','failed','cancelled')),
 stage text not null default 'queued',
 progress integer not null default 0 check (progress between 0 and 100),
 attempts integer not null default 0 check (attempts >= 0),
 error_code text,
 created_at timestamptz not null default now(),
 started_at timestamptz,
 finished_at timestamptz
);
create index processing_jobs_study_idx on public.processing_jobs(study_id);
create index processing_jobs_queue_idx on public.processing_jobs(status,created_at);
create table public.quality_reviews (
 id uuid primary key default gen_random_uuid(),
 study_id uuid not null references public.studies(id) on delete restrict,
 reviewer_id uuid not null references auth.users(id) on delete restrict,
 outcome text not null check (outcome in ('pending','needs_correction','research_accepted','rejected')),
 evidence jsonb not null default '{}'::jsonb check (jsonb_typeof(evidence) = 'object'),
 created_at timestamptz not null default now()
);
create index quality_reviews_study_idx on public.quality_reviews(study_id);
create index quality_reviews_reviewer_idx on public.quality_reviews(reviewer_id);
create table public.audit_events (
 id uuid primary key default gen_random_uuid(),
 study_id uuid not null references public.studies(id) on delete restrict,
 actor_id uuid references auth.users(id) on delete restrict,
 event_type text not null check (char_length(event_type) between 1 and 80),
 detail jsonb not null default '{}'::jsonb check (jsonb_typeof(detail) = 'object'),
 created_at timestamptz not null default now()
);
create index audit_events_study_idx on public.audit_events(study_id);
create index audit_events_actor_idx on public.audit_events(actor_id);
alter table public.studies enable row level security;
alter table public.study_assets enable row level security;
alter table public.processing_jobs enable row level security;
alter table public.quality_reviews enable row level security;
alter table public.audit_events enable row level security;
revoke all on public.studies,public.study_assets,public.processing_jobs,public.quality_reviews,public.audit_events from public,anon,authenticated;
grant select on public.studies,public.study_assets,public.processing_jobs,public.quality_reviews,public.audit_events to authenticated;
grant select,insert,update,delete on public.studies,public.study_assets,public.processing_jobs,public.quality_reviews,public.audit_events to service_role;
create policy study_owner_read on public.studies for select to authenticated using (owner_id = (select auth.uid()));
create policy asset_owner_read on public.study_assets for select to authenticated using (exists (select 1 from public.studies s where s.id=study_id and s.owner_id=(select auth.uid())));
create policy job_owner_read on public.processing_jobs for select to authenticated using (exists (select 1 from public.studies s where s.id=study_id and s.owner_id=(select auth.uid())));
create policy review_owner_read on public.quality_reviews for select to authenticated using (exists (select 1 from public.studies s where s.id=study_id and s.owner_id=(select auth.uid())));
create policy audit_owner_read on public.audit_events for select to authenticated using (exists (select 1 from public.studies s where s.id=study_id and s.owner_id=(select auth.uid())));
-- Private storage: no client object policies yet; server-side only.
insert into storage.buckets (id,name,public) values ('pulmo-study-assets','pulmo-study-assets',false);
commit;
