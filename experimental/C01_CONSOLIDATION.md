# Consolidation baseline — C01

Baseline inspected: 8db6797002b3af7237dccc04b824db96b3f8c2dc. All 55 existing test cases across 20 files ran successfully in an adapted isolated JavaScript harness using the repository source and test bodies. Native Node and browser execution were not performed in that environment (repository cloning could not resolve GitHub). Inline classic scripts in five public HTML entry points passed syntax compilation. No physical Quest or clinical-data testing.

The native release runner now includes the launcher test and five static entrypoint dependency/syntax tests. GitHub Actions configuration invokes the runner on main pushes, pull requests and manual dispatch. Native CI status must be checked independently; static syntax and dependency checks are not browser smoke tests.

Remaining release gates: actual desktop load, mobile launcher fit, headset hand/controller recovery, room and patient placement, MR capability checks, source-derived CT imports, and measured hardware acceptance. Do not interpret passing numerical tests as clinical validation. Production promotion requires explicit review of the candidate.
