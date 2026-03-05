# TODO

## DSPy/GEPA
- [TODO] Build a small GEPA trainset from successful Terminal-Bench traces
  - Include task instruction, allowed tools, expected behavior, and verifier feedback
- [TODO] Persist compiled GEPA prompts under `state/` or `artifacts/`
  - Feed them through `TINYHARNESS_DSPY_COMPILED_PROMPT_PATH`
- [TODO] Add an integration smoke that runs one sandbox with the DSPy prompt path enabled

## Recruitment
- [TODO] Rewrite README-style project narrative for recruiter review
- [TODO] Clean git history into conventional commits without falsified dates
- [TODO] Remove or justify generated egg-info files tracked in source

## Notes
- `dspy==3.2.1` requires `gepa[dspy]==0.0.27`; `gepa==0.1.1` is not compatible with this DSPy release.
