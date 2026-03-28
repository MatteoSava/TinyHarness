# TODO

## DSPy/GEPA
- [TODO] Build a small GEPA trainset from successful Terminal-Bench traces
  - Include task instruction, allowed tools, expected behavior, and verifier feedback
- [DONE] Persist compiled GEPA prompts under `state/` or `artifacts/`
  - Feed them through `TINYHARNESS_DSPY_COMPILED_PROMPT_PATH`
- [TODO] Add an integration smoke that runs one sandbox with the DSPy prompt path enabled

## Public Readiness
- [DONE] Add README with project narrative, quick start, and current limitations
- [DONE] Remove generated egg-info and local MLflow DB files from git
- [TODO] Pin the agent installer script by checksum if the upstream uv installer publishes stable checksums
- [TODO] Produce one passing Terminal-Bench sample run with the compiled GEPA prompt

## Notes
- `dspy==3.2.1` requires `gepa[dspy]==0.0.27`; `gepa==0.1.1` is not compatible with this DSPy release.
