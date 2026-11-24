---
this_file: WORK.md
---

# Current work

- The Pages site uses fly-january 0.2.1 (`9ab88a7`). The strict build passed with zero warnings. All 13 pages kept unbranded chrome, and 1,001 relative links and asset references resolved.
- The release guide now matches the Trusted Publishing workflow. `./build.sh` rebuilt the docs in strict mode with zero docs warnings, and Twine accepted both distributions.
- The latest pytest run collected 399 tests. It had 317 passes, 82 failures and one warning. All 82 failures also occurred on the `8c795bc` baseline. All 12 configuration integration tests passed.
- The housekeeping audit confirmed that the 40 built-in provider names match `dump_models.py`. Ruff reported 36 complexity findings in `src/` and 18 unused-argument findings in `tests/`. The remaining checks are in `PLAN.md`.
