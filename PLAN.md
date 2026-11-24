---
this_file: PLAN.md
---

# Remaining work

## Reliability and code quality

- Fix 3–4 reproducible test failures and check suite coverage. Start with the CLI help smoke test and the deterministic integrity and utility failures. Rerun the affected tests, then the full suite with the coverage command in `pyproject.toml`.
- Verify aichat, codex and mods config generation end to end. The parser integration tests pass, but the documented `vmc fetch litellm --force --output_config` run wrote no snippets. Repeat in a temporary catalog and validate one generated file per tool.
- Reduce complexity in high-impact CLI and core functions. Use `ruff check --select C901,PLR0912,PLR0913,PLR0915 src` and the affected tests to verify the changes.
- Clean up unused test arguments without losing fixture setup. Review `ruff check --select ARG001,ARG002 tests` and rerun each changed test module.

## Documentation

- Document custom provider setup with a working Python example. Extend the `ProviderConfig` example in `src_docs/md/python-api.md` through an actual fetch and output write. Verify with a controlled provider endpoint.
- Document the configuration API and backup and recovery with a working example. Show how `ConfigManager` updates existing tool files and restores a backup in a temporary catalog. Verify the example on aichat, codex and mods fixtures.
- Generate the API reference from code docstrings. Check what the strict ProperDocs build generates for the catalog, provider, storage and configuration APIs.
- Run documentation examples in CI. Verify the example jobs in an isolated working directory, with fixtures for any network calls.
