# Development

## Repository layout

| Path | Content |
| --- | --- |
| `src/vexy_co_model_catalog/` | The package |
| `src/vexy_co_model_catalog/cli.py` | The `CLI` class behind `vmc` |
| `src/vexy_co_model_catalog/core/provider.py` | Provider table and `ProviderConfig` |
| `src/vexy_co_model_catalog/core/fetcher.py` | `ModelFetcher`, the HTTP side |
| `src/vexy_co_model_catalog/core/storage.py` | `StorageManager`, atomic file writes |
| `src/vexy_co_model_catalog/core/config.py` | aichat, codex and mods snippet generators |
| `tests/` | pytest suite |
| `scripts/` | Helper scripts, including the migration assistant |
| `external/` | The original `dump_models.py` and saved outputs, for comparison |
| `src_docs/` | Documentation source (this site) |
| `docs/` | The built documentation site, served by GitHub Pages |

## Set up an environment

1. Clone the repository and enter it.

    ```bash
    git clone https://github.com/vexyart/vexy-co-model-catalog.git
    cd vexy-co-model-catalog
    ```

2. Create a virtual environment and install the package with the test and development extras.

    ```bash
    uv venv
    uv pip install -e ".[dev,test]"
    ```

## Run the tests

```bash
uv run python -m pytest -n auto
```

`pyproject.toml` sets `-v --durations=10 -p no:briefcase` as default options. To get a short summary, override them:

```bash
uv run python -m pytest -q -o addopts="" -n auto
```

Turn off analytics and the performance monitor while testing so the suite does not write into the repository:

```bash
VEXY_ANALYTICS_ENABLED=false VEXY_PERFORMANCE_ENABLED=false uv run python -m pytest -q -o addopts="" -n auto
```

The suite does not pass in full. A run on the current main branch gave 302 passed, 74 failed and 11 errors. Most failures are in the integrity, monitoring, production-diagnostics and quality-gate tests; `tests/test_cli_smoke.py`, `tests/test_cli_optimization.py` and `tests/test_package.py` pass except `test_help_command`.

## Lint and format

The project uses Ruff. The `pre-commit` configuration runs `ruff --fix`, `ruff-format` and a set of whitespace and file checks.

```bash
uvx ruff check src tests
uvx ruff format --respect-gitignore src tests
```

## Versioning

The version is derived from git tags by `hatch-vcs`. Tags have the form `vX.Y.Z`. At build time the version is written to `src/vexy_co_model_catalog/_version.py`, which is not tracked in git. Between tags the version looks like `1.0.5.dev0+g16e389c9f`: the next patch version, a development counter and the commit.

When `_version.py` is missing (a plain checkout that was never built or installed), `vexy_co_model_catalog.__version__` falls back to the installed distribution's metadata, then to `0.0.0`.

## Build

The root `build.sh` builds the documentation, the package, or both:

```bash
./build.sh            # docs and package
./build.sh docs       # src_docs/ → docs/
./build.sh package    # dist/*.whl and dist/*.tar.gz, checked with twine
```

The source distribution contains the package, the tests, `README.md`, `CHANGELOG.md`, `LICENSE` and `pyproject.toml`. The wheel contains the package only.

## Documentation

The site is built with ProperDocs and the MaterialX theme. Sources are in `src_docs/md/`, the configuration in `src_docs/mkdocs.yml`.

```bash
src_docs/build.sh          # strict build into docs/
src_docs/build.sh serve    # live preview on http://127.0.0.1:8000
```

The build runs in strict mode, so a broken link or a page missing from the navigation stops it. The `Docs` workflow (`.github/workflows/docs.yml`) rebuilds `docs/` on every push to `main` that touches `src_docs/` and commits the result.

## Release

1. Make sure the tests you care about pass and the docs build.
2. Export a PyPI token for `uv publish`, for example `UV_PUBLISH_TOKEN`.
3. Run:

    ```bash
    ./publish.sh
    ```

    `publish.sh` runs `gitnextver`, which commits pending changes, creates the next `vX.Y.Z` tag and pushes. It then builds the package from that tag and uploads it with `uv publish`.

Pushing the tag also starts the `Release` workflow (`.github/workflows/release.yml`). It builds the package, uploads it to PyPI with `skip-existing`, so a version already uploaded by `publish.sh` is not an error, and creates a GitHub release with the files attached.
