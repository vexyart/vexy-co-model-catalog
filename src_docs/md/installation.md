# Installation

## Requirements

- Python 3.10 or newer (`requires-python = ">=3.10"` in `pyproject.toml`).
- Network access to the providers you want to query.
- An API key for each provider that needs one. See [Providers](providers.md) for the variable names.

The package depends on `fire`, `httpx[http2]`, `loguru`, `psutil`, `pyyaml`, `rich`, `tomli` and `tomli-w`. The installer pulls them in.

## Install from GitHub

The package is not yet on PyPI. Install it straight from the repository.

=== "uv"

    ```bash
    uv tool install git+https://github.com/vexyart/vexy-co-model-catalog
    ```

=== "pip"

    ```bash
    python -m pip install git+https://github.com/vexyart/vexy-co-model-catalog
    ```

Both commands build the package from the latest commit. The version string comes from the nearest git tag, so a build made after `v1.0.4` reports a version such as `1.0.5.dev0+g<commit>`.

Once the package is on PyPI, the shorter form will work:

```bash
uv tool install vexy-co-model-catalog
```

## Install from a local checkout

Use this when you want to change the code or run the tests.

1. Clone the repository.

    ```bash
    git clone https://github.com/vexyart/vexy-co-model-catalog.git
    cd vexy-co-model-catalog
    ```

2. Create a virtual environment and install the package in editable mode with the test extras.

    ```bash
    uv venv
    uv pip install -e ".[test]"
    ```

3. Check that the command runs.

    ```bash
    uv run vmc version
    ```

    The first line of the output names the package and its version, for example `vexy-co-model-catalog version 1.0.5.dev0+g16e389c9f`.

## Commands that get installed

The package installs two console scripts. Both run the same program.

| Command | Entry point |
| --- | --- |
| `vmc` | `vexy_co_model_catalog.cli:main` |
| `vexy-model-catalog` | `vexy_co_model_catalog.cli:main` |

You can also run the module directly:

```bash
python -m vexy_co_model_catalog providers
```

!!! note "Older examples"
    Earlier versions of the README used `vexy-co-model-catalog` as the command name. No script with that name is installed; use `vmc` or `vexy-model-catalog`.

## Verify the installation

1. List the providers. This needs no network access.

    ```bash
    vmc providers
    ```

    The result is a table with 40 rows and the line `Total: 40 providers configured`.

2. Check the configuration of one provider. This reads environment variables only.

    ```bash
    vmc validate openai
    ```

    The result ends with a `Validation Summary` block that counts valid and invalid providers.

## Uninstall

=== "uv"

    ```bash
    uv tool uninstall vexy-co-model-catalog
    ```

=== "pip"

    ```bash
    python -m pip uninstall vexy-co-model-catalog
    ```

Uninstalling does not remove files the tool wrote into your working directories (`config/`, `models/`, `analytics/`, `performance/`, `.cache/vexy_model_catalog/`). Delete those by hand if you no longer need them. [Configuration](configuration.md) lists what each one holds.
