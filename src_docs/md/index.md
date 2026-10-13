# vexy-co-model-catalog

`vexy-co-model-catalog` downloads the list of available models from AI providers, writes each list to disk in a stable, sorted form, and keeps track of which providers answered and which failed. It ships a command-line tool, `vmc` (also installed as `vexy-model-catalog`), and a Python package, `vexy_co_model_catalog`.

## What goes in, what comes out

You give the tool a provider name, for example `openai`, and an API key in the environment variable that provider expects, for example `OPENAI_API_KEY`.

The tool then:

1. Looks up the provider in its built-in table of 40 providers. The table holds the provider type, the API key variable and the variable that holds the base URL.
2. Sends one `GET` request to `<base URL>/models` with the right authentication header (`Authorization: Bearer …` for OpenAI-compatible APIs, `x-api-key` for Anthropic).
3. Sorts the response by model ID.
4. Writes two files into the current directory:
    - `config/json/models_openai.json` — the full response, sorted.
    - `config/txt/models_openai.txt` — one model ID per line.
5. Records the result in `models/extra/failed_providers.json`, so a later run can skip providers that are failing.

```bash
export OPENAI_API_KEY="sk-..."
vmc fetch openai
```

```text
✓ openai: <N> models

Fetch Complete: 1 successful, 0 failed, 0 skipped
```

`<N>` is the number of model IDs the provider returned.

## Who it is for

- People who maintain configuration for several LLM command-line tools and want a current list of model IDs per provider.
- Scripts that need a local, versionable snapshot of what each provider offers.
- Users of the older `dump_models.py` script, which this package replaces. The file names (`models_<provider>.json`, `models_<provider>.txt`) are the same.

## Where to go next

| If you want to… | Read |
| --- | --- |
| Install the tool | [Installation](installation.md) |
| Fetch your first catalog | [Quick start](quickstart.md) |
| Look up a command or flag | [CLI reference](cli.md) |
| See which providers exist and which variables they read | [Providers](providers.md) |
| Understand the files the tool writes | [Output files](output-files.md) |
| Control logging, analytics and caching | [Configuration](configuration.md) |
| Call the library from Python | [Python API](python-api.md) |
| Move from `dump_models.py` | [Migrating from dump_models.py](migration.md) |
| Fix a problem | [Troubleshooting](troubleshooting.md) |
| Work on the code | [Development](development.md) |

## Status

The package is versioned from git tags (`v1.0.0` to `v1.0.4` exist) and is not yet published on PyPI. Until it is, install it from the GitHub repository; see [Installation](installation.md).
