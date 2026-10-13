# Python API

The package is `vexy_co_model_catalog`. Importing it is cheap: the top level exposes the version and two classes, and everything else lives in `vexy_co_model_catalog.core.*`.

```python
import vexy_co_model_catalog

print(vexy_co_model_catalog.__version__)
```

The functions and classes on this page are the ones the CLI itself uses. They are not declared a stable public API; pin the package version if you build on them.

## Fetch one provider from Python

This example does what `vmc fetch litellm` does, step by step. It needs no API key.

```python
import asyncio

from vexy_co_model_catalog.core.fetcher import ModelFetcher
from vexy_co_model_catalog.core.provider import get_provider_by_name
from vexy_co_model_catalog.core.storage import StorageManager


async def main() -> None:
    provider = get_provider_by_name("litellm")  # a "url" provider: no API key needed
    async with ModelFetcher(max_concurrency=4) as fetcher:
        data = await fetcher.fetch_provider_models(provider)
        data = fetcher.sort_json_data(data)
        model_ids = fetcher.extract_model_ids(data)

    storage = StorageManager("catalog")  # root directory for output
    storage.write_config_json("models_litellm", data)
    storage.write_config_txt("models_litellm", model_ids)
    print(f"{len(model_ids)} models, first: {model_ids[0]}")


asyncio.run(main())
```

```text
4060 models, first: 1024-x-1024/50-steps/bedrock/amazon.nova-canvas-v1:0
```

The files land in `catalog/config/json/models_litellm.json` and `catalog/config/txt/models_litellm.txt`.

## Providers

Module: `vexy_co_model_catalog.core.provider`

`get_all_providers() -> list[ProviderConfig]`
:   Return the 40 built-in providers. As a side effect it sets each provider's base-URL environment variable to its default, unless the variable is already set.

`get_provider_by_name(name: str) -> ProviderConfig | None`
:   Return one provider, or `None` if the name is unknown.

`ProviderKind`
:   String enum with the values `"openai"`, `"anthropic"` and `"url"`.

`ProviderConfig`
:   A dataclass describing one provider.

    | Field | Type | Default |
    | --- | --- | --- |
    | `name` | `str` | required |
    | `kind` | `ProviderKind` | required |
    | `api_key_env` | `str | None` | `None` |
    | `base_url_env` | `str | None` | `None` |
    | `base_url` | `str | None` | `None` |
    | `headers` | `dict[str, str] | None` | `None` |
    | `timeout` | `float` | `15.0` |
    | `verify_ssl` | `bool` | `True` |

    `get_base_url()` returns the value of `base_url_env` if that variable is set, otherwise `base_url`. `build_models_url()` returns the base URL itself for `url` providers and `<base URL>/models` for the others; it raises `ValueError` when there is no base URL.

You can describe a provider that is not in the table and fetch it the same way:

```python
import os

from vexy_co_model_catalog.core.provider import ProviderConfig, ProviderKind

os.environ["MYHOST_API_KEY"] = "..."
provider = ProviderConfig(
    name="myhost",
    kind=ProviderKind.OPENAI,
    api_key_env="MYHOST_API_KEY",
    base_url="https://llm.example.com/v1",
)
print(provider.build_models_url())  # https://llm.example.com/v1/models
```

## Fetching

Module: `vexy_co_model_catalog.core.fetcher`

`ModelFetcher(max_concurrency: int = 8, timeout: float = 30.0, failure_tracker: FailureTracker | None = None)`
:   An asynchronous HTTP client. Use it as an async context manager so its connection pool is closed.

`await fetcher.fetch_provider_models(provider, use_cache=True)`
:   Request the provider's model list and return the parsed JSON (`dict` or `list`). Retries with exponential back-off. Raises `AuthenticationError` on `401`/`403`, `RateLimitError` on `429` and `FetchError` on other HTTP errors, invalid JSON or a URL that fails the [URL check](providers.md#url-check). When a `failure_tracker` is given, every outcome is recorded in it.

`fetcher.sort_json_data(data)`
:   Return the data sorted as described in [Output files](output-files.md#json-files).

`fetcher.extract_model_ids(data) -> list[str]`
:   Return the model IDs.

## Storage

Module: `vexy_co_model_catalog.core.storage` (also exported from `vexy_co_model_catalog.core`)

`StorageManager(root_path=None)`
:   File storage below `root_path`, or below the current directory when `root_path` is `None`. All writes are atomic.

| Method | Writes to |
| --- | --- |
| `write_config_json(name, data)` | `<root>/config/json/<name>.json` |
| `write_config_txt(name, lines)` | `<root>/config/txt/<name>.txt` |
| `write_json(name, data)` | `<root>/models/json/<name>.json` |
| `write_text(name, lines)` | `<root>/models/text/<name>.txt` |
| `write_yaml(name, data, directory=…)` | `config/<name>.yaml`, `config/aichat/<name>.yaml` or `config/mods/<name>.yml` |
| `write_config_toml(name, data)` | `<root>/config/codex/<name>.toml` |

`read_json(name, directory=…)` reads back JSON from `"json"`, `"extra"` or `"config_json"`. `list_files(directory, pattern="*")` lists files in one of `"json"`, `"text"`, `"extra"`, `"config_json"`, `"config_txt"`, `"config_aichat"`, `"config_codex"`, `"config_mods"`. `get_file_stats()` returns a file count per directory.

## Catalog and registry

Exported from the top-level package.

`ProviderRegistry()`
:   A name-to-`ProviderConfig` mapping. Methods: `register(name, provider)` (raises `ValueError` for an empty name or a non-`ProviderConfig`), `get(name)`, `get_required(name)` (raises `KeyError`), `list_names()`, `list_providers()`, `items()`, `contains(name)`, `remove(name)`, `clear()`, `count()`.

`ModelCatalog(storage=None, registry=None, storage_root=None)`
:   Bundles a `StorageManager` and a `ProviderRegistry`. Missing parts are created. Methods: `get_storage_root()`, `get_provider_count()`, `validate_configuration()`; the last returns `False` when the storage root does not exist yet.

```python
from vexy_co_model_catalog import ModelCatalog
from vexy_co_model_catalog.core.provider import get_all_providers

catalog = ModelCatalog(storage_root="catalog")
for provider in get_all_providers():
    catalog.registry.register(provider.name, provider)

print(catalog.get_provider_count())                      # 40
print(catalog.registry.get_required("groq").api_key_env)  # GROQ_API_KEY
```

`ModelCatalog` does not fetch anything by itself. Use `ModelFetcher` and `StorageManager` as in the first example.

## Failure tracking

Module: `vexy_co_model_catalog.core.failure_tracker`

`FailureTracker(storage: StorageManager)`
:   Reads and writes `models/extra/failed_providers.json` below the storage root. Methods include `is_provider_failed(name)`, `get_provider_failure_info(name)`, `get_failed_providers()`, `get_failure_summary()`, `reset_provider_failures(name)` and `reset_all_failures()`.

```python
from vexy_co_model_catalog.core.failure_tracker import FailureTracker
from vexy_co_model_catalog.core.storage import StorageManager

tracker = FailureTracker(StorageManager("catalog"))
print(tracker.get_failure_summary()["currently_failed"])
```
