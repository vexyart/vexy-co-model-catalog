# Quick start

This page follows one provider through a complete run: check it, fetch it, look at the files, fetch it again, and link the results into a second directory.

The example uses the `litellm` provider because it needs no API key. Its "API" is a public JSON file in the LiteLLM repository that lists model prices and context windows. The steps are the same for any other provider once its API key is set.

## Prerequisites

- `vmc` is installed. See [Installation](installation.md).
- You have network access to `raw.githubusercontent.com`.

## 1. Make a working directory

`vmc` writes everything relative to the directory you run it from. Start in an empty one so you can see what appears.

```bash
mkdir catalog-demo
cd catalog-demo
```

## 2. Look at the provider

```bash
vmc providers show litellm
```

```text
Provider: litellm
Type: url
API Key Env: None
Base URL: https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json
```

`Type: url` means the base URL is the file to download; the tool does not append `/models`.

## 3. Check the configuration

```bash
vmc validate litellm
```

`validate` reads environment variables only; it sends no requests. For a provider that needs a key, this is where a missing `…_API_KEY` shows up.

## 4. Fetch the catalog

```bash
vmc fetch litellm
```

```text
Fetching from 1 providers...
Using config/ directory structure
Validating provider configurations...
✓ All provider configurations valid
Processing litellm
✓ litellm: 4060 models

Fetch Complete: 1 successful, 0 failed, 0 skipped
```

The count was 4060 when this page was written; it changes as the upstream file changes.

## 5. Look at the files

```bash
find . -type f -path './config/*' -o -type f -path './models/*'
```

```text
./config/json/models_litellm.json
./config/txt/models_litellm.txt
./models/extra/failed_providers.json
```

- `config/json/models_litellm.json` holds the downloaded data. For `litellm` it is an object keyed by model ID, sorted by key:

    ```json
    {
      "1024-x-1024/50-steps/bedrock/amazon.nova-canvas-v1:0": {
        "deprecation_date": "2026-09-30",
        "litellm_provider": "bedrock",
        "max_input_tokens": 2600,
        "mode": "image_generation",
        "output_cost_per_image": 0.06
      },
    ```

- `config/txt/models_litellm.txt` holds one model ID per line, in the same order.
- `models/extra/failed_providers.json` records that `litellm` succeeded. A later `--only_successful` run uses it to skip providers that failed.

The tool also creates `analytics/`, `performance/` and `.cache/vexy_model_catalog/` in the working directory. [Configuration](configuration.md) explains each and how to turn the first two off.

## 6. Run the fetch again

```bash
vmc fetch litellm
```

```text
Processing litellm
Skipping litellm (files exist, use --force to overwrite)

Fetch Complete: 0 successful, 0 failed, 1 skipped
```

A provider is skipped when both its JSON file and its TXT file already exist. Add `--force` to write them again.

## 7. Link the files somewhere else

Some tools expect the files flat in one directory. `link` creates symbolic links to them:

```bash
vmc link
```

```text
✓ litellm

Linked 2 files for 1 providers
```

The links land in `models/` by default:

```text
models/models_litellm.json -> …/catalog-demo/config/json/models_litellm.json
models/models_litellm.txt  -> …/catalog-demo/config/txt/models_litellm.txt
```

Use `--target_dir` to pick another directory and `--copy_files` to copy instead of linking. See [CLI reference](cli.md#link).

## 8. Fetch a provider that needs a key

1. Find the key variable in [Providers](providers.md). For Groq it is `GROQ_API_KEY`.
2. Export the key.

    ```bash
    export GROQ_API_KEY="..."
    ```

3. Fetch.

    ```bash
    vmc fetch groq
    ```

    If the key is missing, the validation step stops the provider before any request is sent:

    ```text
    ✗ groq:
      • Environment variable GROQ_API_KEY is not set
    ...
    No valid providers remaining after validation. Please fix configuration issues.
    ```

## 9. Fetch several providers at once

Pass the names as one comma-separated string. Because of how the command-line parser reads commas, wrap the list in two sets of quotes:

```bash
vmc fetch --providers='"openai,groq,mistral"'
```

Run `vmc fetch` with no provider to fetch every provider in the table. Providers whose key is not set fail validation and are dropped from the run; the others are fetched.
