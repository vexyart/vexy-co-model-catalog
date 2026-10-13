# Migrating from dump_models.py

`vexy-co-model-catalog` replaces `dump_models.py`, a single script that fetched model lists from the same providers. A copy of that script is kept in the repository as `external/dump_models.py` for reference.

## What stays the same

- **Provider names.** The provider table is the same list of names, types, key variables and base-URL variables.
- **Environment variables.** Keys are read from the same variables (`OPENAI_API_KEY`, `GROQ_API_KEY`, `HUGGINGFACEHUB_API_TOKEN`, `FRIENDLI_TOKEN`, …), and base URLs from the same `<NAME>_API_OPENAI` variables.
- **File names.** Output files are still called `models_<provider>.json` and `models_<provider>.txt`.
- **Re-running.** Both tools skip a provider whose files already exist, unless you pass `--force`.

## What changes

| | `dump_models.py` | `vmc` |
| --- | --- | --- |
| Where files go | The current directory, flat | `config/json/` and `config/txt/` (or `models/json/` and `models/text/` with `--legacy_output`) |
| TOML output | `models_<provider>.toml` next to the others | Not written |
| Failure record | `failed_models.json` in the current directory | `models/extra/failed_providers.json` |
| Skip check | JSON, TXT and TOML files must all exist | JSON and TXT files must both exist |
| Retrying failed providers | `--failed_rescan` | `vmc clean --failed`, then fetch again |
| Concurrency limit | — | `--max_concurrency`, default 8 |

## Migrate step by step

### Prerequisites

- `vmc` is installed. See [Installation](installation.md).
- Your API key variables are exported in the shell you use.

### Steps

1. Change to the directory that holds your old `models_*.json` files.

    ```bash
    cd ~/models
    ```

2. Optional: run the migration assistant. It needs a source checkout of the repository, because the assistant script is not part of the installed package.

    ```bash
    python /path/to/vexy-co-model-catalog/scripts/migrate_from_dump_models.py
    ```

    The assistant looks for `models_*.json`, `models_*.txt` and `models_*.toml` in `.`, `models`, `../models` and `~/models`. It prints how many providers and models it found, which of nine common key variables are set, and a suggested `fetch` command. It changes no files.

3. Fetch fresh catalogs.

    ```bash
    vmc fetch
    ```

4. Compare the new files with the old ones. Because the new files are sorted, sort the old ones the same way before you diff, or compare the TXT lists only:

    ```bash
    diff <(sort models_openai.txt) <(sort config/txt/models_openai.txt)
    ```

5. Point your other tools at the new files. If they need the files flat in one directory, create links there:

    ```bash
    vmc link --target_dir=.
    ```

    This links `./models_<provider>.json` and `./models_<provider>.txt` to the files in `config/`. Existing files with those names are replaced by the links, so keep a copy of the old files first if you still need them.

6. When the new files are in use, archive `dump_models.py` and the old `failed_models.json`.

## The `migrate` command

`vmc migrate` runs the same assistant script. It looks for the script relative to the installed package, which only works when you run from a source checkout; an installed copy prints `Migration script not found`. The `--search_paths` argument is passed to the script as the `SEARCH_PATHS` variable, but the script does not read it in this version, so it always searches the four default locations listed above.
