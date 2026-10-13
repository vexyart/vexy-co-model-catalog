# Output files

`vmc` writes all its files below the directory you run it from. The storage root is the current working directory; there is no option to change it on the command line. In Python you can pass another root to `StorageManager` or `ModelCatalog` (see [Python API](python-api.md)).

## Layout

After `vmc fetch openai` and `vmc fetch groq` in an empty directory, the tree looks like this:

```text
.
├── config/
│   ├── json/
│   │   ├── models_groq.json
│   │   └── models_openai.json
│   └── txt/
│       ├── models_groq.txt
│       └── models_openai.txt
├── models/
│   └── extra/
│       └── failed_providers.json
├── analytics/              # usage analytics, one JSON file per run
├── performance/            # performance monitor data
└── .cache/
    └── vexy_model_catalog/ # response cache
```

Directories are created the first time something is written to them.

| Path | Written by | Content |
| --- | --- | --- |
| `config/json/models_<provider>.json` | `fetch` | The provider's response, sorted |
| `config/txt/models_<provider>.txt` | `fetch` | One model ID per line |
| `config/aichat/models_<provider>.yaml` | `fetch --output_config` | aichat client entry |
| `config/codex/models_<provider>.toml` | `fetch --output_config` | codex profiles |
| `config/mods/models_<provider>.yml` | `fetch --output_config` | mods API entry |
| `models/json/models_<provider>.json` | `fetch --legacy_output` | Same as `config/json/…` |
| `models/text/models_<provider>.txt` | `fetch --legacy_output` | Same as `config/txt/…` |
| `models/extra/failed_providers.json` | every fetch | Failure tracker |

## JSON files

The JSON file holds the provider's response after sorting, written with two-space indentation and UTF-8 characters kept as they are.

Sorting depends on the shape of the response:

- `{"data": [...]}` (OpenAI style): the `data` array is sorted by each entry's `id`.
- An object keyed by model ID (for example `litellm`): the keys are sorted.
- A plain array: sorted by each entry's `id`.

Sorting makes two fetches of an unchanged catalog produce identical files, so you can keep them in git and read real changes from the diff.

## TXT files

The TXT file lists model IDs, one per line, in the same order as the JSON file, ending with a newline. For an object keyed by model ID, the key `sample_spec` is left out.

For `litellm` the file starts like this:

```text
1024-x-1024/50-steps/bedrock/amazon.nova-canvas-v1:0
1024-x-1024/50-steps/stability.stable-diffusion-xl-v1
1024-x-1024/gpt-image-1.5
...
```

## How files are written

Every file is written atomically:

1. The content goes to a temporary file in the target directory, named `.<filename>.<random>.tmp`.
2. The temporary file is renamed over the target.

A reader therefore sees either the old file or the new one, never half a file. If the process is killed between the two steps, the temporary file stays behind; `vmc clean --temp` removes such files.

## The failure tracker

`models/extra/failed_providers.json` has one entry per provider that has been fetched:

```json
{
  "litellm": {
    "consecutive_failures": 0,
    "errors": [],
    "failed": false,
    "failure_count": 0,
    "first_success": "2026-10-12T20:46:43.084930+00:00",
    "last_failure": null,
    "last_success": "2026-10-12T20:46:43.834184+00:00",
    "last_success_date": "Mon, 12 Oct 2026 20:46:43 GMT",
    "success_count": 2
  }
}
```

- `failed` is `true` after a failed fetch and `false` after a successful one.
- `errors` keeps the last 10 errors, each with a `timestamp`, the `error` message and the HTTP `status_code`.
- `vmc providers` shows `Failed` for providers whose `failed` is `true`; `vmc fetch --only_successful` skips them.
- `vmc clean --failed` resets the tracker.

## Tool config files

With `--output_config`, `fetch` also tries to write per-provider configuration snippets for three LLM command-line tools. Each snippet includes at most the first 5 model IDs, with `max_input_tokens` set to `8192` and `max_output_tokens` (or `max_tokens`) set to `4096`, because the provider lists do not carry those limits.

=== "aichat (YAML)"

    ```yaml
    groq:
      type: openai
      api_base: ...
      api_key: ${GROQ_API_KEY}
      models:
      - name: <model-id>
        max_input_tokens: 8192
        max_output_tokens: 4096
    ```

    The `type` is `claude` for Anthropic providers and `openai` for all others.

=== "codex (TOML)"

    ```toml
    [profiles.groq-<model-id>]
    name = "groq-<model-id>"
    base_url = "..."
    api_key_env = "GROQ_API_KEY"
    model = "<model-id>"
    max_tokens = 4096
    context_window = 8192
    ```

    The profile name is `<provider>-<model ID>` with `/` and `:` replaced by `-`.

=== "mods (YAML)"

    ```yaml
    apis:
      groq:
        base_url: ...
        api_key: ${GROQ_API_KEY}
        models:
        - name: <model-id>
          provider: groq
          max_tokens: 4096
    ```

Each snippet is validated before it is written. A snippet with validation errors is not written; the run logs a warning such as `Generated aichat config for groq has 1 errors - skipping write`.

!!! warning "Config generation is unreliable in 1.0.x"
    The generators take the base URL from the provider's fixed `base_url` field, which the built-in providers leave empty (they use a base-URL variable instead). In a test run with `vmc fetch litellm --force --output_config`, no snippet was written: the mods generator failed with an internal error and the others did not produce a file. Treat `--output_config` as experimental. The JSON and TXT files are not affected.

The snippets are not merged into your real tool configuration. Earlier documentation described `update-configs` and `generate-configs` commands; they do not exist in this version.

## Legacy layout

`--legacy_output` writes the JSON and TXT files to `models/json/` and `models/text/` instead of `config/json/` and `config/txt/`. The "files exist" check then looks in the legacy directories.

If a tool expects all files flat in one directory, use `link` instead. It creates links (or copies with `--copy_files`) named `models_<provider>.json` and `models_<provider>.txt` in one directory, `models/` by default. See [CLI reference](cli.md#link).
