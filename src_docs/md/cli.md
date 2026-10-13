# CLI reference

The command is `vmc`; `vexy-model-catalog` and `python -m vexy_co_model_catalog` run the same program. Every command below is a method of the `CLI` class in `src/vexy_co_model_catalog/cli.py`, exposed through [Python Fire](https://github.com/google/python-fire).

## How arguments work

Fire maps method arguments to the command line:

- A positional argument can be given by position or by name: `vmc fetch openai` and `vmc fetch --providers=openai` are the same.
- A boolean flag is switched on by its name alone: `--force`. Switch a flag that defaults to on off with `--no<name>`, for example `--nooutput_txt`.
- Flag names use underscores, as in the source: `--max_concurrency`, `--only_successful`.
- To see the flags of one command without running it, put `--` before `--help`:

    ```bash
    vmc fetch -- --help
    ```

    Without the `--`, Fire passes `--help` to the command, which fails.

!!! warning "Comma-separated lists need extra quotes"
    Fire turns an unquoted `openai,groq` into a tuple, and the commands that take a provider list expect a string. Wrap the list in a second pair of quotes:

    ```bash
    vmc fetch --providers='"openai,groq"'
    vmc validate '"openai,groq"'
    ```

    A single provider name needs no extra quotes.

Running `vmc` with no command prints the list of commands.

## Core commands

### providers

Show the built-in provider table, or one provider.

```bash
vmc providers                 # table of all providers
vmc providers show openai     # one provider
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `action` | `list` | `list`, `show`, `add` or `remove` |
| `name` | `""` | Provider name for `show`, `add`, `remove` |

The table has the columns Name, Type, Status, API Key Env and Base URL. Status is `Failed` when the failure tracker has the provider marked as failed, otherwise `Ready`; a key symbol follows when the API key variable is set.

`add` and `remove` only print `Provider management is currently read-only`. The provider list is defined in code; see [Providers](providers.md).

Aliases: `ls`, `list`.

### fetch

Download model lists and write them to disk.

```bash
vmc fetch                                  # every provider
vmc fetch groq                             # one provider
vmc fetch --providers='"openai,groq"'      # several providers
vmc fetch groq --force                     # overwrite existing files
vmc fetch --legacy_output                  # write to models/json and models/text
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--providers`, `-p` | `""` | Provider names, comma-separated. Empty or `all` means every provider. |
| `--max_concurrency`, `-m` | `8` | Maximum concurrent requests. |
| `--force`, `-f` | off | Fetch even when both output files already exist. |
| `--output_json` | on | Write the JSON response. |
| `--output_txt` | on | Write the list of model IDs. |
| `--output_config` | off | Also generate per-provider config files for aichat, codex and mods. |
| `--only_successful` | off | Skip providers the failure tracker has marked as failed. |
| `--legacy_output`, `-l` | off | Use the `models/json/` and `models/text/` layout instead of `config/json/` and `config/txt/`. |

What happens, in order:

1. The provider names are resolved against the built-in table. Unknown names print `Warning: Provider '<name>' not found`.
2. With `--only_successful`, providers marked as failed are removed.
3. Each provider is validated (API key variable set, base URL present). Invalid providers are listed with guidance and removed from the run.
4. For each remaining provider: if both output files exist and `--force` is off, it is skipped. Otherwise the tool requests the model list, sorts it, extracts the model IDs and writes the files.
5. A summary line reports how many providers succeeded, failed and were skipped.

Failed requests are retried. Authentication errors (`401`, `403`), rate limits (`429`) and other HTTP errors mark the provider as failed in the failure tracker.

Successful responses go into the model cache with a time-to-live of two hours (`openai`, `anthropic`, `groq`) or one hour (all others). The cache keeps its files in `.cache/vexy_model_catalog/`. `vmc cache` reports on it and clears it.

Aliases: `get`, `dl`, `download`, `sync`.

### validate

Check provider configuration without sending requests.

```bash
vmc validate            # all providers
vmc validate openai
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `providers` | `all` | Provider names, comma-separated, or `all`. |

For each provider the output is `Valid`, `Valid with warnings` or `INVALID`, with the issues found and guidance such as where to get a key. A summary gives the totals and the success rate.

Aliases: `check`, `test`, `verify`.

### provider_health

Send a request to each provider's endpoint and classify it as healthy, degraded or unhealthy.

```bash
vmc provider_health groq --timeout=5
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `providers` | `all` | Provider names, comma-separated, or `all`. |
| `timeout` | `10.0` | Request timeout in seconds. |

### stats

Count the files the tool has written and summarise provider reliability.

```bash
vmc stats
```

```text
System Statistics

Storage:
  Config JSON files: 1
  Config TXT files: 1
  Aichat configs: 0
  Codex configs: 0
  Mods configs: 0

Provider Reliability:
  Total tracked: 1
  Currently working: 1
  Currently failed: 0
  Overall success rate: 100.0%
```

Aliases: `st`, `status`.

### clean

Remove files the tool created.

```bash
vmc clean --temp       # leftover .*.tmp files from interrupted writes
vmc clean --configs    # generated models_* files in config/aichat, config/codex, config/mods
vmc clean --failed     # reset the failure tracker
```

With no flag the command prints `Specify what to clean: --temp, --configs, or --failed`. `--configs` does not touch `config/json/` or `config/txt/`.

Aliases: `rm`, `clear`, `remove`.

### link

Create symbolic links (or copies) to the fetched files in one flat directory.

```bash
vmc link                                   # link every provider into models/
vmc link --target_dir=~/models --copy_files
vmc link --providers=openai
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--target_dir`, `-t` | `models` | Directory to put the links or copies in. Created if missing. |
| `--source_dir`, `-s` | `config` | Source layout. `config` reads `config/json/` and `config/txt/`. |
| `--copy_files`, `-c` | off | Copy the files instead of linking. |
| `--providers`, `-p` | `""` | Provider names; empty or `all` means every provider that has files. |

An existing file or link with the same name in the target directory is replaced.

### migrate

Run the assistant that inspects `dump_models.py` output and suggests a migration plan. See [Migrating from dump_models.py](migration.md).

```bash
vmc migrate
```

The assistant is the script `scripts/migrate_from_dump_models.py`, which is not part of the installed package. The command only works from a source checkout; otherwise it prints `Migration script not found`.

### health

Run the built-in health checks: system resources, storage, network connectivity, configuration, dependencies and providers. The result is an overall status of `healthy`, `warning`, `critical` or `unknown`, followed by the issues found per component.

```bash
vmc health
vmc health --detailed      # include system metrics
vmc health --fix           # attempt safe automatic fixes
```

Aliases: `diag`, `diagnostics`, `healthcheck`.

### cache

Inspect or clear the response cache.

```bash
vmc cache                                # statistics (action defaults to stats)
vmc cache invalidate --provider=openai
vmc cache clear --provider=openai
vmc cache clear --clear_all
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `action` | `stats` | `stats`, `clear` or `invalidate` |
| `--provider` | `""` | Provider to target |
| `--clear_all` | off | With `clear`: empty every cache |

### completion

Print a shell completion script.

```bash
vmc completion bash
vmc completion zsh --install    # print installation instructions instead
```

Supported shells: `bash`, `zsh`, `fish`.

## Other commands

These commands exist and run, but they report on the tool's own monitoring subsystems rather than on model catalogs.

| Command | Arguments | What it does |
| --- | --- | --- |
| `version` | — | Print the package version. |
| `help` | — | Print an overview of commands with examples. Some examples in this overview name commands that do not exist in this version (`generate-configs`, `backup-configs`, `restore-configs`). |
| `aliases` | — | Print the table of command aliases. |
| `setup_wizard` | — | Interactive first-run wizard that walks through API key setup. It asks questions, so run it in a terminal. |
| `validate_config` | `config_path`, `tool_name=""` | Check the syntax of an aichat, codex or mods config file. The tool is guessed from the path when `tool_name` is empty. |
| `validate_models` | `file_path`, `provider_name=""` | Validate a saved provider response, such as `config/json/models_openai.json`. |
| `rate_limits` | `provider="all"` | Show the rate limiter's statistics. |
| `show_analytics` | `--show` (on), `--disable` | Show local usage analytics, or disable collection. |
| `performance` | `action="stats"`, `--save`, `--clear` | Show or manage stored performance metrics. Actions: `stats`, `history`, `save`, `clear`. |
| `integrity` | `action="report"`, `--verify_all`, `--repair`, `--cleanup_days=30` | Checksums of tracked files. Actions: `report`, `verify`, `cleanup`. |
| `monitor` | `action="dashboard"`, `--export`, `--start`, `--stop` | Monitoring dashboard. Actions: `dashboard`, `metrics`, `alerts`, `trends`, `export`. |
| `monitoring_status` | — | Show the monitoring system's status. |
| `production_status` | — | Show deployment status. |
| `production_init` | — | Turn on production-mode error handling and logging. |
| `production_readiness` | — | Check deployment readiness. |
| `production_errors` | — | Summarise recorded errors. |
| `production_diagnostics` | `level="standard"`, `--auto_fix` | Diagnostics at level `basic`, `standard`, `enterprise` or `critical`. |
| `production_health_advanced` | `--continuous`, `--interval=300` | Health checks, optionally repeated every `interval` seconds until <kbd>Ctrl</kbd>+<kbd>C</kbd>. |
| `cli_performance`, `optimization_info` | — | Show the CLI's start-up optimisation data. |

## Exit status

A provider that fails or is unknown does not change the exit status: `vmc fetch nosuchprovider` prints a warning and exits with `0`. An unknown command or a malformed argument makes Fire exit with `2`. When you script around `vmc`, check the printed summary or the failure tracker file described in [Output files](output-files.md).
