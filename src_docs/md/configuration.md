# Configuration

`vmc` has no configuration file. It is controlled by environment variables and by the directory you run it from.

## API keys and base URLs

Each provider reads one variable for its API key and one for its base URL. The names are listed in [Providers](providers.md). Set them in your shell profile or export them before a run:

```bash
export OPENAI_API_KEY="sk-..."
export GROQ_API_KEY="..."
```

A provider whose key variable is empty fails validation and is left out of the run. The `litellm` provider needs no key.

## Tool behaviour variables

| Variable | Default | Effect |
| --- | --- | --- |
| `VEXY_ANALYTICS_ENABLED` | `true` | Write a usage record for every run into `analytics/`. Set to `false`, `0` or `no` to turn it off. |
| `VEXY_PERFORMANCE_ENABLED` | `true` | Run the performance monitor, which keeps data in `performance/`. Set to `false` to turn it off; `vmc performance` then prints a notice instead. |
| `VEXY_MEMORY_PROFILING` | `false` | With `true`, the performance monitor also records memory snapshots. |
| `VEXY_DETAILED_PROFILING` | `false` | With `true`, the performance monitor records detailed counters. |
| `VMC_PRODUCTION_MODE` | `false` | With `true`, `yes` or `1`, every run starts production-mode error handling and file logging. |
| `VMC_LOG_DIR` | see below | Directory for production-mode log files. |
| `VMC_LOG_LEVEL` | `INFO` | Console log level in production mode. |

Values are compared case-insensitively. `VEXY_ANALYTICS_ENABLED` and `VEXY_PERFORMANCE_ENABLED` accept `true`, `1` or `yes` as on.

Without `VMC_LOG_DIR`, production-mode logs go to:

| Platform | Directory |
| --- | --- |
| macOS | `~/Library/Logs/vexy-co-model-catalog` |
| Windows | `~/AppData/Local/vexy-co-model-catalog` |
| Linux and other Unix systems | `~/.local/share/vexy-co-model-catalog/logs` |

## Files written to the working directory

Every `vmc` command creates the CLI object first, and that step already touches the working directory. Running even `vmc providers` in a directory can leave these behind:

| Path | Purpose | How to avoid it |
| --- | --- | --- |
| `analytics/analytics_<date>_<time>_<id>.json` | One usage record per run | `VEXY_ANALYTICS_ENABLED=false` |
| `performance/` | Performance monitor data | `VEXY_PERFORMANCE_ENABLED=false` |
| `.cache/vexy_model_catalog/` | Response cache | — |

A usage record looks like this. It stays on your machine; the analytics module makes no network requests.

```json
{
  "session_id": "2438d552",
  "timestamp": "2026-10-12T20:46:42.853890+00:00",
  "total_duration_ms": 326.6279697418213,
  "python_version": "3.12.12",
  "platform": "Darwin",
  "package_version": "1.0.5.dev0+g16e389c9f.d20261012",
  "commands": [
    {
      "command_name": "fetch",
      "execution_time_ms": 325.76632499694824,
      "success": true,
      "error_type": null,
      "error_message": null,
      "parameter_count": 7,
      "has_force_flag": false,
      "has_config_flags": true,
      "provider_count": 1,
      "memory_usage_mb": 53.234375
    }
  ]
}
```

The catalog files themselves (`config/`, `models/`) are described in [Output files](output-files.md).

## A clean setup for scheduled runs

To refresh catalogs on a schedule without collecting analytics or performance data:

1. Pick a directory for the catalogs, for example `~/model-catalog`.
2. Create a script `~/model-catalog/refresh.sh`:

    ```bash
    #!/usr/bin/env bash
    set -euo pipefail
    cd ~/model-catalog
    export VEXY_ANALYTICS_ENABLED=false
    export VEXY_PERFORMANCE_ENABLED=false
    vmc fetch --force --only_successful
    ```

3. Make it executable with `chmod +x ~/model-catalog/refresh.sh`.
4. Add it to `cron` or another scheduler.

The script needs the API keys in its environment. `cron` does not read your interactive shell profile, so export the keys in the script or in a file the script sources.

`--only_successful` skips providers that failed last time. Run `vmc clean --failed` now and then so those providers get another try.

## Response cache

Successful responses are cached with a time-to-live of two hours for `openai`, `anthropic` and `groq` and one hour for all other providers. Manage the cache with `vmc cache`:

```bash
vmc cache                           # statistics
vmc cache clear --provider=openai   # drop entries for one provider
vmc cache clear --clear_all         # drop everything
```
