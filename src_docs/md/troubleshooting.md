# Troubleshooting

Start with these three commands. None of them changes your catalogs.

```bash
vmc version            # which version is running
vmc validate           # are the key variables set?
vmc providers          # which providers are marked Failed?
```

## Errors when starting `vmc`

### `TypeError: CLI.__init__() takes 1 positional argument but 2 were given`

Every command fails with this error in version 1.0.4 and earlier. A wrapper around the CLI class hid its real signature from Fire, so Fire passed the command name to the constructor. Install a build newer than `v1.0.4`; see [Installation](installation.md).

### `ImportError: Using http2=True, but the 'h2' package is not installed`

`fetch` fails with this error in installs that lack the `h2` package. The HTTP client is created with HTTP/2 enabled, which needs `httpx[http2]`. Newer builds declare that dependency. For an older install, add the package to the same environment:

```bash
python -m pip install h2
```

### `ModuleNotFoundError: No module named 'psutil'`

The environment was created from an outdated lock file. Install the package again so its declared dependencies are resolved, or add the module:

```bash
uv pip install psutil
```

### A traceback after `--help`

Fire passes `--help` to the command when it comes straight after it. Put `--` in front:

```bash
vmc fetch -- --help
```

## Errors in arguments

### `AttributeError: 'tuple' object has no attribute 'split'`

You passed several providers without the extra quotes. Fire turns `openai,groq` into a tuple. Quote the list twice:

```bash
vmc fetch --providers='"openai,groq"'
```

### `Warning: Provider '<name>' not found`

The name is not in the provider table. Names are lower case and match the first column of `vmc providers`, for example `togetherai` (not `together`) and `inference` (not `inferencenet`).

## A provider is skipped or dropped

### `Skipping <provider> (files exist, use --force to overwrite)`

Both `config/json/models_<provider>.json` and `config/txt/models_<provider>.txt` already exist. Add `--force`:

```bash
vmc fetch openai --force
```

### `Environment variable <NAME> is not set`

Validation found no API key. The provider is removed from the run before any request is sent. Look up the variable name in [Providers](providers.md) and export it:

```bash
export GROQ_API_KEY="..."
vmc validate groq
```

### `Skipped <n> failed providers`

You used `--only_successful`, and the failure tracker has those providers marked as failed. Reset the tracker and try again:

```bash
vmc clean --failed
vmc fetch
```

## A request fails

### `Authentication failed (401)` or `(403)`

The provider rejected the key. Check that the variable holds the right key for that provider and that the key is still active. The provider is now marked `Failed` in `vmc providers`; a successful fetch clears the mark.

### `Rate limited`

The provider answered `429 Too Many Requests`. Wait and try again, or fetch fewer providers at once with a lower `--max_concurrency`.

### `SECURITY: URL failed security validation for <provider>`

The base URL did not pass the [URL check](providers.md#url-check). The usual causes:

- The URL uses `http://`. Only `lmstudio` may use `http://`, and only with `localhost` or `127.0.0.1`.
- The host name contains `internal`, `private`, `metadata`, `169.254` or `10.`.

Check the effective URL with `vmc providers show <provider>`, then fix the base-URL variable.

### `No base URL configured for provider atlascloud`

The built-in table sets no default base URL for `atlascloud`. Export `ATLASCLOUD_API_OPENAI` yourself; see the note in [Providers](providers.md).

### `Invalid JSON response from <provider>`

The endpoint answered with something other than JSON, often an HTML error page. Check the provider's status page and the base URL.

## Files are not where you expect

`vmc` writes into the directory you run it from. If you ran it from your home directory, the catalogs are in `~/config/json/` and `~/config/txt/`. Pick one directory for the catalogs and always `cd` there first, as in the scheduled-run example in [Configuration](configuration.md).

### `analytics/` and `performance/` folders appear everywhere

They come from usage analytics and the performance monitor, which are on by default. Turn them off:

```bash
export VEXY_ANALYTICS_ENABLED=false
export VEXY_PERFORMANCE_ENABLED=false
```

### `--output_config` writes nothing

Tool config generation does not work reliably in 1.0.x; see the warning in [Output files](output-files.md#tool-config-files). The JSON and TXT catalogs are written regardless.

### `Migration script not found`

`vmc migrate` only works from a source checkout. Run the script directly instead; see [Migrating from dump_models.py](migration.md).

## In-app help mentions commands that do not exist

`vmc help` and some error messages mention `generate-configs`, `backup-configs`, `restore-configs`, and the `fetch` flags `--verbose`, `--timeout` and `--use-cache`. None of these exist in this version. Older copies of the README also mention `update-configs`, which does not exist either. The commands and flags that do exist are listed in [CLI reference](cli.md), and `vmc <command> -- --help` shows the real flags of any command.
