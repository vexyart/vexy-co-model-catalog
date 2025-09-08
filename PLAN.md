---
this_file: PLAN.md
---

# Model Catalog Manager — Detailed SPEC and Plan

## Overview

Goal: Provide a robust, import-safe Python package and CLI to fetch, normalize, and store AI model catalogs from many providers, while keeping minimal surface at import time. The system replaces and extends the functionality of `external/dump_models.py` with a cleaner architecture and better config file integration.

## Analysis of Existing Implementation

Based on `external/dump_models.py` and `external/bak/` artifacts:

### Current Functionality (dump_models.py)
- Fetches models from 45+ AI providers (OpenAI, Anthropic, Groq, etc.)
- Supports 3 provider types: `oai` (OpenAI-compatible), `ant` (Anthropic), `url` (direct JSON)
- Generates 3 file types per provider:
  - `models_<provider>.json` — raw API response
  - `models_<provider>.txt` — simple model ID list
  - `models_<provider>.toml` — Codex-compatible config with profiles
- Special handling for Chutes provider (merges two APIs)
- Failed provider tracking in `failed_models.json`
- 112 files produced total (38 JSON, 38 TXT, 36 TOML)

### Target Config Integration
The new system must integrate with existing config files:
- `config/aichat/config.yaml` — aichat tool configuration
- `config/codex/config.toml` — Codex tool configuration  
- `config/mods/mods.yml` — mods tool configuration

### Enhanced Requirements
1. **Config File Modification**: Update existing tool configs with new providers/models
2. **Per-Provider Config Generation**: Generate tool-specific config files like `config/codex/models_anthropic.toml`
3. **Structured Output Directories**: 
   - JSON files → `config/json/`
   - TXT files → `config/txt/`
   - Tool-specific configs → `config/<tool>/models_<provider>.<ext>`
4. **Provider Coverage**: Match or exceed the 45+ providers from dump_models.py
5. **Compatibility**: Maintain backward compatibility with existing external tool output format



## Architecture

- Core layers
  - utils.exceptions: Typed exception hierarchy for clear failure modes.
  - core.storage: Atomic file I/O for JSON, TOML, TXT, YAML, and auxiliary data; flexible directory layout supporting both `models/` and `config/` outputs.
  - core.provider: ProviderKind enum, Model and ProviderConfig data classes. Encapsulates provider metadata and env overrides. Extended from dump_models.py provider definitions.
  - core.fetcher: Async JSON fetcher with concurrency limit, retries, and basic error mapping (401/403→AuthenticationError, 429→RateLimitError, others→FetchError).
  - core.catalog: Registry for provider instances and a catalog façade to orchestrate fetching and persistence.
  - core.config: Config file parsers and generators for aichat (YAML), codex (TOML), and mods (YAML) formats.

- CLI
  - A Fire-based CLI that keeps imports light. Current commands: `version`, `list_providers` (placeholder).
  - Extended commands: `fetch`, `stats`, `clean`, `providers add/list/remove`, `update-configs` for tool config modification.

- Config Integration
  - Config file readers/writers for existing tool configurations
  - Template-based config generation for each provider
  - Backup and merge strategies for config updates

- Migration from external/dump_models.py
  - Provider definitions imported and enhanced
  - Chutes special handling preserved
  - Failed provider tracking maintained
  - Output format compatibility ensured

## Data and Storage Model

- Directory layout (root-relative):
  - `config/json/` — per-provider JSON payloads (raw API responses)
  - `config/txt/` — per-provider TXT lists (model IDs only)
  - `config/codex/` — Codex-compatible TOML configs (main + per-provider)
  - `config/aichat/` — aichat YAML configs (main + per-provider)  
  - `config/mods/` — mods YAML configs (main + per-provider)
  - `models/extra/` — auxiliary JSON (summary stats, failures, run metadata)

- File naming conventions:
  - Raw data: `config/json/models_{provider}.json`, `config/txt/models_{provider}.txt`
  - Tool configs: `config/{tool}/models_{provider}.{ext}` (e.g. `config/codex/models_openai.toml`)
  - Main configs: `config/{tool}/config.{ext}` (updated in-place with new providers)
  - Extra artifacts: `models/extra/summary_stats.json`, `models/extra/failed_providers.json`, `models/extra/run_{timestamp}.json`

- Legacy compatibility:
  - Optional output to `models/` directory structure for backward compatibility
  - Symlinks or copies to maintain existing tooling integration

## Provider Abstraction

- ProviderConfig: identifies provider kind, URL, headers, and env-based overrides. Extended from dump_models.py with 45+ provider definitions.
- ProviderKind: `openai` (oai), `anthropic` (ant), `url` (direct JSON sources). Maps to dump_models.py provider types.
- Model: normalized metadata for downstream use (id, provider, costs, limits, flags, optional raw payload, tool-specific fields).
- Provider instance: encapsulates how to build requests (base URL, headers), transform raw payload → List[Model], and generate tool configs.

### Provider Migration Strategy
- Import all 45+ providers from dump_models.py PROVIDER_CONFIG and PROVIDER_URL_CONFIG
- Preserve special cases like Chutes provider dual-API handling
- Maintain environment variable patterns for API keys and URLs
- Add support for provider-specific config generation rules

## Fetching Flow

- CLI/Orchestrator resolves set of providers to fetch.
- For each provider:
  - Build endpoint URL (e.g., `${BASE}/models` for OpenAI-like providers, or direct URL for `url` kind).
  - Use ModelFetcher.get_json with small retry/backoff.
  - Optionally merge or enhance data for special providers (e.g., chutes), using side API calls.
  - Sort/stabilize JSON for deterministic outputs.
  - Extract model IDs to generate TXT.
  - Build TOML profiles with name, base URL, context window, max output tokens, and any pricing if present.
- Persist artifacts atomically via StorageManager; update auxiliary `failed_providers` and `summary_stats`.

## Error Handling

- AuthenticationError: missing/invalid credentials; provider marked failed.
- RateLimitError: 429 after retries; provider marked failed (with retry info).
- FetchError: network/HTTP/parse issues; provider marked failed.
- StorageError: file I/O problems; surfaced early and fail fast.

## CLI Commands (target)

- `version` — print version.
- `providers list` — show registered providers and status.
- `providers add NAME --kind KIND [--api-key-env ENV] [--base-url BASE] [--base-url-env ENV]` — add/update provider.
- `providers remove NAME` — remove provider.
- `fetch [NAME1,NAME2,...] [--all] [--force] [--max-concurrency N]` — fetch models, write JSON/TXT, generate tool configs.
- `update-configs [--tool TOOL] [--backup]` — update existing tool configs with new providers/models.
- `generate-configs [--tool TOOL] [--provider PROVIDER]` — generate per-provider config files.
- `stats` — show counts from storage plus last summary.
- `clean [--temp] [--configs]` — remove temp artifacts or generated configs; optionally add full cleanup with confirmation.

### New Commands for Config Integration
- `backup-configs` — create timestamped backups of existing tool configs
- `restore-configs --timestamp TIMESTAMP` — restore configs from backup
- `merge-configs --source SOURCE --target TARGET` — intelligent config merging

## Testing and Verification

- Unit tests: ensure `import vexy_co_model_catalog` is light and exposes `__version__`.
- Storage roundtrip tests: write/read JSON, TOML, TXT; ensure atomic writes do not leave temp files.
- Fetcher tests (mocked httpx): retry logic and error mapping.
- CLI smoke tests: `version`, `providers list` do not error.

## Migration Notes

- The new minimal modules are import-safe. Higher-level features will be layered in small increments to avoid regressions.

## Implementation Phases

### Phase 0: Core Infrastructure (Current)
- [x] Basic package structure with minimal imports
- [x] Exception hierarchy and storage foundations
- [x] CLI framework with Fire
- [ ] Enhanced storage layer with config/ directory support

### Phase 1: Provider Migration
- [ ] Import provider definitions from dump_models.py (45+ providers)
- [ ] Extend ProviderConfig with tool-specific generation rules
- [ ] Implement fetcher with retry logic and error mapping
- [ ] Add special provider handling (Chutes dual-API, etc.)

### Phase 2: Config Integration
- [ ] Config file parsers for aichat YAML, codex TOML, mods YAML
- [ ] Template engines for each tool's config format
- [ ] Backup and merge strategies for safe config updates
- [ ] Per-provider config file generation

### Phase 3: CLI Enhancement
- [ ] Complete fetch command with all output formats
- [ ] Config management commands (update-configs, backup-configs, etc.)
- [ ] Provider management commands (add/remove/list)
- [ ] Stats and cleanup commands

### Phase 4: Testing & Polish
- [ ] Comprehensive test suite for all provider types
- [ ] Integration tests with actual config files
- [ ] Error handling and edge case validation
- [ ] Documentation and usage examples

## Future Considerations

- Normalization schema: introduce a simple adapter per provider to map API → Model fields.
- Pricing and context windows: unify extraction across providers.
- Config generators: produce config blocks for third-party tools (e.g., LiteLLM, OpenRouter-compatible clients).
- Caching and ETag/Last-Modified support to reduce bandwidth.
- Web UI for config management and provider status monitoring.

