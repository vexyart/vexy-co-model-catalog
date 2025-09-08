---
this_file: TODO.md
---

## Phase 0: Enhanced Core Infrastructure
- [ ] Storage: extend StorageManager to support `config/` directory structure
- [ ] Storage: create directory layout (`config/json/`, `config/txt/`, etc.)
- [ ] Storage: add support for YAML file operations alongside JSON/TOML/TXT
- [ ] Core: create config module for tool-specific parsers and generators

## Phase 1: Provider Migration from dump_models.py
- [ ] Provider: extract provider definitions from external/dump_models.py
- [ ] Provider: import PROVIDER_CONFIG (45+ providers) into ProviderConfig classes
- [ ] Provider: import PROVIDER_URL_CONFIG environment variable mappings
- [ ] Provider: implement ProviderKind mapping (`oai`→`openai`, `ant`→`anthropic`, `url`)
- [ ] Provider: add Chutes special handling (dual-API merge functionality)
- [ ] Provider: preserve failed provider tracking from failed_models.json
- [ ] Fetcher: implement async JSON fetching with retry logic and error mapping
- [ ] Fetcher: add support for Anthropic version headers and URL providers

## Phase 2: Config File Integration
- [ ] Config: create aichat YAML parser and generator
- [ ] Config: create codex TOML parser and generator  
- [ ] Config: create mods YAML parser and generator
- [ ] Config: implement backup/restore functionality for existing configs
- [ ] Config: design template system for per-provider config generation
- [ ] Config: implement merge strategies for updating existing tool configs
- [ ] Storage: per-provider config file generation (`config/{tool}/models_{provider}.{ext}`)

## Phase 3: CLI Enhancement  
- [ ] CLI: implement `fetch` command with JSON/TXT/config outputs
- [ ] CLI: add `update-configs` command to modify existing tool configs
- [ ] CLI: add `generate-configs` command for per-provider files
- [ ] CLI: add `backup-configs` and `restore-configs` commands
- [ ] CLI: extend `providers` commands with add/remove/list functionality
- [ ] CLI: implement `stats` command with provider status and file counts
- [ ] CLI: implement `clean` command with config cleanup options
- [ ] Catalog: orchestrate fetching with config generation workflow

## Phase 4: Testing & Compatibility
- [ ] Test: unit tests for all provider types and special cases
- [ ] Test: integration tests with real config files  
- [ ] Test: CLI smoke tests for all commands
- [ ] Test: backup/restore functionality validation
- [ ] Compatibility: ensure output matches external/dump_models.py format
- [ ] Compatibility: maintain backward compatibility with external/bak structure
- [ ] Documentation: README updates for config integration usage
- [ ] Documentation: migration guide from dump_models.py

## Legacy Compatibility & Migration
- [ ] External: verify alignment with external/bak output format (112 files)
- [ ] External: support optional output to `models/` directory for backward compatibility  
- [ ] External: add symlink/copy options for existing tooling integration
- [ ] Migration: create import script to migrate from dump_models.py provider configs

