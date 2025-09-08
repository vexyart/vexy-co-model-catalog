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

## POST-IMPLEMENTATION: Comprehensive Documentation Plan for src_docs/

> **NOTE**: This documentation phase begins ONLY after all above implementation phases (0-4) are complete and the codebase is stable and fully tested.

### Prerequisites for Documentation Phase
- [ ] All Phases 0-4 above completed and tested
- [ ] Core provider system working with 45+ providers
- [ ] Config integration for aichat, codex, mods fully functional
- [ ] CLI commands robust and error-handling complete
- [ ] Test suite comprehensive with high coverage

### Phase 5: Foundation Documentation (Post-Code-Complete)
- [ ] `src_docs/README.md` - Main documentation hub with navigation
- [ ] `src_docs/ARCHITECTURE.md` - Complete system design documentation
- [ ] `src_docs/QUICKSTART.md` - Getting started with working examples
- [ ] `src_docs/INSTALLATION.md` - Installation and setup guide

### Phase 6: Core System Documentation
- [ ] `src_docs/PROVIDERS.md` - Provider system implementation guide
- [ ] `src_docs/FETCHING.md` - Model fetching pipeline and error patterns
- [ ] `src_docs/STORAGE.md` - File storage system and directory layouts
- [ ] `src_docs/CLI_REFERENCE.md` - Complete CLI command documentation
- [ ] `src_docs/CONFIG_INTEGRATION.md` - Tool config management patterns

### Phase 7: Provider Ecosystem Documentation
- [ ] `src_docs/providers/OPENAI.md` - OpenAI provider integration guide
- [ ] `src_docs/providers/ANTHROPIC.md` - Anthropic/Claude provider specifics
- [ ] `src_docs/providers/OPENROUTER.md` - OpenRouter unified API patterns
- [ ] `src_docs/providers/GROQ.md` - Groq high-performance integration
- [ ] `src_docs/providers/CEREBRAS.md` - Cerebras ultra-fast inference
- [ ] `src_docs/providers/CHUTES.md` - Chutes dual-API handling patterns
- [ ] `src_docs/providers/TOGETHER_AI.md` - Together AI community models
- [ ] `src_docs/providers/HUGGINGFACE.md` - HuggingFace inference endpoints
- [ ] `src_docs/providers/CUSTOM_PROVIDER.md` - Creating new providers

### Phase 8: Tool Integration Guides
- [ ] `src_docs/integrations/AICHAT.md` - aichat configuration patterns
- [ ] `src_docs/integrations/CODEX.md` - Codex tool integration
- [ ] `src_docs/integrations/MODS.md` - mods terminal tool integration
- [ ] `src_docs/integrations/LANGCHAIN.md` - LangChain framework integration
- [ ] `src_docs/integrations/LLAMAINDEX.md` - LlamaIndex integration patterns
- [ ] `src_docs/integrations/CREWAI.md` - CrewAI multi-agent system integration
- [ ] `src_docs/integrations/CUSTOM_TOOLS.md` - Building new tool integrations

### Phase 9: Advanced Features and Operations
- [ ] `src_docs/advanced/BATCH_OPERATIONS.md` - Bulk provider operations
- [ ] `src_docs/advanced/CACHING.md` - Performance optimization strategies
- [ ] `src_docs/advanced/ERROR_HANDLING.md` - Comprehensive error handling
- [ ] `src_docs/advanced/MONITORING.md` - System monitoring and observability
- [ ] `src_docs/operations/DEPLOYMENT.md` - Production deployment patterns
- [ ] `src_docs/operations/TROUBLESHOOTING.md` - Common issues and solutions
- [ ] `src_docs/operations/BACKUP_RECOVERY.md` - Config backup and recovery

### Phase 10: API Reference and Development
- [ ] `src_docs/api/CORE_API.md` - Core classes and methods reference
- [ ] `src_docs/api/PROVIDER_API.md` - Provider interface documentation
- [ ] `src_docs/api/STORAGE_API.md` - Storage system API reference
- [ ] `src_docs/api/CONFIG_API.md` - Configuration management API
- [ ] `src_docs/development/CONTRIBUTING.md` - Contribution guidelines
- [ ] `src_docs/development/TESTING.md` - Testing strategies and frameworks
- [ ] `src_docs/development/PROVIDER_DEVELOPMENT.md` - Provider development guide

### Phase 11: Examples and Use Cases
- [ ] `src_docs/examples/basic_usage.py` - Simple catalog fetching examples
- [ ] `src_docs/examples/custom_provider.py` - Complete custom provider impl
- [ ] `src_docs/examples/batch_update.py` - Bulk operations example
- [ ] `src_docs/examples/config_automation.py` - Automated config management
- [ ] `src_docs/use_cases/AI_TOOL_DEVELOPMENT.md` - Building AI tools with catalogs
- [ ] `src_docs/use_cases/ENTERPRISE_DEPLOYMENT.md` - Large-scale deployment
- [ ] `src_docs/use_cases/RESEARCH_APPLICATIONS.md` - Academic and research uses

### Phase 12: Ecosystem Analysis and Research
- [ ] `src_docs/ecosystem/AI_PROVIDER_LANDSCAPE.md` - Comprehensive provider analysis
- [ ] `src_docs/ecosystem/TOOL_INTEGRATION_MATRIX.md` - Tool compatibility overview
- [ ] `src_docs/ecosystem/MARKET_ANALYSIS.md` - AI tooling market positioning
- [ ] `src_docs/ecosystem/COMPETITIVE_ANALYSIS.md` - Alternative solutions comparison
- [ ] `src_docs/research/PERFORMANCE_BENCHMARKS.md` - System performance analysis
- [ ] `src_docs/research/PROVIDER_RELIABILITY.md` - Provider uptime and reliability
- [ ] `src_docs/research/COST_ANALYSIS.md` - Economic analysis of providers
- [ ] `src_docs/research/FUTURE_ROADMAP.md` - Technology evolution planning

### Documentation Quality Standards
- All code examples tested against implemented system
- Provider information validated against current live APIs
- Integration guides tested with actual external tools
- External reference materials fully integrated and cited
- Progressive complexity from basic to advanced topics
- Cross-references and navigation optimized for usability

### External Research Integration Sources
Drawing from the comprehensive external/ reference materials:
- `external/reference/`: Technical documentation from major AI libraries
- `external/writings/`: Provider-specific insights and usage patterns  
- `external/clitools/`: CLI tool integration patterns and examples
- `external/api_inference/`: API integration strategies and best practices
- `external/chutes_chutes.json`: Real-world provider catalog structure

### Documentation Infrastructure Setup
- [ ] MkDocs configuration for professional documentation site
- [ ] Automated API reference generation from code docstrings
- [ ] Code example testing integration in CI/CD pipeline
- [ ] Cross-reference validation and link checking automation

