# Vexy Co Model Catalog

A robust, import-safe Python library and CLI tool for fetching, normalizing, and storing AI model catalogs from various providers. Supports 45+ AI providers including OpenAI, Anthropic, Groq, and others, enabling seamless integration with third-party tools like aichat, codex, and mods.

## Overview

The vexy-co-model-catalog project unifies AI model catalog management across the rapidly evolving landscape of AI providers. It serves as critical infrastructure for developers and tools that need reliable, normalized access to model information from diverse providers.

## Features

- **Multi-Provider Support**: Fetch models from 45+ AI providers (OpenAI, Anthropic, Groq, etc.)
- **Unified Format**: Normalize diverse provider APIs into consistent data structures  
- **Tool Integration**: Generate configuration files for aichat, codex, mods, and other CLI tools
- **Multiple Output Formats**: JSON, TXT, and TOML generation per provider
- **Import-Safe Architecture**: Lightweight package imports with lazy loading
- **Modern Python**: Type hints, async support, and PEP 621 compliance
- **Comprehensive Testing**: Full test suite with provider mocking and integration tests
- **CI/CD Ready**: Automated testing, linting, and release configuration

## Installation

```bash
pip install vexy-co-model-catalog
```

## Quick Start

### CLI Usage

```bash
# Show available providers
vexy-co-model-catalog providers list

# Fetch models from all providers
vexy-co-model-catalog fetch --all

# Fetch models from specific providers
vexy-co-model-catalog fetch openai,anthropic,groq

# Update existing tool configurations
vexy-co-model-catalog update-configs --tool aichat --backup

# Generate per-provider config files
vexy-co-model-catalog generate-configs --tool codex

# Show system stats
vexy-co-model-catalog stats

# Clean up temporary files
vexy-co-model-catalog clean
```

### Python API Usage

```python
import vexy_co_model_catalog

# Basic usage - import-safe with lazy loading
print(vexy_co_model_catalog.__version__)
```

## Tool Integration

### aichat Configuration

Update your aichat config with new providers:

```bash
# Backup existing config and update with new models
vexy-co-model-catalog update-configs --tool aichat --backup

# Generate individual provider config files
vexy-co-model-catalog generate-configs --tool aichat --provider openai
```

Example integration in `~/.config/aichat/config.yaml`:
```yaml
model: gpt-4
temperature: 0.7
clients:
  - type: openai
    api_base: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}
    models:
      - gpt-4
      - gpt-3.5-turbo
      - gpt-4-turbo
```

### codex Configuration  

Integrate with codex tool configurations:

```bash
# Update codex TOML config with new providers
vexy-co-model-catalog update-configs --tool codex --backup

# Generate provider-specific TOML files
vexy-co-model-catalog generate-configs --tool codex
```

### mods Configuration

Configure mods with AI provider catalogs:

```bash
# Update mods YAML config with new models
vexy-co-model-catalog update-configs --tool mods --backup
```

## Migration from dump_models.py

If you're migrating from the external `dump_models.py` script, here's how to transition:

### 1. File Format Compatibility

The new tool maintains **100% compatibility** with dump_models.py output:

```bash
# Old: dump_models.py generates files like models_openai.json
# New: Same files generated in config/json/models_openai.json
vexy-co-model-catalog fetch --all
```

### 2. Provider Configuration

All 40+ providers from dump_models.py are supported:

```bash
# List all available providers (same as dump_models.py)
vexy-co-model-catalog providers list

# Check specific provider details
vexy-co-model-catalog providers show openai
```

### 3. Environment Variables

Same environment variable patterns are used:

```bash
# Set API keys (same as dump_models.py)
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here" 
export GROQ_API_KEY="your-key-here"

# Custom base URLs work the same way
export OPENAI_API_OPENAI="https://api.openai.com/v1"
```

### 4. Output Directory Structure

Choose between new structured config or legacy models directory:

```bash
# New structured approach (recommended)
vexy-co-model-catalog fetch --all
# Files saved to: config/json/, config/txt/, config/{tool}/

# Legacy compatibility mode (optional)
vexy-co-model-catalog fetch --all --legacy-output
# Files saved to: models/ (same as dump_models.py)
```

### 5. Migration Steps

1. **Install the new tool**: `pip install vexy-co-model-catalog`
2. **Test compatibility**: `vexy-co-model-catalog providers list`
3. **Fetch models**: `vexy-co-model-catalog fetch --all`
4. **Compare outputs**: Verify JSON/TXT files match dump_models.py format
5. **Update integrations**: Point tools to new config/ directory structure
6. **Remove old script**: Archive dump_models.py once migration is verified

### 6. Enhanced Features

New capabilities beyond dump_models.py:

- **Config Integration**: Automatic tool configuration updates
- **Backup/Restore**: Safe config modification with rollback
- **Provider Management**: Add/remove providers via CLI
- **Better Error Handling**: Detailed failure tracking and retry logic
- **Testing**: Comprehensive test suite for reliability
- **Modern Architecture**: Async operations and type safety

## Development

This project uses [Hatch](https://hatch.pypa.io/) for development workflow management.

### Setup Development Environment

```bash
# Install hatch if you haven't already
pip install hatch

# Create and activate development environment
hatch shell

# Run tests
hatch run test

# Run tests with coverage
hatch run test-cov

# Run linting
hatch run lint

# Format code
hatch run format
```

## License

MIT License 