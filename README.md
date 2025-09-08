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

## Usage

```python
import vexy_co_model_catalog
```

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