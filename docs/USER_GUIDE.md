# vexy-co-model-catalog User Guide

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [CLI Reference](#cli-reference)
4. [Provider Management](#provider-management)
5. [Configuration Integration](#configuration-integration)
6. [API Reference](#api-reference)
7. [Troubleshooting](#troubleshooting)
8. [Examples](#examples)
9. [Production Deployment](#production-deployment)

## Quick Start

### Installation
```bash
# Install from PyPI
pip install vexy-co-model-catalog

# Or install with uv
uv add vexy-co-model-catalog
```

### First Use
```bash
# List all available providers (40+ providers)
vexy-model-catalog providers

# Fetch models from OpenAI
vexy-model-catalog fetch openai

# Show system statistics
vexy-model-catalog stats

# Get help for any command
vexy-model-catalog help
```

### Interactive Setup (Recommended for First-Time Users)
```bash
vexy-model-catalog setup_wizard
```

## Installation

### Requirements
- Python 3.10, 3.11, or 3.12
- Internet connection for fetching model catalogs
- Operating Systems: macOS, Linux, Windows

### Install Options

#### Option 1: PyPI Installation (Recommended)
```bash
pip install vexy-co-model-catalog
```

#### Option 2: UV Installation (Modern Python Package Manager)
```bash
uv add vexy-co-model-catalog
```

#### Option 3: Development Installation
```bash
git clone https://github.com/vexyart/vexy-co-model-catalog
cd vexy-co-model-catalog
uv install -e .
```

### Verify Installation
```bash
vexy-model-catalog version
vmc version  # Short alias
```

## CLI Reference

### Core Commands

#### `providers` - List Available Providers
```bash
# List all providers with status
vexy-model-catalog providers

# Aliases
vmc ls
vmc providers
```

**Example Output:**
```
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Provider               ┃ Kind           ┃ Status              ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│ openai                │ openai         │ ✅ Ready            │
│ anthropic             │ anthropic      │ ✅ Ready            │
│ groq                  │ openai         │ ✅ Ready            │
└────────────────────────┴────────────────┴─────────────────────┘
```

#### `fetch` - Download Model Catalogs
```bash
# Fetch from specific provider
vexy-model-catalog fetch openai

# Fetch from multiple providers
vexy-model-catalog fetch openai anthropic groq

# Fetch from all providers
vexy-model-catalog fetch --all

# Aliases
vmc get openai
vmc download anthropic
vmc sync groq
```

**Example Output:**
```
🚀 Fetching models from providers: openai
✅ OpenAI: 145 models fetched
📁 Generated config files:
   - config/aichat/models_openai.yaml
   - config/codex/models_openai.toml  
   - config/mods/models_openai.yaml
```

#### `stats` - Show System Statistics
```bash
# Show comprehensive statistics
vexy-model-catalog stats

# Aliases  
vmc st
vmc status
```

**Example Output:**
```
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Provider               ┃ Models         ┃ Last Updated        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━┩
│ openai                │ 145            │ 2025-01-09 14:30:22 │
│ anthropic             │ 12             │ 2025-01-09 14:28:15 │
│ groq                  │ 28             │ 2025-01-09 14:25:10 │
└────────────────────────┴────────────────┴─────────────────────┘
```

#### `validate` - Check Provider Configurations
```bash
# Validate all providers
vexy-model-catalog validate

# Validate specific provider
vexy-model-catalog validate openai

# Aliases
vmc check
vmc test
vmc verify
```

#### `health` - System Health Check
```bash
# Run comprehensive health check
vexy-model-catalog health

# Aliases
vmc diag
vmc diagnostics  
vmc healthcheck
```

### Advanced Commands

#### `cache` - Cache Management
```bash
# Show cache status
vexy-model-catalog cache status

# Clear all caches
vexy-model-catalog cache clear

# Clear specific cache
vexy-model-catalog cache clear --type models
```

#### `clean` - Cleanup Operations
```bash
# Clean temporary files
vexy-model-catalog clean

# Clean with specific options
vexy-model-catalog clean --cache --logs --temp

# Aliases
vmc rm
vmc clear
vmc remove
```

#### `performance` - Performance Monitoring
```bash
# Show performance statistics
vexy-model-catalog performance stats

# Show command history with timing
vexy-model-catalog performance history

# Clear performance metrics
vexy-model-catalog performance clear
```

### Production Commands

#### `production_init` - Initialize Production Mode
```bash
# Initialize production environment
vexy-model-catalog production_init
```

#### `production_status` - Show Production Health
```bash
# Show production status and metrics
vexy-model-catalog production_status
```

#### `production_readiness` - Check Deployment Readiness
```bash
# Check if ready for production deployment
vexy-model-catalog production_readiness
```

#### `production_errors` - Show Error Statistics
```bash
# Show production error summary
vexy-model-catalog production_errors
```

### Help and Information

#### `help` - Get Help
```bash
# Show general help
vexy-model-catalog help

# Show help for specific command
vexy-model-catalog help fetch
```

#### `aliases` - Show Command Aliases
```bash
# Show all available command aliases
vexy-model-catalog aliases
```

#### `version` - Show Version Information
```bash
# Show version
vexy-model-catalog version
vmc version
```

### Environment Variables

#### Core Configuration
- `VMC_PRODUCTION_MODE=true` - Enable production mode with enhanced error handling
- `VMC_LOG_DIR=/custom/path` - Override default log directory
- `VMC_LOG_LEVEL=INFO` - Set logging level (DEBUG, INFO, WARNING, ERROR)

#### Analytics and Monitoring
- `VEXY_ANALYTICS_ENABLED=true` - Enable usage analytics (default: true)
- `VEXY_PERFORMANCE_ENABLED=true` - Enable performance monitoring (default: true)

#### Provider Configuration
- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `GROQ_API_KEY` - Groq API key
- And 40+ other provider-specific keys

## Provider Management

### Supported Providers (40+)

#### OpenAI-Compatible APIs (37 providers)
- **OpenAI** - Original GPT models
- **Groq** - Ultra-fast inference
- **Cerebras** - High-performance models
- **DeepInfra** - Cost-effective inference
- **Together AI** - Community models
- **OpenRouter** - Unified API access
- **Anyscale** - Ray-powered inference
- **Fireworks** - Fast model serving
- **Perplexity** - Search-augmented models
- **And 28 more...**

#### Anthropic API (1 provider)
- **Anthropic** - Claude models

#### Direct URL Providers (2 providers)
- **LiteLLM Catalog** - Community model listings
- **Custom catalogs** - Direct JSON endpoints

### Provider Configuration

#### Setting Up API Keys
```bash
# Set API keys via environment variables
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export GROQ_API_KEY="your-groq-key"

# Or use the setup wizard
vexy-model-catalog setup_wizard
```

#### Provider Status
```bash
# Check which providers are properly configured
vexy-model-catalog validate

# View provider details
vexy-model-catalog providers
```

### Adding Custom Providers

You can extend the catalog by adding custom providers. See the API Reference section for details.

## Configuration Integration

The tool automatically generates configuration files for popular AI tools.

### Supported Tools

#### aichat (YAML Configuration)
```bash
# Fetch and generate aichat config
vexy-model-catalog fetch openai

# Config generated at: config/aichat/models_openai.yaml
```

**Generated Config Example:**
```yaml
model_configs:
  - name: gpt-4-turbo
    provider: openai
    max_input_tokens: 128000
    max_output_tokens: 4096
```

#### Codex (TOML Configuration)  
```bash
# Generate codex profiles
vexy-model-catalog fetch anthropic

# Config generated at: config/codex/models_anthropic.toml
```

**Generated Config Example:**
```toml
[models.claude-3-opus]
provider = "anthropic"
max_input_tokens = 200000
max_output_tokens = 4096
```

#### mods (YAML Configuration)
```bash
# Generate mods config
vexy-model-catalog fetch groq

# Config generated at: config/mods/models_groq.yaml
```

### Integration Workflow
1. Run `vexy-model-catalog fetch <provider>` 
2. Config files are automatically generated in `config/` directory
3. Copy or symlink configs to your tool's config directory
4. Use the `link` command for automatic integration

```bash
# Automatically link configs to tool directories
vexy-model-catalog link aichat
vexy-model-catalog link codex  
vexy-model-catalog link mods
```

## API Reference

### Core Classes

#### ModelCatalog
The main class for managing model catalogs.

```python
from vexy_co_model_catalog import ModelCatalog
from pathlib import Path

# Create catalog instance
catalog = ModelCatalog(storage_root=Path("./models"))

# List available providers  
providers = catalog.list_providers()
print(f"Found {len(providers)} providers")

# Get failed providers
failed = catalog.get_failed_providers()
```

#### ProviderConfig
Represents a provider configuration.

```python
from vexy_co_model_catalog.core.provider import ProviderConfig, ProviderKind

# Access provider information
config = ProviderConfig(
    name="openai",
    kind=ProviderKind.OPENAI,
    base_url="https://api.openai.com/v1",
    env_var="OPENAI_API_KEY"
)
```

#### StorageManager
Manages file storage operations.

```python  
from vexy_co_model_catalog.core.storage import StorageManager
from pathlib import Path

# Create storage manager
storage = StorageManager(root_path=Path("./data"))

# Store JSON data
storage.store_json("openai_models.json", {"models": []}, "models")

# Read stored data
data = storage.read_json("openai_models.json", "models")
```

### Programmatic Usage

#### Basic Model Fetching
```python
import asyncio
from vexy_co_model_catalog.core.fetcher import ModelFetcher
from vexy_co_model_catalog.core.provider import get_provider_by_name

async def fetch_models():
    provider = get_provider_by_name("openai")
    if provider:
        fetcher = ModelFetcher()
        models = await fetcher.fetch_models(provider)
        print(f"Fetched {len(models)} models from OpenAI")

# Run the async function
asyncio.run(fetch_models())
```

#### Configuration Generation
```python
from vexy_co_model_catalog.core.config import ConfigManager
from pathlib import Path

# Create config manager
config_manager = ConfigManager(Path("./config"))

# Generate aichat config
models_data = {"models": [...]}  # Your model data
config_manager.generate_aichat_config("openai", models_data)
```

#### Error Handling with Production Features
```python
from vexy_co_model_catalog.core.production_error_handling import (
    production_error_handler,
    error_context
)

# Use decorator for automatic error handling
@production_error_handler(user_action="fetching models")
def fetch_with_error_handling():
    # Your code here
    pass

# Use context manager for error handling
with error_context("processing models"):
    # Your code here
    pass
```

## Troubleshooting

### Common Issues

#### 1. API Key Not Found
**Problem:** `No API key found for provider 'openai'`

**Solution:**
```bash
# Set the API key environment variable
export OPENAI_API_KEY="your-api-key-here"

# Or use the setup wizard
vexy-model-catalog setup_wizard

# Verify the key is set
vexy-model-catalog validate openai
```

#### 2. Network Connection Issues
**Problem:** `ConnectionError: Failed to fetch from provider`

**Solution:**
```bash
# Check network connectivity
ping api.openai.com

# Check proxy settings if behind corporate firewall
export HTTP_PROXY="http://proxy.company.com:8080"
export HTTPS_PROXY="https://proxy.company.com:8080"

# Retry with increased timeout
vexy-model-catalog fetch openai --timeout 60
```

#### 3. Permission Denied Errors
**Problem:** `PermissionError: Cannot write to config directory`

**Solution:**
```bash
# Check directory permissions
ls -la config/

# Fix permissions
chmod 755 config/
chmod 644 config/**/*

# Or specify custom directory
export VMC_LOG_DIR="$HOME/.vexy-model-catalog"
```

#### 4. Import Errors
**Problem:** `ModuleNotFoundError: No module named 'vexy_co_model_catalog'`

**Solution:**
```bash
# Reinstall the package
pip uninstall vexy-co-model-catalog
pip install vexy-co-model-catalog

# Check Python path
python -c "import sys; print(sys.path)"

# Use full module path
python -m vexy_co_model_catalog.cli version
```

#### 5. Rate Limiting Issues
**Problem:** `Rate limit exceeded for provider`

**Solution:**
```bash
# Check rate limit status
vexy-model-catalog rate_limits

# Wait and retry with delay
vexy-model-catalog fetch openai --delay 5

# Use batch processing with smaller chunks
vexy-model-catalog fetch --batch-size 10
```

#### 6. Corrupted Cache Issues
**Problem:** `CacheError: Corrupted cache detected`

**Solution:**
```bash
# Clear all caches
vexy-model-catalog cache clear

# Check cache status
vexy-model-catalog cache status

# Rebuild cache
vexy-model-catalog fetch --all --force-refresh
```

### Debugging Commands

#### Enable Debug Logging
```bash
export VMC_LOG_LEVEL=DEBUG
vexy-model-catalog fetch openai
```

#### Run Health Diagnostics
```bash
vexy-model-catalog health
```

#### Check Production Status
```bash
vexy-model-catalog production_status
vexy-model-catalog production_readiness
```

#### View Performance Metrics
```bash
vexy-model-catalog performance stats
```

### Getting Help

#### Built-in Help
```bash
# General help
vexy-model-catalog help

# Command-specific help
vexy-model-catalog help fetch
vexy-model-catalog fetch --help
```

#### Setup Wizard
```bash
# Interactive setup for first-time users
vexy-model-catalog setup_wizard
```

#### Error Messages
The tool provides detailed error messages with:
- Error category classification
- Specific error codes  
- Recovery suggestions
- Documentation links

## Examples

### Example 1: Basic Setup and First Fetch
```bash
# 1. Install the tool
pip install vexy-co-model-catalog

# 2. Set up API keys
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"

# 3. Run interactive setup
vexy-model-catalog setup_wizard

# 4. Fetch your first models
vexy-model-catalog fetch openai

# 5. Check the results
vexy-model-catalog stats
ls config/
```

### Example 2: Multi-Provider Workflow
```bash
# Fetch from multiple providers
vexy-model-catalog fetch openai anthropic groq cerebras

# Check what was fetched
vexy-model-catalog providers

# Validate configurations
vexy-model-catalog validate

# View comprehensive statistics
vexy-model-catalog stats
```

### Example 3: Tool Integration
```bash
# Generate configs for all tools
vexy-model-catalog fetch openai

# Link configs to tool directories
vexy-model-catalog link aichat
vexy-model-catalog link codex
vexy-model-catalog link mods

# Backup existing configs first
vexy-model-catalog backup-configs

# Verify integration
ls ~/.config/aichat/
ls ~/.config/codex/
```

### Example 4: Production Deployment
```bash
# Enable production mode
export VMC_PRODUCTION_MODE=true

# Initialize production environment
vexy-model-catalog production_init

# Check deployment readiness
vexy-model-catalog production_readiness

# Monitor system health
vexy-model-catalog production_status

# View error statistics
vexy-model-catalog production_errors
```

### Example 5: Performance Monitoring
```bash
# Enable performance monitoring
export VEXY_PERFORMANCE_ENABLED=true

# Run commands and monitor performance
vexy-model-catalog fetch --all

# View performance statistics
vexy-model-catalog performance stats

# View command execution history
vexy-model-catalog performance history

# Save performance metrics
vexy-model-catalog performance save
```

### Example 6: Programmatic Usage
```python
#!/usr/bin/env python3

import asyncio
from pathlib import Path
from vexy_co_model_catalog import ModelCatalog
from vexy_co_model_catalog.core.provider import get_all_providers

async def main():
    # Create catalog
    catalog = ModelCatalog(storage_root=Path("./my_models"))
    
    # List all providers
    providers = catalog.list_providers()
    print(f"Available providers: {len(providers)}")
    
    # Get specific provider
    openai_provider = next(p for p in providers if p.name == "openai")
    print(f"OpenAI provider: {openai_provider.base_url}")
    
    # Check for failures
    failed = catalog.get_failed_providers()
    if failed:
        print(f"Failed providers: {list(failed.keys())}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Production Deployment

### Production Mode Features

Enable production mode for enhanced error handling, monitoring, and logging:

```bash
export VMC_PRODUCTION_MODE=true
```

#### Production Features
- **Enhanced Error Handling**: Automatic error categorization and recovery
- **Comprehensive Logging**: Structured logs with rotation and retention  
- **Health Monitoring**: System health checks and diagnostics
- **Performance Metrics**: Detailed performance tracking and reporting
- **Circuit Breakers**: Automatic failure detection and recovery
- **Graceful Degradation**: Fallback mechanisms for service failures

### Deployment Checklist

#### 1. Environment Setup
```bash
# Set production mode
export VMC_PRODUCTION_MODE=true

# Configure logging
export VMC_LOG_DIR="/var/log/vexy-model-catalog"  
export VMC_LOG_LEVEL=INFO

# Initialize production environment
vexy-model-catalog production_init
```

#### 2. Configuration Validation
```bash
# Check deployment readiness
vexy-model-catalog production_readiness

# Validate all provider configurations
vexy-model-catalog validate

# Run health diagnostics
vexy-model-catalog health
```

#### 3. Performance Baseline
```bash
# Run performance benchmark
vexy-model-catalog performance stats

# Test core functionality
vexy-model-catalog fetch --all --dry-run

# Monitor resource usage
vexy-model-catalog production_status
```

#### 4. Monitoring Setup
```bash
# Check production metrics
vexy-model-catalog production_status

# Monitor error rates
vexy-model-catalog production_errors

# Set up log monitoring (varies by platform)
tail -f /var/log/vexy-model-catalog/vexy-co-model-catalog.log
```

### Container Deployment

#### Dockerfile Example
```dockerfile
FROM python:3.12-slim

# Install dependencies
RUN pip install vexy-co-model-catalog

# Set production mode
ENV VMC_PRODUCTION_MODE=true
ENV VMC_LOG_LEVEL=INFO

# Create log directory
RUN mkdir -p /app/logs
ENV VMC_LOG_DIR=/app/logs

# Initialize production environment
RUN vexy-model-catalog production_init

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD vexy-model-catalog health || exit 1

# Run application
CMD ["vexy-model-catalog", "fetch", "--all", "--schedule", "hourly"]
```

#### Docker Compose Example
```yaml
version: '3.8'

services:
  vexy-model-catalog:
    image: vexy-model-catalog:latest
    environment:
      - VMC_PRODUCTION_MODE=true
      - VMC_LOG_LEVEL=INFO
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./config:/app/config
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "vexy-model-catalog", "health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Monitoring and Alerting

#### System Metrics
- **Health Status**: Overall system health
- **Error Rates**: Error counts by category and severity
- **Performance**: Response times and resource usage
- **Provider Status**: Individual provider health and availability
- **Cache Statistics**: Hit rates and cache performance

#### Log Analysis
Production logs include structured JSON with:
- Error correlation IDs
- Performance metrics
- User actions and context
- Recovery suggestions
- Severity levels

#### Operational Dashboards
Create dashboards monitoring:
- Provider availability and response times
- Error rates and types
- Cache hit rates and performance
- System resource utilization
- API key usage and rate limits

---

For additional support, check the [troubleshooting section](#troubleshooting) or visit the project repository.