# API Reference

## Core Classes

### ModelCatalog

The main class for managing AI model catalogs from multiple providers.

```python
from vexy_co_model_catalog import ModelCatalog
from pathlib import Path

catalog = ModelCatalog(storage_root=Path("./models"))
```

#### Constructor

```python
ModelCatalog(storage_root: Path | None = None)
```

**Parameters:**
- `storage_root` (Path, optional): Root directory for storing model data. Defaults to current working directory.

#### Methods

##### `list_providers() -> list[ProviderConfig]`

Returns a list of all available AI providers.

**Returns:**
- `list[ProviderConfig]`: List of provider configurations

**Example:**
```python
providers = catalog.list_providers()
for provider in providers:
    print(f"{provider.name}: {provider.kind.value}")
```

##### `get_failed_providers() -> dict[str, str]`

Returns a dictionary of providers that failed during the last operation.

**Returns:**
- `dict[str, str]`: Mapping of provider names to error messages

**Example:**
```python
failed = catalog.get_failed_providers()
if failed:
    print(f"Failed providers: {list(failed.keys())}")
```

---

### ProviderConfig

Represents the configuration for an AI provider.

```python
from vexy_co_model_catalog.core.provider import ProviderConfig, ProviderKind
```

#### Attributes

- `name: str` - Provider name (e.g., "openai", "anthropic")
- `kind: ProviderKind` - Provider type (OPENAI, ANTHROPIC, URL)  
- `base_url: str` - API base URL
- `env_var: str` - Environment variable for API key
- `headers: dict[str, str]` - Additional HTTP headers
- `models_endpoint: str` - Endpoint for fetching models

#### Example

```python
config = ProviderConfig(
    name="openai",
    kind=ProviderKind.OPENAI,
    base_url="https://api.openai.com/v1",
    env_var="OPENAI_API_KEY"
)
```

---

### StorageManager

Manages file storage operations with atomic writes and organized directory structure.

```python
from vexy_co_model_catalog.core.storage import StorageManager
from pathlib import Path

storage = StorageManager(root_path=Path("./data"))
```

#### Constructor

```python
StorageManager(root_path: str | os.PathLike | None = None)
```

**Parameters:**
- `root_path` (Path, optional): Root directory for storage. Defaults to current working directory.

#### Methods

##### `store_json(filename: str, data: Any, subdirectory: str = "models") -> None`

Store data as JSON file.

**Parameters:**
- `filename` (str): Name of the JSON file
- `data` (Any): Data to store (must be JSON serializable)
- `subdirectory` (str): Subdirectory within root path

**Example:**
```python
models_data = {"models": [...]}
storage.store_json("openai_models.json", models_data, "models")
```

##### `read_json(filename: str, subdirectory: str = "models") -> Any`

Read data from JSON file.

**Parameters:**
- `filename` (str): Name of the JSON file
- `subdirectory` (str): Subdirectory within root path

**Returns:**
- `Any`: Deserialized JSON data

**Example:**
```python
data = storage.read_json("openai_models.json", "models")
```

##### `store_text(filename: str, content: str, subdirectory: str = "models") -> None`

Store text content to file.

**Parameters:**
- `filename` (str): Name of the text file
- `content` (str): Text content to store
- `subdirectory` (str): Subdirectory within root path

##### `list_files(subdirectory: str, pattern: str = "*") -> list[Path]`

List files in a subdirectory matching a pattern.

**Parameters:**
- `subdirectory` (str): Subdirectory to search
- `pattern` (str): Glob pattern for file matching

**Returns:**
- `list[Path]`: List of matching file paths

---

### ModelFetcher

Handles asynchronous fetching of model catalogs from AI providers.

```python
from vexy_co_model_catalog.core.fetcher import ModelFetcher

fetcher = ModelFetcher()
```

#### Methods

##### `async fetch_models(provider: ProviderConfig) -> list[dict[str, Any]]`

Fetch models from a specific provider.

**Parameters:**
- `provider` (ProviderConfig): Provider configuration

**Returns:**
- `list[dict[str, Any]]`: List of model dictionaries

**Example:**
```python
import asyncio
from vexy_co_model_catalog.core.provider import get_provider_by_name

async def fetch_openai():
    provider = get_provider_by_name("openai")
    if provider:
        models = await fetcher.fetch_models(provider)
        print(f"Fetched {len(models)} models")

asyncio.run(fetch_openai())
```

---

### ConfigManager

Manages configuration file generation for external tools (aichat, codex, mods).

```python
from vexy_co_model_catalog.core.config import ConfigManager
from pathlib import Path

config_manager = ConfigManager(config_root=Path("./config"))
```

#### Constructor

```python
ConfigManager(config_root: Path)
```

**Parameters:**
- `config_root` (Path): Root directory for configuration files

#### Methods

##### `generate_aichat_config(provider_name: str, models_data: dict[str, Any]) -> Path`

Generate aichat YAML configuration.

**Parameters:**
- `provider_name` (str): Name of the provider
- `models_data` (dict): Model data from provider

**Returns:**
- `Path`: Path to generated config file

##### `generate_codex_config(provider_name: str, models_data: dict[str, Any]) -> Path`

Generate codex TOML configuration.

##### `generate_mods_config(provider_name: str, models_data: dict[str, Any]) -> Path`

Generate mods YAML configuration.

---

## Provider Functions

### `get_all_providers() -> list[ProviderConfig]`

Get all available provider configurations.

**Returns:**
- `list[ProviderConfig]`: List of all provider configurations

**Example:**
```python
from vexy_co_model_catalog.core.provider import get_all_providers

providers = get_all_providers()
print(f"Total providers: {len(providers)}")
```

### `get_provider_by_name(name: str) -> ProviderConfig | None`

Get a specific provider by name.

**Parameters:**
- `name` (str): Provider name

**Returns:**
- `ProviderConfig | None`: Provider configuration or None if not found

**Example:**
```python
from vexy_co_model_catalog.core.provider import get_provider_by_name

openai = get_provider_by_name("openai")
if openai:
    print(f"OpenAI base URL: {openai.base_url}")
```

---

## Error Handling

### Production Error Handler

```python
from vexy_co_model_catalog.core.production_error_handling import (
    production_error_handler,
    error_context,
    handle_critical_error
)
```

#### `@production_error_handler(user_action: str = None, reraise: bool = False)`

Decorator for automatic error handling with logging and recovery.

**Parameters:**
- `user_action` (str, optional): Description of the user action
- `reraise` (bool): Whether to reraise the exception after handling

**Example:**
```python
@production_error_handler(user_action="fetching models")
def fetch_with_error_handling():
    # Your code here that might fail
    pass
```

#### `error_context(operation: str, reraise: bool = False)`

Context manager for error handling.

**Parameters:**
- `operation` (str): Description of the operation
- `reraise` (bool): Whether to reraise exceptions

**Example:**
```python
with error_context("processing models"):
    # Your code here
    pass
```

---

## Performance Monitoring

### Performance Monitor

```python
from vexy_co_model_catalog.core.performance import get_performance_monitor

monitor = get_performance_monitor()
```

#### Methods

##### `record_command_start(command: str) -> str`

Start recording performance metrics for a command.

**Parameters:**
- `command` (str): Command name

**Returns:**
- `str`: Performance tracking ID

##### `record_command_end(perf_id: str, success: bool = True) -> None`

End performance recording for a command.

**Parameters:**
- `perf_id` (str): Performance tracking ID
- `success` (bool): Whether the command succeeded

---

## Caching

### Cache Management

```python
from vexy_co_model_catalog.core.caching import get_model_cache

cache = get_model_cache()
```

#### Methods

##### `get(key: str) -> Any | None`

Get cached value.

**Parameters:**
- `key` (str): Cache key

**Returns:**
- `Any | None`: Cached value or None if not found

##### `set(key: str, value: Any, ttl: int = 3600) -> None`

Set cached value with TTL.

**Parameters:**
- `key` (str): Cache key
- `value` (Any): Value to cache
- `ttl` (int): Time to live in seconds

##### `clear() -> None`

Clear all cached values.

---

## Security

### Data Protection

```python
from vexy_co_model_catalog.core.security import get_protector

protector = get_protector()
```

#### Methods

##### `mask_sensitive_data(text: str) -> str`

Mask sensitive data in text (API keys, tokens, etc.).

**Parameters:**
- `text` (str): Text potentially containing sensitive data

**Returns:**
- `str`: Text with sensitive data masked

**Example:**
```python
safe_text = protector.mask_sensitive_data("API key: sk-1234567890")
# Result: "API key: sk-***[REDACTED]"
```

---

## Validation

### Provider Validation

```python
from vexy_co_model_catalog.core.validator import ProviderValidator

validator = ProviderValidator()
```

#### Methods

##### `validate_provider(provider: ProviderConfig) -> ValidationResult`

Validate a provider configuration.

**Parameters:**
- `provider` (ProviderConfig): Provider to validate

**Returns:**
- `ValidationResult`: Validation results with status and messages

---

## Health Monitoring

### Health Checks

```python
from vexy_co_model_catalog.core.health_monitor import get_health_monitor

health = get_health_monitor()
```

#### Methods

##### `check_system_health() -> HealthStatus`

Perform comprehensive system health check.

**Returns:**
- `HealthStatus`: System health status with details

##### `check_provider_health(provider_name: str) -> bool`

Check if a specific provider is healthy.

**Parameters:**
- `provider_name` (str): Provider name

**Returns:**
- `bool`: True if provider is healthy

---

## Analytics

### Usage Analytics

```python
from vexy_co_model_catalog.core.analytics import get_analytics

analytics = get_analytics()
```

#### Methods

##### `record_command_usage(command: str, success: bool, duration: float) -> None`

Record command usage statistics.

**Parameters:**
- `command` (str): Command name
- `success` (bool): Whether command succeeded
- `duration` (float): Execution duration in seconds

---

## Rate Limiting

### Rate Limiter

```python
from vexy_co_model_catalog.core.rate_limiter import get_rate_limiter

limiter = get_rate_limiter()
```

#### Methods

##### `is_allowed(provider_name: str) -> bool`

Check if request to provider is allowed.

**Parameters:**
- `provider_name` (str): Provider name

**Returns:**
- `bool`: True if request is allowed

##### `record_request(provider_name: str, success: bool) -> None`

Record a request to update rate limiting stats.

**Parameters:**
- `provider_name` (str): Provider name  
- `success` (bool): Whether request succeeded

---

## Data Integrity

### Integrity Manager

```python
from vexy_co_model_catalog.core.integrity import get_integrity_manager

integrity = get_integrity_manager()
```

#### Methods

##### `verify_file_integrity(file_path: Path) -> bool`

Verify file integrity using checksums.

**Parameters:**
- `file_path` (Path): Path to file

**Returns:**
- `bool`: True if file integrity is valid

##### `repair_corrupted_file(file_path: Path) -> bool`

Attempt to repair a corrupted file from backup.

**Parameters:**
- `file_path` (Path): Path to corrupted file

**Returns:**
- `bool`: True if repair was successful

---

## Type Definitions

### ProviderKind

```python
from vexy_co_model_catalog.core.provider import ProviderKind

class ProviderKind(Enum):
    OPENAI = "openai"      # OpenAI-compatible API
    ANTHROPIC = "anthropic"  # Anthropic Claude API  
    URL = "url"            # Direct URL endpoint
```

### ValidationResult

```python
from vexy_co_model_catalog.core.validator import ValidationResult

@dataclass
class ValidationResult:
    is_valid: bool
    messages: list[str]
    severity: ValidationSeverity
```

### HealthStatus

```python
from vexy_co_model_catalog.core.health_monitor import HealthStatus

@dataclass  
class HealthStatus:
    overall_health: bool
    provider_health: dict[str, bool]
    system_metrics: dict[str, Any]
    last_check: datetime
```

---

## Usage Patterns

### Basic Model Fetching

```python
import asyncio
from vexy_co_model_catalog import ModelCatalog
from pathlib import Path

async def fetch_models_example():
    # Create catalog
    catalog = ModelCatalog(storage_root=Path("./models"))
    
    # Get available providers
    providers = catalog.list_providers()
    openai_provider = next(p for p in providers if p.name == "openai")
    
    # Fetch models (this would typically be done via CLI)
    # The ModelCatalog class focuses on provider management
    # Actual fetching is handled by the CLI or ModelFetcher directly
    
    print(f"OpenAI provider configured: {openai_provider.base_url}")

asyncio.run(fetch_models_example())
```

### Error Handling Pattern

```python
from vexy_co_model_catalog.core.production_error_handling import (
    production_error_handler,
    error_context
)

@production_error_handler(user_action="custom operation")
def safe_operation():
    # Operations with automatic error handling
    with error_context("nested operation"):
        # Nested operations also handled
        pass

# Call the function - errors will be automatically handled
result = safe_operation()
```

### Configuration Generation

```python
from vexy_co_model_catalog.core.config import ConfigManager
from pathlib import Path

def generate_configs_example():
    config_manager = ConfigManager(Path("./config"))
    
    # Example model data (normally from API)
    models_data = {
        "models": [
            {
                "id": "gpt-4-turbo",
                "max_input_tokens": 128000,
                "max_output_tokens": 4096
            }
        ]
    }
    
    # Generate configs for different tools
    aichat_path = config_manager.generate_aichat_config("openai", models_data)
    codex_path = config_manager.generate_codex_config("openai", models_data)
    mods_path = config_manager.generate_mods_config("openai", models_data)
    
    print(f"Generated configs:")
    print(f"  aichat: {aichat_path}")
    print(f"  codex: {codex_path}")
    print(f"  mods: {mods_path}")
```

### Performance Monitoring

```python
from vexy_co_model_catalog.core.performance import get_performance_monitor

def monitored_operation():
    monitor = get_performance_monitor()
    
    # Start monitoring
    perf_id = monitor.record_command_start("custom_operation")
    
    try:
        # Your operation here
        time.sleep(1)  # Simulate work
        
        # Record success
        monitor.record_command_end(perf_id, success=True)
    except Exception as e:
        # Record failure
        monitor.record_command_end(perf_id, success=False)
        raise
```

---

## Advanced Usage

### Custom Provider Integration

While the tool comes with 40+ built-in providers, you can extend it for custom providers:

```python
from vexy_co_model_catalog.core.provider import ProviderConfig, ProviderKind

# Define custom provider
custom_provider = ProviderConfig(
    name="custom_api",
    kind=ProviderKind.OPENAI,  # Use OpenAI-compatible format
    base_url="https://api.custom.com/v1",
    env_var="CUSTOM_API_KEY",
    headers={"Custom-Header": "value"}
)

# Use with ModelFetcher
from vexy_co_model_catalog.core.fetcher import ModelFetcher

async def fetch_custom():
    fetcher = ModelFetcher()
    models = await fetcher.fetch_models(custom_provider)
    return models
```

### Batch Processing with Error Recovery

```python
import asyncio
from vexy_co_model_catalog.core.production_graceful_degradation import (
    with_graceful_degradation,
    FallbackConfig,
    FallbackStrategy
)

# Configure graceful degradation
fallback_config = FallbackConfig(
    strategy=FallbackStrategy.RETRY,
    max_retries=3,
    retry_delay=1.0,
    default_value=[]
)

@with_graceful_degradation("batch_processing", fallback_config)
async def process_providers_batch(providers):
    results = []
    for provider in providers:
        # Process each provider with automatic retry and fallback
        result = await process_single_provider(provider)
        results.append(result)
    return results
```

This API reference covers the core functionality of the vexy-co-model-catalog package. For additional examples and use cases, see the [User Guide](USER_GUIDE.md).