# Troubleshooting Guide

## Quick Diagnostic Commands

Before diving into specific issues, run these commands to get an overview of system status:

```bash
# Check system health
vexy-model-catalog health

# Show production status
vexy-model-catalog production_status

# Validate all configurations
vexy-model-catalog validate

# Show version and basic info
vexy-model-catalog version
```

## Common Issues

### 1. Installation Problems

#### Issue: Package Installation Fails
```
ERROR: Could not install vexy-co-model-catalog
```

**Diagnosis:**
```bash
python --version  # Check Python version (need 3.10+)
pip --version     # Check pip version
```

**Solutions:**
```bash
# Option 1: Upgrade pip
python -m pip install --upgrade pip
pip install vexy-co-model-catalog

# Option 2: Use uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv add vexy-co-model-catalog

# Option 3: Install from source
git clone https://github.com/vexyart/vexy-co-model-catalog
cd vexy-co-model-catalog
pip install -e .
```

#### Issue: Import Errors After Installation
```
ModuleNotFoundError: No module named 'vexy_co_model_catalog'
```

**Diagnosis:**
```bash
pip list | grep vexy
python -c "import sys; print(sys.path)"
which python
```

**Solutions:**
```bash
# Ensure you're using the right Python environment
python -m pip install vexy-co-model-catalog

# Use full module path
python -m vexy_co_model_catalog version

# Check virtual environment
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### 2. API Key and Authentication Issues

#### Issue: No API Key Found
```
Error: No API key found for provider 'openai'
```

**Diagnosis:**
```bash
# Check environment variables
env | grep -i api
echo $OPENAI_API_KEY
```

**Solutions:**
```bash
# Set API key (temporary)
export OPENAI_API_KEY="your-api-key-here"

# Set API key (permanent - add to ~/.bashrc or ~/.zshrc)
echo 'export OPENAI_API_KEY="your-api-key"' >> ~/.bashrc

# Use setup wizard
vexy-model-catalog setup_wizard

# Validate after setting
vexy-model-catalog validate openai
```

#### Issue: Invalid API Key
```
Error: Authentication failed for provider 'anthropic'
HTTP 401: Invalid API key
```

**Diagnosis:**
```bash
# Test API key manually
curl -H "Authorization: Bearer $ANTHROPIC_API_KEY" \
     -H "anthropic-version: 2023-06-01" \
     https://api.anthropic.com/v1/messages

# Check key format
echo $ANTHROPIC_API_KEY | head -c 20
```

**Solutions:**
- Verify API key is correct (check your provider dashboard)
- Ensure key has proper permissions
- Check for extra spaces or characters:
```bash
export ANTHROPIC_API_KEY=$(echo "$ANTHROPIC_API_KEY" | tr -d '[:space:]')
```

#### Issue: API Key Environment Variable Not Set
```
Error: Environment variable 'GROQ_API_KEY' not found
```

**Common Environment Variables:**
```bash
# Major providers
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export GROQ_API_KEY="gsk_..."
export CEREBRAS_API_KEY="csk-..."

# OpenRouter and others
export OPENROUTER_API_KEY="sk-or-..."
export TOGETHER_API_KEY="..."
export DEEPINFRA_API_KEY="..."

# Validate all at once
vexy-model-catalog validate
```

### 3. Network and Connection Issues

#### Issue: Connection Timeouts
```
Error: Request timeout when connecting to api.openai.com
```

**Diagnosis:**
```bash
# Test connectivity
ping api.openai.com
nslookup api.openai.com

# Check proxy settings
echo $HTTP_PROXY
echo $HTTPS_PROXY

# Test with curl
curl -I https://api.openai.com/v1/models
```

**Solutions:**
```bash
# Set proxy if behind corporate firewall
export HTTP_PROXY="http://proxy.company.com:8080"
export HTTPS_PROXY="https://proxy.company.com:8080"

# Increase timeout
vexy-model-catalog fetch openai --timeout 60

# Use different DNS
export DNS_SERVER="8.8.8.8"
```

#### Issue: SSL Certificate Errors
```
Error: SSL certificate verification failed
```

**Solutions:**
```bash
# Temporary fix (not recommended for production)
export PYTHONHTTPSVERIFY=0

# Better: Update certificates
# macOS
/Applications/Python\ 3.x/Install\ Certificates.command

# Ubuntu/Debian
sudo apt update && sudo apt install ca-certificates

# CentOS/RHEL
sudo yum update ca-certificates
```

#### Issue: Rate Limiting
```
Error: Rate limit exceeded for provider 'openai'
Status: 429 Too Many Requests
```

**Diagnosis:**
```bash
# Check rate limit status
vexy-model-catalog rate_limits

# View error statistics
vexy-model-catalog production_errors
```

**Solutions:**
```bash
# Wait and retry
sleep 60
vexy-model-catalog fetch openai

# Use batch processing with delays
vexy-model-catalog fetch --all --delay 5

# Check usage limits in your provider dashboard
```

### 4. File System and Permission Issues

#### Issue: Permission Denied
```
PermissionError: [Errno 13] Permission denied: './config/aichat/models_openai.yaml'
```

**Diagnosis:**
```bash
# Check permissions
ls -la config/
whoami
pwd
```

**Solutions:**
```bash
# Fix directory permissions
chmod 755 config/
chmod 755 config/*/
chmod 644 config/*/*.yaml

# Use different directory
export VMC_LOG_DIR="$HOME/.vexy-model-catalog"
mkdir -p "$HOME/.vexy-model-catalog"

# Run with proper permissions
sudo chown -R $USER:$USER config/
```

#### Issue: Disk Space Full
```
OSError: [Errno 28] No space left on device
```

**Diagnosis:**
```bash
# Check disk space
df -h
du -sh config/
du -sh ~/.vexy-model-catalog/
```

**Solutions:**
```bash
# Clean up cache
vexy-model-catalog cache clear

# Clean up logs
vexy-model-catalog clean --logs

# Remove old files
find config/ -name "*.json" -mtime +30 -delete

# Use different storage location
export VMC_LOG_DIR="/tmp/vexy-model-catalog"
```

### 5. Configuration and Validation Issues

#### Issue: Invalid Configuration File
```
Error: Invalid YAML syntax in config file
```

**Diagnosis:**
```bash
# Validate specific config
vexy-model-catalog validate_config config/aichat/models_openai.yaml

# Check syntax manually
python -c "import yaml; yaml.safe_load(open('config/aichat/models_openai.yaml'))"
```

**Solutions:**
```bash
# Regenerate config
vexy-model-catalog fetch openai --force

# Restore from backup
vexy-model-catalog restore-configs

# Fix syntax errors manually or delete and regenerate
rm config/aichat/models_openai.yaml
vexy-model-catalog fetch openai
```

#### Issue: Provider Not Found
```
Error: Provider 'unknown_provider' not found
```

**Diagnosis:**
```bash
# List available providers
vexy-model-catalog providers

# Search for similar names
vexy-model-catalog providers | grep -i partial_name
```

**Solutions:**
```bash
# Use correct provider name
vexy-model-catalog providers  # Find exact name

# Common name variations:
# "openai" not "gpt"
# "anthropic" not "claude" 
# "groq" not "groq-ai"
```

### 6. Performance and Memory Issues

#### Issue: Slow Command Execution
```
Commands taking longer than expected to complete
```

**Diagnosis:**
```bash
# Check performance metrics
vexy-model-catalog performance stats

# Monitor resource usage
top -p $(pgrep -f vexy-model-catalog)

# Check cache hit rates
vexy-model-catalog cache status
```

**Solutions:**
```bash
# Clear and rebuild cache
vexy-model-catalog cache clear
vexy-model-catalog cache warm

# Enable performance optimizations
export VEXY_PERFORMANCE_ENABLED=true

# Use production mode for better performance
export VMC_PRODUCTION_MODE=true
```

#### Issue: High Memory Usage
```
Memory usage exceeding expected levels
```

**Diagnosis:**
```bash
# Check memory usage
vexy-model-catalog production_status

# Monitor with system tools
ps aux | grep vexy
```

**Solutions:**
```bash
# Clear caches
vexy-model-catalog cache clear

# Reduce batch size
vexy-model-catalog fetch --batch-size 5

# Use streaming mode for large datasets
vexy-model-catalog fetch --stream
```

### 7. Cache-Related Issues

#### Issue: Corrupted Cache
```
CacheError: Corrupted cache detected for key 'models_openai'
```

**Diagnosis:**
```bash
# Check cache status
vexy-model-catalog cache status

# Check integrity
vexy-model-catalog integrity verify
```

**Solutions:**
```bash
# Clear corrupted cache
vexy-model-catalog cache clear

# Clear specific cache type
vexy-model-catalog cache clear --type models

# Rebuild cache
vexy-model-catalog fetch --all --force-refresh

# Check for disk issues
fsck /path/to/cache/directory
```

#### Issue: Cache Not Working
```
Cache hit rate remains at 0%
```

**Diagnosis:**
```bash
# Check cache configuration
vexy-model-catalog cache status

# Check permissions
ls -la ~/.vexy-model-catalog/cache/
```

**Solutions:**
```bash
# Warm up cache
vexy-model-catalog cache warm

# Check disk space
df -h ~/.vexy-model-catalog/

# Reset cache system
rm -rf ~/.vexy-model-catalog/cache/
vexy-model-catalog cache init
```

## Error Code Reference

### HTTP Status Codes

| Code | Meaning | Common Causes | Solutions |
|------|---------|---------------|-----------|
| 401 | Unauthorized | Invalid API key | Check and update API key |
| 403 | Forbidden | API key lacks permissions | Check API key permissions |
| 404 | Not Found | Wrong endpoint URL | Verify provider configuration |
| 429 | Too Many Requests | Rate limit exceeded | Wait and retry, implement delays |
| 500 | Internal Server Error | Provider-side issue | Retry later, check provider status |
| 502 | Bad Gateway | Provider infrastructure issue | Retry later |
| 503 | Service Unavailable | Provider maintenance | Check provider status page |

### Application Error Codes

| Error ID | Category | Description | Action |
|----------|----------|-------------|--------|
| E001 | Config | Missing configuration file | Run setup wizard |
| E002 | Network | Connection timeout | Check network, increase timeout |
| E003 | Auth | Invalid credentials | Verify API keys |
| E004 | Validation | Schema validation failed | Check data format |
| E005 | Cache | Cache corruption | Clear and rebuild cache |
| E006 | Storage | File system error | Check permissions and disk space |
| E007 | Provider | Provider-specific error | Check provider documentation |

## Debugging Techniques

### Enable Debug Logging

```bash
# Enable debug mode
export VMC_LOG_LEVEL=DEBUG
vexy-model-catalog fetch openai

# View logs in real-time
tail -f ~/.vexy-model-catalog/logs/vexy-co-model-catalog.log

# Enable verbose output
vexy-model-catalog --verbose fetch openai
```

### Network Debugging

```bash
# Trace network calls
export PYTHONPATH=/usr/local/lib/python3.x/site-packages
python -c "
import httpx
import logging
logging.basicConfig(level=logging.DEBUG)
"

# Use proxy for inspection
export HTTP_PROXY=http://localhost:8080
export HTTPS_PROXY=http://localhost:8080
# Then run commands and inspect traffic
```

### Performance Profiling

```bash
# Enable detailed performance tracking
export VEXY_PERFORMANCE_ENABLED=true
export VMC_LOG_LEVEL=DEBUG

# Run command and check metrics
vexy-model-catalog fetch openai
vexy-model-catalog performance stats

# Profile memory usage
python -m memory_profiler -c "vexy-model-catalog fetch openai"
```

### Configuration Debugging

```bash
# Dump current configuration
vexy-model-catalog config dump

# Validate all configurations
vexy-model-catalog validate --verbose

# Show provider details
vexy-model-catalog providers --detailed
```

## Recovery Procedures

### Complete System Reset

```bash
# Stop all processes
pkill -f vexy-model-catalog

# Clear all data (CAUTION: This removes everything)
rm -rf ~/.vexy-model-catalog/

# Reinstall package
pip uninstall vexy-co-model-catalog
pip install vexy-co-model-catalog

# Run setup wizard
vexy-model-catalog setup_wizard
```

### Configuration Recovery

```bash
# Backup current config
cp -r config/ config_backup/

# Restore from backup
vexy-model-catalog restore-configs

# Regenerate all configs
vexy-model-catalog fetch --all --force
```

### Cache Recovery

```bash
# Clear problematic cache
vexy-model-catalog cache clear

# Verify integrity
vexy-model-catalog integrity verify --repair

# Warm cache with essential data
vexy-model-catalog cache warm --providers openai,anthropic
```

## Prevention Best Practices

### Environment Setup

```bash
# Use virtual environment
python -m venv venv
source venv/bin/activate

# Pin versions
pip freeze > requirements.txt

# Set up proper logging directory
mkdir -p ~/.vexy-model-catalog/logs
export VMC_LOG_DIR=~/.vexy-model-catalog
```

### Configuration Management

```bash
# Regular backups
vexy-model-catalog backup-configs

# Validation before deployment
vexy-model-catalog validate
vexy-model-catalog production_readiness

# Monitor health regularly
vexy-model-catalog health
```

### Monitoring and Alerts

```bash
# Set up health monitoring
echo "0 */6 * * * /usr/local/bin/vexy-model-catalog health" | crontab -

# Log monitoring
tail -f ~/.vexy-model-catalog/logs/errors.log | grep -i critical

# Performance monitoring
vexy-model-catalog performance stats --json > /tmp/performance.json
```

## Getting Additional Help

### Built-in Help System

```bash
# Comprehensive help
vexy-model-catalog help

# Command-specific help
vexy-model-catalog help fetch

# Interactive setup
vexy-model-catalog setup_wizard
```

### System Information Collection

When reporting issues, collect this information:

```bash
# System info
vexy-model-catalog version
python --version
pip list | grep vexy
uname -a

# Configuration status
vexy-model-catalog validate
vexy-model-catalog health
vexy-model-catalog production_status

# Recent logs
tail -50 ~/.vexy-model-catalog/logs/vexy-co-model-catalog.log

# Error statistics
vexy-model-catalog production_errors
```

### Log File Locations

Default log file locations by platform:

- **macOS**: `~/Library/Logs/vexy-co-model-catalog/`
- **Linux**: `~/.local/share/vexy-co-model-catalog/logs/`
- **Windows**: `%LOCALAPPDATA%\vexy-co-model-catalog\logs\`

Custom location: Set `VMC_LOG_DIR` environment variable.

### Community Resources

- **Documentation**: [User Guide](USER_GUIDE.md), [API Reference](API_REFERENCE.md)
- **Examples**: Check `docs/examples/` directory
- **Issues**: Report bugs on GitHub repository
- **Discussions**: Community discussions and Q&A

This troubleshooting guide should help resolve most common issues. If you encounter problems not covered here, please create a detailed issue report including system information, error messages, and steps to reproduce the problem.