#!/usr/bin/env -S uv run -s
# /// script
# dependencies = ["httpx", "fire", "rich", "loguru", "asyncio", "tomli-w"]
# ///
# this_file: vexy_co_model_catalog/dump_models.py

"""
Model dumper for various AI API providers.
Ports functionality from dump_models.sh to Python with enhanced URL handling.
"""

import asyncio
import json
import os
import sys
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from pathlib import Path

import fire  # type: ignore
import httpx
import tomli_w
from loguru import logger
from rich.console import Console

console = Console()


class ProviderKind(Enum):
    OAI = "oai"
    ANT = "ant"
    URL = "url"


@dataclass(frozen=True)
class Provider:
    name: str
    kind: ProviderKind
    api_key_env: str | None = None
    url_env: str | None = None

    def build_models_url(self) -> str:
        base_url = os.environ.get(self.url_env or "") if self.url_env else None

        if self.kind == ProviderKind.URL and base_url:
            return base_url

        if not base_url:
            raise ValueError(
                f"No base URL configured for provider {self.name} (missing {self.url_env})"
            )

        # For OpenAI-compatible and Anthropic APIs, append /models to the base URL
        base = base_url.rstrip("/")
        return f"{base}/models"


# Provider URL config: api_url_env, base_url
PROVIDER_URL_CONFIG = """
AIHORDE_API_OPENAI, https://oai.aihorde.net/v1
ANTHROPIC_API_OPENAI, https://api.anthropic.com/v1
ATLASCLOUD_API_KEY, https://api.atlascloud.ai/v1
AVIAN_API_OPENAI, https://api.avian.io/v1
BASETEN_API_OPENAI, https://inference.baseten.co/v1
CEREBRAS_API_OPENAI, https://api.cerebras.ai/v1
CHUTES_API_OPENAI, https://llm.chutes.ai/v1
DEEPINFRA_API_OPENAI, https://api.deepinfra.com/v1/openai
DEEPSEEK_API_OPENAI, https://api.deepseek.com/v1
ENFER_API_OPENAI, https://api.enfer.ai/v1
FEATHERLESS_API_OPENAI, https://api.featherless.ai/v1
FIREWORKS_API_OPENAI, https://api.fireworks.ai/inference/v1
FRIENDLI_API_OPENAI, https://api.friendli.ai/serverless/v1
GEMINI_API_OPENAI, https://generativelanguage.googleapis.com/v1beta/openai
GROQ_API_OPENAI, https://api.groq.com/openai/v1
HUGGINGFACE_API_OPENAI, https://router.huggingface.co/v1
HYPERBOLIC_API_OPENAI, https://api.hyperbolic.xyz/v1
INFERENCENET_API_OPENAI, https://api.inference.net/v1
INFERMATIC_API_OPENAI, https://api.totalgpt.ai/v1
LITELLM_API_OPENAI, https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json
LLM7_API_OPENAI, https://api.llm7.io/v1
LMSTUDIO_API_OPENAI, http://othello.local:1234/v1
MANCER_API_OPENAI, https://neuro.mancer.tech/oai/v1
MISTRAL_API_OPENAI, https://api.mistral.ai/v1
MOONSHOT_API_OPENAI, https://api.moonshot.ai/v1
MORPHLLM_API_OPENAI, https://api.morphllm.com/v1
NEBIUS_API_OPENAI, https://api.studio.nebius.com/v1
NINETEENAI_API_OPENAI, https://api.nineteen.ai/v1
NOVITA_API_OPENAI, https://api.novita.ai/v3/openai
OPENAI_API_OPENAI, https://api.openai.com/v1
OPENROUTER_API_OPENAI, https://openrouter.ai/api/v1
PARASAIL_API_OPENAI, https://api.parasail.io/v1
POE_API_OPENAI, https://api.poe.com/v1
POLLINATIONS_API_OPENAI, https://text.pollinations.ai/openai
REDPILL_API_OPENAI, https://api.redpill.ai/v1
SAMBANOVA_API_OPENAI, https://api.sambanova.ai/v1
TARGON_API_OPENAI, https://api.targon.com/v1
TOGETHERAI_API_OPENAI, https://api.together.xyz/v1
XAI_API_OPENAI, https://api.x.ai/v1
ZAI_API_OPENAI, https://api.z.ai/api/paas/v4
""".strip()

# Provider configuration: name, kind, api_key_env, url_env
PROVIDER_CONFIG = """
aihorde, oai, AIHORDE_KEY, AIHORDE_API_OPENAI
anthropic, ant, ANTHROPIC_API_KEY, ANTHROPIC_API_OPENAI
atlascloud, oai, ATLASCLOUD_API_KEY, ATLASCLOUD_API_OPENAI
avian, oai, AVIAN_API_KEY, AVIAN_API_OPENAI
baseten, oai, BASETEN_API_KEY, BASETEN_API_OPENAI
cerebras, oai, CEREBRAS_API_KEY, CEREBRAS_API_OPENAI
chutes, oai, CHUTES_API_KEY, CHUTES_API_OPENAI
deepinfra, oai, DEEPINFRA_API_KEY, DEEPINFRA_API_OPENAI
deepseek, oai, DEEPSEEK_API_KEY, DEEPSEEK_API_OPENAI
enfer, oai, ENFER_API_KEY, ENFER_API_OPENAI
featherless, oai, FEATHERLESS_API_KEY, FEATHERLESS_API_OPENAI
fireworks, oai, FIREWORKS_API_KEY, FIREWORKS_API_OPENAI
friendli, oai, FRIENDLI_TOKEN, FRIENDLI_API_OPENAI
gemini, oai, GOOGLE_API_KEY, GEMINI_API_OPENAI
groq, oai, GROQ_API_KEY, GROQ_API_OPENAI
huggingface, oai, HUGGINGFACEHUB_API_TOKEN, HUGGINGFACE_API_OPENAI
hyperbolic, oai, HYPERBOLIC_API_KEY, HYPERBOLIC_API_OPENAI
inference, oai, INFERENCENET_API_KEY, INFERENCENET_API_OPENAI
infermatic, oai, INFERMATIC_API_KEY, INFERMATIC_API_OPENAI
litellm, url, , LITELLM_API_OPENAI
llm7, oai, LLM7_API_KEY, LLM7_API_OPENAI
lmstudio, oai, LMSTUDIO_API_KEY, LMSTUDIO_API_OPENAI
mancer, oai, MANCER_API_KEY, MANCER_API_OPENAI
mistral, oai, MISTRAL_API_KEY, MISTRAL_API_OPENAI
moonshot, oai, MOONSHOT_API_KEY, MOONSHOT_API_OPENAI
morphllm, oai, MORPHLLM_API_KEY, MORPHLLM_API_OPENAI
nebius, oai, NEBIUS_API_KEY, NEBIUS_API_OPENAI
nineteenai, oai, NINETEENAI_API_KEY, NINETEENAI_API_OPENAI
novita, oai, NOVITA_API_KEY, NOVITA_API_OPENAI
openai, oai, OPENAI_API_KEY, OPENAI_API_OPENAI
openrouter, oai, OPENROUTER_API_KEY, OPENROUTER_API_OPENAI
parasail, oai, PARASAIL_API_KEY, PARASAIL_API_OPENAI
poe, oai, POE_API_KEY, POE_API_OPENAI
pollinations, oai, POLLINATIONS_API_KEY, POLLINATIONS_API_OPENAI
redpill, oai, REDPILL_API_KEY, REDPILL_API_OPENAI
sambanova, oai, SAMBANOVA_API_KEY, SAMBANOVA_API_OPENAI
targon, oai, TARGON_API_KEY, TARGON_API_OPENAI
togetherai, oai, TOGETHERAI_API_KEY, TOGETHERAI_API_OPENAI
xai, oai, XAI_API_KEY, XAI_API_OPENAI
zai, oai, ZAI_API_KEY, ZAI_API_OPENAI
""".strip()


class ModelDumper:
    """Dumps model lists from various AI API providers."""

    ANTHROPIC_VERSION = "2023-06-01"

    def __init__(self):
        """Initialize with provider configurations."""
        # Change to script directory to ensure all outputs go to the same folder
        script_dir = Path(__file__).parent
        os.chdir(script_dir)

        self._setup_url_environment()
        self.providers = self._parse_provider_config()
        self._init_failed_models()

    def _setup_url_environment(self):
        """Set up environment variables for provider URLs from PROVIDER_URL_CONFIG."""
        for line in PROVIDER_URL_CONFIG.strip().split("\n"):
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 2:
                continue

            env_var, url = parts
            # Only set if not already present in environment
            if env_var not in os.environ:
                os.environ[env_var] = url

    def _parse_provider_config(self) -> list[Provider]:
        """Parse the simple text-based provider configuration."""
        providers = []
        for line in PROVIDER_CONFIG.strip().split("\n"):
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 4:
                continue

            name, kind_str, api_key_env, url_env = parts

            # Map kind string to enum
            kind_map = {
                "oai": ProviderKind.OAI,
                "ant": ProviderKind.ANT,
                "url": ProviderKind.URL,
            }
            kind = kind_map.get(kind_str, ProviderKind.OAI)

            # Handle empty API key env for direct URLs
            api_key_env = api_key_env if api_key_env else None

            providers.append(
                Provider(name=name, kind=kind, api_key_env=api_key_env, url_env=url_env)
            )

        return providers

    def _init_failed_models(self):
        """Initialize failed models tracking."""
        self.failed_models_file = Path("failed_models.json")
        self.failed_models = self._load_failed_models()

    def _load_failed_models(self):
        """Load failed models configuration from JSON file."""
        if self.failed_models_file.exists():
            try:
                with open(self.failed_models_file, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load failed models config: {e}")
        return {}

    def _save_failed_models(self):
        """Save failed models configuration to JSON file."""
        try:
            with open(self.failed_models_file, "w", encoding="utf-8") as f:
                json.dump(self.failed_models, f, indent=2)
        except OSError as e:
            logger.error(f"Failed to save failed models config: {e}")

    def _files_exist(self, api_name: str) -> bool:
        """Check if output files already exist for an API."""
        json_file = Path(f"models_{api_name}.json")
        txt_file = Path(f"models_{api_name}.txt")
        toml_file = Path(f"models_{api_name}.toml")
        return json_file.exists() and txt_file.exists() and toml_file.exists()

    def _build_curl(self, provider: Provider, models_url: str) -> str:
        if provider.kind == ProviderKind.ANT:
            return (
                'curl -s -H "x-api-key: $%s" -H "anthropic-version: %s" "%s" | jq'
            ) % (
                provider.api_key_env or "",
                self.ANTHROPIC_VERSION,
                models_url,
            )
        if provider.kind == ProviderKind.URL:
            return f'curl "{models_url}" > models_{provider.name}.json'
        return 'curl -s -H "Authorization: Bearer ${}" "{}" | jq'.format(
            provider.api_key_env or "",
            models_url,
        )

    def _sort_json_data(self, data):
        """Sort JSON data based on its format."""
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                # OpenAI format with .data array
                data["data"] = sorted(
                    data["data"],
                    key=lambda x: x.get("id", ""),
                )
                return data
            else:
                # Object format (sort by keys)
                return dict(sorted(data.items()))
        elif isinstance(data, list):
            # Direct array format
            return sorted(data, key=lambda x: x.get("id", ""))
        else:
            return data

    def _extract_model_ids(self, data):
        """Extract model IDs from JSON data."""
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                # OpenAI format
                return [item.get("id", "") for item in data["data"] if item.get("id")]
            else:
                # Object format (keys are model IDs)
                return [key for key in data.keys() if key != "sample_spec"]
        elif isinstance(data, list):
            # Array format
            return [item.get("id", "") for item in data if item.get("id")]
        else:
            return []

    def _get_provider_base_url(self, provider: Provider) -> str:
        """Get the base URL for a provider."""
        base_url = os.environ.get(provider.url_env or "") if provider.url_env else None
        if base_url:
            return base_url.rstrip("/models").rstrip("/")
        return ""

    def _generate_profile_name(self, provider_name: str, model_id: str) -> str:
        """Generate a profile name from provider and model ID."""
        # Clean the model ID to create a valid profile name
        model_clean = model_id.replace("/", "_").replace("-", "_").replace(".", "_")
        # Take only alphanumeric and underscores
        import re

        model_clean = re.sub(r"[^a-zA-Z0-9_]", "", model_clean).lower()

        # Truncate if too long and append provider prefix
        if len(model_clean) > 20:
            model_clean = model_clean[:20]

        # Use provider abbreviation
        provider_abbrev = provider_name[:3] if len(provider_name) > 3 else provider_name
        return f"{provider_abbrev}_{model_clean}"

    def _extract_model_details(self, data, model_id: str):
        """Extract model details from JSON data for a specific model."""
        if isinstance(data, dict):
            if "data" in data and isinstance(data["data"], list):
                # OpenAI format - find the model in the data array
                for item in data["data"]:
                    if item.get("id") == model_id:
                        return item
            else:
                # Object format - model ID is the key
                return data.get(model_id, {})
        elif isinstance(data, list):
            # Array format - find the model
            for item in data:
                if item.get("id") == model_id:
                    return item
        return {}

    def _generate_toml_config(self, provider: Provider, sorted_data) -> dict:
        """Generate TOML configuration for a provider and its models."""
        config = {}

        # Add provider configuration
        provider_key = f"model_providers.{provider.name}"
        provider_config = {
            "base_url": self._get_provider_base_url(provider),
            "name": provider.name,
            "wire_api": "chat",  # Default to chat API
        }

        if provider.api_key_env:
            provider_config["env_key"] = provider.api_key_env

        config["model_providers"] = {provider.name: provider_config}

        # Add model profiles
        config["profiles"] = {}
        model_ids = self._extract_model_ids(sorted_data)

        for model_id in model_ids:
            profile_name = self._generate_profile_name(provider.name, model_id)
            model_details = self._extract_model_details(sorted_data, model_id)

            profile_config = {
                "approval_policy": "never",
                "model": model_id,
                "model_provider": provider.name,
                "model_reasoning_effort": "high",
            }

            # Add context window if available
            context_length = None
            if isinstance(model_details, dict):
                # Try various field names for context length
                context_length = (
                    model_details.get("context_length")
                    or model_details.get("context_window")
                    or model_details.get("max_context_length")
                    or model_details.get("max_tokens")
                    or model_details.get("max_model_len")  # For chutes provider
                )

                # For OpenRouter format
                if "top_provider" in model_details:
                    context_length = model_details["top_provider"].get("context_length")

            if context_length and isinstance(context_length, (int, str)):
                try:
                    profile_config["model_context_window"] = int(context_length)
                except (ValueError, TypeError):
                    pass

            # Add max output tokens if available
            max_output_tokens = None
            if isinstance(model_details, dict):
                max_output_tokens = (
                    model_details.get("max_completion_tokens")
                    or model_details.get("max_output_tokens")
                    or model_details.get("max_tokens_output")
                )

                # For OpenRouter format
                if "top_provider" in model_details:
                    max_output_tokens = model_details["top_provider"].get(
                        "max_completion_tokens"
                    )

            if max_output_tokens and isinstance(max_output_tokens, (int, str)):
                try:
                    profile_config["model_max_output_tokens"] = int(max_output_tokens)
                except (ValueError, TypeError):
                    pass

            config["profiles"][profile_name] = profile_config

        return config

    async def _process_provider(self, provider: Provider, force: bool = False) -> bool:
        """Process a provider into JSON and TXT files."""
        # Build and display the fake curl call for every provider
        try:
            models_url = provider.build_models_url()
            curl_cmd = self._build_curl(provider, models_url)
            console.print(f"[dim]{curl_cmd}[/dim]")
        except Exception as e:
            console.print(
                f"[dim red]Error building URL for {provider.name} - {e}[/dim red]"
            )

        if not force and self._files_exist(provider.name):
            logger.info(
                f"[SKIP] {provider.name} - files exist (use --force to overwrite)"
            )
            return True

        api_key = os.environ.get(provider.api_key_env or "")
        if provider.kind != ProviderKind.URL and not api_key:
            logger.warning(
                f"[SKIP] {provider.name} - missing API key in {provider.api_key_env}"
            )
            return False

        logger.info(f"Processing {provider.name}")

        async with httpx.AsyncClient() as client:
            try:
                if provider.kind == ProviderKind.ANT:
                    headers = {
                        "x-api-key": api_key or "",
                        "anthropic-version": self.ANTHROPIC_VERSION,
                        "User-Agent": "ModelDumper/1.0",
                    }
                    response = await client.get(
                        models_url,
                        headers=headers,
                        timeout=30,
                    )
                elif provider.kind == ProviderKind.URL:
                    response = await client.get(
                        models_url,
                        timeout=30,
                    )
                else:
                    headers = {
                        "Authorization": f"Bearer {api_key}",
                        "User-Agent": "ModelDumper/1.0",
                    }
                    response = await client.get(
                        models_url,
                        headers=headers,
                        timeout=30,
                    )

                response.raise_for_status()

                data = response.json()

                # Special handling for chutes provider - supplement with chutes API data
                if provider.name == "chutes":
                    chutes_data = await self._fetch_chutes_data(client)
                    data = self._merge_chutes_data(data, chutes_data)

                sorted_data = self._sort_json_data(data)

                json_file = Path(f"models_{provider.name}.json")
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(sorted_data, f, indent=2)

                model_ids = self._extract_model_ids(sorted_data)
                txt_file = Path(f"models_{provider.name}.txt")
                with open(txt_file, "w", encoding="utf-8") as f:
                    f.write("\n".join(model_ids))

                # Generate TOML configuration
                toml_config = self._generate_toml_config(provider, sorted_data)
                toml_file = Path(f"models_{provider.name}.toml")
                with open(toml_file, "wb") as f:
                    tomli_w.dump(toml_config, f)

                logger.success(
                    f"✓ {provider.name}: {len(model_ids)} models saved (JSON, TXT, TOML)"
                )

                if provider.name in self.failed_models:
                    self.failed_models[provider.name]["failed"] = False
                    self.failed_models[provider.name]["last_success"] = (
                        response.headers.get("Date", "")
                    )

                return True
            except httpx.RequestError as e:
                logger.error(f"✗ {provider.name}: Network error - {e}")
                self._mark_api_failed(provider.name, str(e))
                return False
            except json.JSONDecodeError as e:
                logger.error(f"✗ {provider.name}: Invalid JSON response - {e}")
                self._mark_api_failed(provider.name, f"JSON decode error: {e}")
                return False
            except Exception as e:
                logger.error(f"✗ {provider.name}: Unexpected error - {e}")
                self._mark_api_failed(provider.name, f"Unexpected error: {e}")
                return False

    def _mark_api_failed(self, api_name: str, error: str):
        """Mark an API as failed in the configuration."""
        self.failed_models[api_name] = {
            "failed": True,
            "error": error,
            "last_attempt": datetime.now().isoformat(),
        }

    async def _fetch_chutes_data(self, client: httpx.AsyncClient) -> dict | None:
        """
        Fetch chutes data from the Chutes API.

        Returns additional model information from the chutes API
        to supplement the OpenAI-compatible API response.
        """
        chutes_api_url = (
            "https://api.chutes.ai/chutes/?include_public=true&include_schemas=false"
        )

        try:
            # Check if API key is available
            api_key = os.environ.get("CHUTES_API_KEY")
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = api_key

            response = await client.get(
                chutes_api_url,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()

            chutes_data = response.json()

            # Save the raw chutes data for debugging/reference
            chutes_file = Path("chutes_chutes.json")
            with open(chutes_file, "w", encoding="utf-8") as f:
                json.dump(chutes_data, f, indent=2)

            return chutes_data

        except Exception as e:
            logger.warning(f"Failed to fetch chutes API data: {e}")
            return None

    def _merge_chutes_data(self, openai_data: dict, chutes_data: dict | None) -> dict:
        """
        Merge chutes API data with OpenAI API model data.

        Enhances OpenAI model entries with additional information from chutes API.
        """
        if not chutes_data or "items" not in chutes_data:
            return openai_data

        # Create a lookup map from chutes data by model name
        chutes_lookup = {}
        for chute in chutes_data["items"]:
            if chute.get("name"):
                chutes_lookup[chute["name"]] = chute

        # Enhance OpenAI data with chutes information
        if isinstance(openai_data, dict) and "data" in openai_data:
            for model in openai_data["data"]:
                model_id = model.get("id", "")
                if model_id in chutes_lookup:
                    chute = chutes_lookup[model_id]

                    # Add chutes-specific fields
                    model["chute_id"] = chute.get("chute_id")
                    model["tagline"] = chute.get("tagline", "")
                    model["public"] = chute.get("public", True)
                    model["invocation_count"] = chute.get("invocation_count", 0)
                    model["hot"] = chute.get("hot", False)

                    # Add pricing information from chutes if available
                    if "current_estimated_price" in chute:
                        price_info = chute["current_estimated_price"]
                        if "per_million_tokens" in price_info:
                            model["chutes_pricing"] = price_info["per_million_tokens"]

                    # Add GPU requirements
                    if "supported_gpus" in chute:
                        model["supported_gpus"] = chute["supported_gpus"]

                    if "node_selector" in chute:
                        node_selector = chute["node_selector"]
                        if "gpu_count" in node_selector:
                            model["gpu_count"] = node_selector["gpu_count"]
                        if "min_vram_gb_per_gpu" in node_selector:
                            model["min_vram_gb_per_gpu"] = node_selector[
                                "min_vram_gb_per_gpu"
                            ]

                    # Add instance information
                    if "instances" in chute:
                        active_instances = [
                            inst
                            for inst in chute["instances"]
                            if inst.get("active", False)
                        ]
                        model["active_instances"] = len(active_instances)
                        model["total_instances"] = len(chute["instances"])

        return openai_data

    # Special-case processor removed; unified into _process_provider

    async def dump_models(
        self,
        force: bool = False,
        failed_rescan: bool = False,
        verbose: bool = False,
    ):
        """
        Dump model lists from all configured APIs.

        Args:
            force: Overwrite existing files
            failed_rescan: Only process APIs marked as failed
            verbose: Enable verbose logging
        """
        # Configure logging
        logger.remove()
        log_level = "DEBUG" if verbose else "INFO"
        logger.add(
            sys.stderr,
            level=log_level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | {message}"
            ),
        )

        logger.info("Starting model dump process")

        successful = 0
        failed = 0

        # Determine which providers to process
        if failed_rescan:
            failed_names = [
                name
                for name, cfg in self.failed_models.items()
                if cfg.get("failed", False)
            ]
            if not failed_names:
                logger.info("No failed APIs found to rescan")
                return
            logger.info(f"Rescanning {len(failed_names)} failed APIs")
            providers = [p for p in self.providers if p.name in failed_names]
        else:
            providers = list(self.providers)

        # Process providers in parallel
        tasks = [self._process_provider(provider, force) for provider in providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
                failed += 1
            elif result:
                successful += 1
            else:
                failed += 1

        # Save failed models configuration
        self._save_failed_models()

        # Summary
        total = successful + failed
        logger.info(f"Process complete: {successful}/{total} APIs successful")
        if failed > 0:
            logger.warning(
                f"{failed} APIs failed - use --failed_rescan to retry failed APIs"
            )


class AsyncModelDumper(ModelDumper):
    """Synchronous wrapper for the async ModelDumper."""

    def dump_models(
        self,
        force: bool = False,
        failed_rescan: bool = False,
        verbose: bool = False,
    ):
        """Synchronous wrapper for the async dump_models method."""
        return asyncio.run(super().dump_models(force, failed_rescan, verbose))


def main():
    """Main entry point for the model dumper."""
    dumper = AsyncModelDumper()
    fire.Fire(dumper.dump_models)


if __name__ == "__main__":
    main()
