"""
this_file: src/vexy_co_model_catalog/core/provider.py

Minimal provider abstractions and data structures.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Any


class ProviderKind(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    URL = "url"


@dataclass
class Model:
    id: str
    provider: str
    name: str | None = None
    context_length: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_tokens: int | None = None  # Legacy field for backward compatibility
    input_price: float | None = None
    output_price: float | None = None
    supports_functions: bool = False
    supports_vision: bool = False
    supports_streaming: bool = True
    created: int | None = None
    description: str | None = None
    raw: dict[str, Any] | None = None


@dataclass
class ProviderConfig:
    name: str
    kind: ProviderKind
    api_key_env: str | None = None
    base_url_env: str | None = None
    base_url: str | None = None
    headers: dict[str, str] | None = None
    timeout: float = 15.0
    verify_ssl: bool = True

    def get_base_url(self) -> str | None:
        if self.base_url_env and os.getenv(self.base_url_env):
            return os.getenv(self.base_url_env)
        return self.base_url

    def build_models_url(self) -> str:
        """Build the models endpoint URL for this provider."""
        base_url = self.get_base_url()
        if not base_url:
            msg = f"No base URL configured for provider {self.name}"
            raise ValueError(msg)

        if self.kind == ProviderKind.URL:
            # URL providers return the direct URL to the JSON file
            return base_url

        # For OpenAI-compatible and Anthropic APIs, append /models
        base = base_url.rstrip("/")
        return f"{base}/models"


# Provider URL configuration mapping environment variables to base URLs
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


def setup_url_environment() -> None:
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


def parse_provider_config() -> list[ProviderConfig]:
    """Parse the provider configuration and return a list of ProviderConfig objects."""
    # Ensure URL environment variables are set up
    setup_url_environment()

    providers = []
    for line in PROVIDER_CONFIG.strip().split("\n"):
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue

        name, kind_str, api_key_env, url_env = parts

        # Map kind string to enum, handling the oai->openai mapping
        kind_mapping = {
            "oai": ProviderKind.OPENAI,
            "ant": ProviderKind.ANTHROPIC,
            "url": ProviderKind.URL,
        }
        kind = kind_mapping.get(kind_str, ProviderKind.OPENAI)

        # Handle empty API key env for direct URLs
        api_key_env = api_key_env if api_key_env else None

        provider = ProviderConfig(
            name=name,
            kind=kind,
            api_key_env=api_key_env,
            base_url_env=url_env,
        )
        providers.append(provider)

    return providers


def get_provider_by_name(name: str) -> ProviderConfig | None:
    """Get a provider configuration by name."""
    providers = parse_provider_config()
    for provider in providers:
        if provider.name == name:
            return provider
    return None


def get_all_providers() -> list[ProviderConfig]:
    """Get all available provider configurations."""
    return parse_provider_config()
