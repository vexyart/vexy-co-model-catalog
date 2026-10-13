# Providers

The provider list is fixed in code, in `src/vexy_co_model_catalog/core/provider.py`. It holds 40 providers. You cannot add or remove providers from the command line; `vmc providers add` and `vmc providers remove` only print a notice.

## Provider types

Each provider has one of three types. The type decides which URL is requested and how the API key is sent.

| Type | Request URL | Authentication header |
| --- | --- | --- |
| `openai` | `<base URL>/models` | `Authorization: Bearer <key>` |
| `anthropic` | `<base URL>/models` | `x-api-key: <key>` and `anthropic-version: 2023-06-01` |
| `url` | `<base URL>` itself | none |

Every request also carries `User-Agent: ModelDumper/1.0`.

## How the base URL is chosen

Each provider names two environment variables: one for the API key and one for the base URL.

1. When the package loads the provider table, it sets each base-URL variable to its default value, unless that variable is already set in your environment.
2. When a request is built, the tool reads the base-URL variable.

So you can point any provider at another endpoint by exporting its base-URL variable before you run `vmc`:

```bash
export LMSTUDIO_API_OPENAI="http://localhost:1234/v1"
vmc fetch lmstudio
```

The tool then requests `http://localhost:1234/v1/models`.

## Provider table

| Name | Type | API key variable | Base-URL variable | Default base URL |
| --- | --- | --- | --- | --- |
| `aihorde` | openai | `AIHORDE_KEY` | `AIHORDE_API_OPENAI` | `https://oai.aihorde.net/v1` |
| `anthropic` | anthropic | `ANTHROPIC_API_KEY` | `ANTHROPIC_API_OPENAI` | `https://api.anthropic.com/v1` |
| `atlascloud` | openai | `ATLASCLOUD_API_KEY` | `ATLASCLOUD_API_OPENAI` | (none, see note) |
| `avian` | openai | `AVIAN_API_KEY` | `AVIAN_API_OPENAI` | `https://api.avian.io/v1` |
| `baseten` | openai | `BASETEN_API_KEY` | `BASETEN_API_OPENAI` | `https://inference.baseten.co/v1` |
| `cerebras` | openai | `CEREBRAS_API_KEY` | `CEREBRAS_API_OPENAI` | `https://api.cerebras.ai/v1` |
| `chutes` | openai | `CHUTES_API_KEY` | `CHUTES_API_OPENAI` | `https://llm.chutes.ai/v1` |
| `deepinfra` | openai | `DEEPINFRA_API_KEY` | `DEEPINFRA_API_OPENAI` | `https://api.deepinfra.com/v1/openai` |
| `deepseek` | openai | `DEEPSEEK_API_KEY` | `DEEPSEEK_API_OPENAI` | `https://api.deepseek.com/v1` |
| `enfer` | openai | `ENFER_API_KEY` | `ENFER_API_OPENAI` | `https://api.enfer.ai/v1` |
| `featherless` | openai | `FEATHERLESS_API_KEY` | `FEATHERLESS_API_OPENAI` | `https://api.featherless.ai/v1` |
| `fireworks` | openai | `FIREWORKS_API_KEY` | `FIREWORKS_API_OPENAI` | `https://api.fireworks.ai/inference/v1` |
| `friendli` | openai | `FRIENDLI_TOKEN` | `FRIENDLI_API_OPENAI` | `https://api.friendli.ai/serverless/v1` |
| `gemini` | openai | `GOOGLE_API_KEY` | `GEMINI_API_OPENAI` | `https://generativelanguage.googleapis.com/v1beta/openai` |
| `groq` | openai | `GROQ_API_KEY` | `GROQ_API_OPENAI` | `https://api.groq.com/openai/v1` |
| `huggingface` | openai | `HUGGINGFACEHUB_API_TOKEN` | `HUGGINGFACE_API_OPENAI` | `https://router.huggingface.co/v1` |
| `hyperbolic` | openai | `HYPERBOLIC_API_KEY` | `HYPERBOLIC_API_OPENAI` | `https://api.hyperbolic.xyz/v1` |
| `inference` | openai | `INFERENCENET_API_KEY` | `INFERENCENET_API_OPENAI` | `https://api.inference.net/v1` |
| `infermatic` | openai | `INFERMATIC_API_KEY` | `INFERMATIC_API_OPENAI` | `https://api.totalgpt.ai/v1` |
| `litellm` | url | (none) | `LITELLM_API_OPENAI` | `https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json` |
| `llm7` | openai | `LLM7_API_KEY` | `LLM7_API_OPENAI` | `https://api.llm7.io/v1` |
| `lmstudio` | openai | `LMSTUDIO_API_KEY` | `LMSTUDIO_API_OPENAI` | `http://othello.local:1234/v1` |
| `mancer` | openai | `MANCER_API_KEY` | `MANCER_API_OPENAI` | `https://neuro.mancer.tech/oai/v1` |
| `mistral` | openai | `MISTRAL_API_KEY` | `MISTRAL_API_OPENAI` | `https://api.mistral.ai/v1` |
| `moonshot` | openai | `MOONSHOT_API_KEY` | `MOONSHOT_API_OPENAI` | `https://api.moonshot.ai/v1` |
| `morphllm` | openai | `MORPHLLM_API_KEY` | `MORPHLLM_API_OPENAI` | `https://api.morphllm.com/v1` |
| `nebius` | openai | `NEBIUS_API_KEY` | `NEBIUS_API_OPENAI` | `https://api.studio.nebius.com/v1` |
| `nineteenai` | openai | `NINETEENAI_API_KEY` | `NINETEENAI_API_OPENAI` | `https://api.nineteen.ai/v1` |
| `novita` | openai | `NOVITA_API_KEY` | `NOVITA_API_OPENAI` | `https://api.novita.ai/v3/openai` |
| `openai` | openai | `OPENAI_API_KEY` | `OPENAI_API_OPENAI` | `https://api.openai.com/v1` |
| `openrouter` | openai | `OPENROUTER_API_KEY` | `OPENROUTER_API_OPENAI` | `https://openrouter.ai/api/v1` |
| `parasail` | openai | `PARASAIL_API_KEY` | `PARASAIL_API_OPENAI` | `https://api.parasail.io/v1` |
| `poe` | openai | `POE_API_KEY` | `POE_API_OPENAI` | `https://api.poe.com/v1` |
| `pollinations` | openai | `POLLINATIONS_API_KEY` | `POLLINATIONS_API_OPENAI` | `https://text.pollinations.ai/openai` |
| `redpill` | openai | `REDPILL_API_KEY` | `REDPILL_API_OPENAI` | `https://api.redpill.ai/v1` |
| `sambanova` | openai | `SAMBANOVA_API_KEY` | `SAMBANOVA_API_OPENAI` | `https://api.sambanova.ai/v1` |
| `targon` | openai | `TARGON_API_KEY` | `TARGON_API_OPENAI` | `https://api.targon.com/v1` |
| `togetherai` | openai | `TOGETHERAI_API_KEY` | `TOGETHERAI_API_OPENAI` | `https://api.together.xyz/v1` |
| `xai` | openai | `XAI_API_KEY` | `XAI_API_OPENAI` | `https://api.x.ai/v1` |
| `zai` | openai | `ZAI_API_KEY` | `ZAI_API_OPENAI` | `https://api.z.ai/api/paas/v4` |

!!! note "atlascloud has no default base URL"
    In the default URL table, the `atlascloud` entry is keyed `ATLASCLOUD_API_KEY` instead of `ATLASCLOUD_API_OPENAI`. As a result the package sets no default for `ATLASCLOUD_API_OPENAI`, and it also writes the URL into `ATLASCLOUD_API_KEY` when that variable is not already set. Set both variables yourself to use this provider:

    ```bash
    export ATLASCLOUD_API_KEY="..."
    export ATLASCLOUD_API_OPENAI="https://api.atlascloud.ai/v1"
    ```

!!! note "lmstudio points at a local host name"
    The default for `lmstudio` is `http://othello.local:1234/v1`, a machine on the author's network. Plain `http://` is only accepted for `localhost` and `127.0.0.1`, so the default is rejected by the URL check. Set `LMSTUDIO_API_OPENAI` to your own server, for example `http://localhost:1234/v1`.

## Providers with special handling

`litellm`
:   A `url` provider. Its base URL is the JSON file `model_prices_and_context_window.json` in the LiteLLM repository, so no key is needed. The response is an object keyed by model ID; the `sample_spec` key is left out of the model ID list.

`chutes`
:   After the normal `/models` request, the tool also requests `https://api.chutes.ai/chutes/?include_public=true&include_schemas=false` and merges the extra fields into the matching model entries. If the second request fails, the tool logs a warning and keeps the first response.

`lmstudio`
:   The only provider in the table allowed to use plain `http://`, and only for `localhost` or `127.0.0.1`.

## URL check

Before each request the tool checks the URL. It must use `https://`; plain `http://` is accepted only for `localhost` and `127.0.0.1`, and only for `lmstudio`. Host names that contain patterns such as `metadata`, `internal`, `private`, `169.254` or `10.` are rejected. A rejected URL fails the provider with a message that starts with `SECURITY: URL failed security validation`.

## Response formats

The tool accepts three response shapes and extracts model IDs from each:

| Shape | Example | Model IDs taken from |
| --- | --- | --- |
| OpenAI style | `{"data": [{"id": "gpt-4o"}, …]}` | each `data[].id`; `data` is sorted by `id` |
| Object keyed by ID | `{"gpt-4o": {…}, …}` | the keys, except `sample_spec`; keys are sorted |
| Plain array | `[{"id": "gpt-4o"}, …]` | each `id`; the array is sorted by `id` |

## Check a provider

```bash
vmc providers show groq      # type, key variable, base URL, failure stats
vmc validate groq            # is the key set? is a base URL available?
vmc provider_health groq     # does the endpoint answer?
```
