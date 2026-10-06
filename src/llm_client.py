"""Provider-agnostic LLM client for Module 4.

Supports every major OpenAI-compatible endpoint from a single .env block —
the user picks a provider, pastes an API key, optionally overrides the model
name, and the report generator just works:

    PROVIDER=gemini            # openai | gemini | openrouter | nvidia
    GEMINI_API_KEY=...
    GEMINI_MODEL=gemini-2.5-flash      # optional; provider default if omitted

All four providers expose an OpenAI-compatible chat-completions API, so one
ChatOpenAI client with a swapped base_url/api_key/model serves them all:

  openai     https://api.openai.com/v1                       (native SDK default)
  gemini     https://generativelanguage.googleapis.com/v1beta/openai/
  openrouter https://openrouter.ai/api/v1
  nvidia     https://integrate.api.nvidia.com/v1
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from data_pipeline import ROOT

PROVIDERS: dict[str, dict] = {
    'openai': {
        'env_key': 'OPENAI_API_KEY',
        'base_url': None,  # langchain-openai default
        'default_model': 'gpt-4o-mini',
    },
    'gemini': {
        'env_key': 'GEMINI_API_KEY',
        'base_url': 'https://generativelanguage.googleapis.com/v1beta/openai/',
        'default_model': 'gemini-2.5-flash',
    },
    'openrouter': {
        'env_key': 'OPENROUTER_API_KEY',
        'base_url': 'https://openrouter.ai/api/v1',
        'default_model': 'openai/gpt-4o-mini',
    },
    'nvidia': {
        'env_key': 'NVIDIA_API_KEY',
        'base_url': 'https://integrate.api.nvidia.com/v1',
        'default_model': 'meta/llama-3.3-70b-instruct',
    },
}


@dataclass
class LLMConfig:
    provider: str
    api_key: str
    base_url: str | None
    model: str


def resolve_llm_config(env: dict | None = None) -> LLMConfig | None:
    """Pick a provider from .env and return its connection config.

    Priority:
      1. Explicit PROVIDER=<name> in .env (error if its key is missing).
      2. Otherwise the first provider (in PROVIDERS order) whose API-key
         environment variable is set.
    Model name: <PROVIDER>_MODEL if set, else the provider's default.
    Returns None when no supported API key is configured at all.
    """
    load_dotenv(ROOT / '.env')
    env = dict(os.environ) if env is None else env

    provider = env.get('PROVIDER', '').strip().lower() or None
    if provider is not None and provider not in PROVIDERS:
        raise ValueError(
            f'Unknown PROVIDER {provider!r}. Choose one of: {", ".join(PROVIDERS)}'
        )

    if provider is None:  # auto-detect: first provider with a key present
        provider = next(
            (name for name, cfg in PROVIDERS.items() if env.get(cfg['env_key'])),
            None,
        )
        if provider is None:
            return None

    cfg = PROVIDERS[provider]
    api_key = env.get(cfg['env_key'], '').strip()
    if not api_key:
        raise ValueError(
            f'PROVIDER={provider} is set but {cfg["env_key"]} is missing from .env'
        )

    model = env.get(f'{provider.upper()}_MODEL', '').strip() or cfg['default_model']
    return LLMConfig(provider=provider, api_key=api_key,
                     base_url=cfg['base_url'], model=model)


def build_llm(config: LLMConfig):
    """Build a LangChain chat model for the resolved provider config."""
    from langchain_openai import ChatOpenAI

    kwargs = {'model': config.model, 'temperature': 0, 'api_key': config.api_key}
    if config.base_url:
        kwargs['base_url'] = config.base_url
    return ChatOpenAI(**kwargs)
