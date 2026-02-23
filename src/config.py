"""
Central configuration for API providers and experiment parameters.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProviderConfig:
    """Configuration for an LLM API provider."""
    name: str
    api_key: str
    base_url: str
    model: str
    max_tokens: int = 2048
    temperature: float = 0.0  # Deterministic for reproducibility
    cost_per_1m_input: float = 0.0   # USD per 1M input tokens
    cost_per_1m_output: float = 0.0  # USD per 1M output tokens


# ── Provider Presets ─────────────────────────────────────

def get_openai_config(api_key: Optional[str] = None) -> ProviderConfig:
    return ProviderConfig(
        name="openai",
        api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
        cost_per_1m_input=0.15,
        cost_per_1m_output=0.60,
    )


def get_siliconflow_config(api_key: Optional[str] = None) -> ProviderConfig:
    return ProviderConfig(
        name="siliconflow",
        api_key=api_key or os.getenv("SILICONFLOW_API_KEY", ""),
        base_url="https://api.siliconflow.cn/v1",
        model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        cost_per_1m_input=0.14,
        cost_per_1m_output=0.28,
    )


def get_gemini_config(api_key: Optional[str] = None) -> ProviderConfig:
    return ProviderConfig(
        name="gemini",
        api_key=api_key or os.getenv("GEMINI_API_KEY", ""),
        base_url="",  # Uses google-generativeai library, not OpenAI compat
        model="gemini-1.5-flash",
        cost_per_1m_input=0.0,  # Free tier
        cost_per_1m_output=0.0,
    )


@dataclass
class ExperimentConfig:
    """Configuration for experiment runs."""
    math_sample_size: int = 500       # Total MATH problems
    math_per_level: int = 100         # Problems per difficulty level
    gsm8k_sample_size: int = 200      # Total GSM8K problems
    random_seed: int = 42
    output_dir: str = "results"
    save_intermediate: bool = True    # Save raw results after each phase

    # Router thresholds (initial, will be optimized)
    tau_low: float = 0.30
    tau_high: float = 0.65
