"""
Chain-of-Thought Neural Reasoning Engine
=========================================

Interfaces with LLM APIs (OpenAI, SiliconFlow/DeepSeek, Gemini)
to perform step-by-step mathematical reasoning.

Tracks token consumption and extracts structured answers.
"""

import re
import time
import logging
from typing import Optional, Tuple
from dataclasses import dataclass
from src.config import ProviderConfig

logger = logging.getLogger(__name__)


@dataclass
class CoTResult:
    """Result from the Chain-of-Thought engine."""
    answer: str
    full_response: str
    tokens_input: int
    tokens_output: int
    tokens_total: int
    latency_seconds: float
    provider: str
    model: str
    success: bool
    error_message: str = ""


# ── Prompt Templates ─────────────────────────────────────

MATH_COT_PROMPT = """Solve the following mathematical problem step by step.
Show your complete reasoning process.
At the end, put your final answer inside \\boxed{{}}.

Problem: {problem}

Solution:"""

GSM8K_COT_PROMPT = """Solve the following problem step by step.
Show each calculation clearly.
At the end, write your final numerical answer after "#### ".

Problem: {problem}

Solution:"""

GENERAL_COT_PROMPT = """Solve the following mathematical problem step by step.
Be precise and show all work.
Put your final answer inside \\boxed{{}}.

Problem: {problem}

Solution:"""


class CoTEngine:
    """
    Chain-of-Thought reasoning engine with multi-provider support.

    Supports:
        - OpenAI API compatible (GPT-4o-mini, DeepSeek via SiliconFlow)
        - Google Gemini (free tier)
    """

    def __init__(self, provider_config: ProviderConfig):
        self.config = provider_config
        self._client = None
        self._setup_client()

    def _setup_client(self):
        """Initialize the API client based on provider."""
        if self.config.name == "gemini":
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.config.api_key)
                self._client = genai.GenerativeModel(self.config.model)
                logger.info(f"Gemini client initialized: {self.config.model}")
            except ImportError:
                raise ImportError(
                    "Install google-generativeai: pip install google-generativeai"
                )
        else:
            # OpenAI-compatible API (works for OpenAI AND SiliconFlow)
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.base_url,
                )
                logger.info(
                    f"OpenAI-compatible client initialized: "
                    f"{self.config.name}/{self.config.model}"
                )
            except ImportError:
                raise ImportError("Install openai: pip install openai")

    def solve(
        self,
        problem: str,
        prompt_type: str = "general",
        timeout: int = 60,
    ) -> CoTResult:
        """
        Solve a mathematical problem using Chain-of-Thought.

        Args:
            problem: The mathematical problem text.
            prompt_type: One of 'math', 'gsm8k', 'general'.
            timeout: Max seconds to wait for response.

        Returns:
            CoTResult with answer, tokens, timing, etc.
        """
        # ── Select prompt template ───────────────────
        template = {
            'math': MATH_COT_PROMPT,
            'gsm8k': GSM8K_COT_PROMPT,
            'general': GENERAL_COT_PROMPT,
        }.get(prompt_type, GENERAL_COT_PROMPT)

        formatted_prompt = template.format(problem=problem)

        # ── Call API ─────────────────────────────────
        start_time = time.time()

        try:
            if self.config.name == "gemini":
                result = self._call_gemini(formatted_prompt, timeout)
            else:
                result = self._call_openai_compatible(formatted_prompt, timeout)

            result.latency_seconds = round(time.time() - start_time, 3)

            # ── Extract final answer ─────────────────
            result.answer = self._extract_answer(
                result.full_response, prompt_type
            )

            return result

        except Exception as e:
            elapsed = round(time.time() - start_time, 3)
            logger.error(f"CoT engine error: {e}")
            return CoTResult(
                answer="",
                full_response="",
                tokens_input=0,
                tokens_output=0,
                tokens_total=0,
                latency_seconds=elapsed,
                provider=self.config.name,
                model=self.config.model,
                success=False,
                error_message=str(e),
            )

    def _call_openai_compatible(
        self, prompt: str, timeout: int
    ) -> CoTResult:
        """Call OpenAI-compatible API (OpenAI, SiliconFlow, etc.)."""
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert mathematician. "
                        "Solve problems step by step with precision."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            timeout=timeout,
        )

        content = response.choices[0].message.content or ""
        usage = response.usage

        return CoTResult(
            answer="",  # Will be extracted later
            full_response=content,
            tokens_input=usage.prompt_tokens if usage else 0,
            tokens_output=usage.completion_tokens if usage else 0,
            tokens_total=usage.total_tokens if usage else 0,
            latency_seconds=0,  # Will be set by caller
            provider=self.config.name,
            model=self.config.model,
            success=True,
        )

    def _call_gemini(self, prompt: str, timeout: int) -> CoTResult:
        """Call Google Gemini API."""
        response = self._client.generate_content(
            prompt,
            generation_config={
                "max_output_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
            },
        )

        content = response.text or ""

        # Gemini doesn't always return token counts, estimate
        tokens_in = len(prompt.split()) * 1.3  # rough estimate
        tokens_out = len(content.split()) * 1.3

        return CoTResult(
            answer="",
            full_response=content,
            tokens_input=int(tokens_in),
            tokens_output=int(tokens_out),
            tokens_total=int(tokens_in + tokens_out),
            latency_seconds=0,
            provider="gemini",
            model=self.config.model,
            success=True,
        )

    def _extract_answer(self, response: str, prompt_type: str) -> str:
        """
        Extract the final answer from LLM response.

        Handles multiple formats:
          - \\boxed{answer}     (MATH format)
          - #### answer         (GSM8K format)
          - Last number         (fallback)
        """
        if not response:
            return ""

        # ── Try \\boxed{...} ────────────────────────
        boxed_patterns = [
            r'\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}',
            r'\\boxed\{(.+?)\}',
        ]
        for pattern in boxed_patterns:
            matches = re.findall(pattern, response, re.DOTALL)
            if matches:
                return matches[-1].strip()

        # ── Try #### answer (GSM8K) ─────────────────
        gsm_match = re.search(r'####\s*(.+?)(?:\n|$)', response)
        if gsm_match:
            return gsm_match.group(1).strip()

        # ── Try "answer is X" / "answer: X" ────────
        answer_patterns = [
            r'[Tt]he\s+(?:final\s+)?answer\s+is[:\s]+(.+?)(?:\.|$)',
            r'[Aa]nswer[:\s]+(.+?)(?:\.|$)',
            r'[Rr]esult[:\s]+(.+?)(?:\.|$)',
            r'=\s*(\d+(?:\.\d+)?)\s*$',
        ]
        for pattern in answer_patterns:
            match = re.search(pattern, response, re.MULTILINE)
            if match:
                return match.group(1).strip()

        # ── Fallback: last number in response ───────
        numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
        if numbers:
            return numbers[-1]

        return ""

    def estimate_cost(self, tokens_input: int, tokens_output: int) -> float:
        """Estimate cost in USD for given token counts."""
        cost = (
            tokens_input * self.config.cost_per_1m_input / 1_000_000
            + tokens_output * self.config.cost_per_1m_output / 1_000_000
        )
        return round(cost, 6)
