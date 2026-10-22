"""Unified client abstraction supporting cloud APIs (OpenAI/Groq) and local HuggingFace models."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import torch

from src.config.model_config import LLMConfig, model_settings

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract contract for LLM invocation."""

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Generate response text given chat messages."""
        pass


class OpenAICompatibleClient(BaseLLMClient):
    """Client for OpenAI, Groq, Ollama, and other OpenAI-compatible endpoints."""

    def __init__(self, config: LLMConfig):
        from openai import OpenAI

        self.config = config
        self.client = OpenAI(
            api_key=self.config.api_key or "not-needed",
            base_url=self.config.base_url,
            timeout=self.config.request_timeout,
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        tokens = max_tokens or self.config.max_tokens
        temp = temperature if temperature is not None else self.config.temperature

        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,  # type: ignore[arg-type]
                max_tokens=tokens,
                temperature=temp,
            )
            content = response.choices[0].message.content
            return content.strip() if content else ""
        except Exception as e:
            logger.error(f"Error calling OpenAI-compatible API: {e}")
            raise RuntimeError(f"LLM API generation failed: {e}") from e


class LocalTransformersClient(BaseLLMClient):
    """Client for local Hugging Face CausalLM generation without remote API calls."""

    def __init__(self, config: LLMConfig):
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

        self.config = config
        logger.info(f"Loading local HF model: {self.config.model_name}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" else -1,
        )

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        tokens = max_tokens or self.config.max_tokens
        temp = temperature if temperature is not None else self.config.temperature

        # Apply chat template if supported by tokenizer
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
            prompt_text += "\nASSISTANT: "

        do_sample = temp > 0.0
        outputs = self.generator(
            prompt_text,
            max_new_tokens=tokens,
            temperature=temp if do_sample else None,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_raw = outputs[0]["generated_text"]
        new_text = generated_raw[len(prompt_text):].strip()
        return new_text


class LLMClientFactory:
    """Dispatches the appropriate LLM client instance based on configuration."""

    @classmethod
    def create_client(cls, config: Optional[LLMConfig] = None) -> BaseLLMClient:
        cfg = config or model_settings.get_llm_config()

        if cfg.api_based:
            logger.info(f"Instantiating API-based LLM client: {cfg.provider} ({cfg.model_name})")
            return OpenAICompatibleClient(cfg)
        else:
            logger.info(f"Instantiating local Transformers LLM client: {cfg.model_name}")
            return LocalTransformersClient(cfg)