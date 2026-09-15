"""
Provider adapter layer.

ALL vendor-specific knowledge lives in this module: endpoint shape, auth
header names, request body format, and where the generated text sits in
the response JSON. Everything above this layer (llm_classifier.py) deals
only with "send a prompt, get back a string".

Swapping provider therefore means adding one small class here and changing
LLM_PROVIDER in .env — no changes to the classifier, the prompt builder,
the routes, or the tests.

This module deliberately knows nothing about crop diseases, categories,
or JSON classification schemas.
"""

import json
import requests


class LLMProviderError(Exception):
    """
    Raised when the provider call fails at the transport/API level.

    This covers timeouts, connection failures, auth rejections, rate
    limits and malformed provider envelopes — i.e. everything that goes
    wrong BEFORE we have model text to parse.
    """

    def __init__(self, message: str, status_code: int = None, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class BaseLLMProvider:
    """
    Interface every provider adapter implements.

    Subclasses must implement generate() and return the model's raw text
    output as a plain string.
    """

    name = "base"

    def __init__(self, api_key: str, model: str, base_url: str, timeout: int = 30):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        raise NotImplementedError

    # -- shared helpers -----------------------------------------------------

    def _post(self, url: str, headers: dict, payload: dict) -> dict:
        """
        POST helper with uniform error translation.

        Note: exception messages never include headers or the API key —
        only status codes and response snippets, so a stack trace or log
        line can never leak the credential.
        """
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=self.timeout
            )
        except requests.exceptions.Timeout as exc:
            raise LLMProviderError(
                f"LLM request timed out after {self.timeout}s.", retryable=True
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise LLMProviderError(
                "Could not connect to the LLM API. Check network settings "
                "and LLM_API_BASE_URL.", retryable=True
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise LLMProviderError(f"LLM request failed: {exc}") from exc

        if response.status_code != 200:
            # 429 and 5xx are transient and worth retrying; 4xx are not.
            retryable = response.status_code == 429 or response.status_code >= 500
            snippet = (response.text or "")[:300]
            raise LLMProviderError(
                f"LLM API returned HTTP {response.status_code}: {snippet}",
                status_code=response.status_code,
                retryable=retryable,
            )

        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise LLMProviderError(
                "LLM API returned a non-JSON response envelope."
            ) from exc


class AnthropicProvider(BaseLLMProvider):
    """Adapter for the Anthropic Messages API."""

    name = "anthropic"
    DEFAULT_BASE_URL = "https://api.anthropic.com/v1/messages"
    API_VERSION = "2023-06-01"

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        url = self.base_url or self.DEFAULT_BASE_URL
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.API_VERSION,
            "content-type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        data = self._post(url, headers, payload)

        # Response shape: {"content": [{"type": "text", "text": "..."}], ...}
        blocks = data.get("content")
        if not isinstance(blocks, list) or not blocks:
            raise LLMProviderError("Anthropic response contained no content blocks.")

        text = "".join(
            block.get("text", "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        )
        if not text.strip():
            raise LLMProviderError("Anthropic response contained no text output.")
        return text


class OpenAICompatibleProvider(BaseLLMProvider):
    """
    Adapter for OpenAI-style /chat/completions APIs.

    Also works with the many providers that mimic this schema (OpenRouter,
    Groq, Together, Ollama, vLLM), which is why it is not named "openai".
    """

    name = "openai"
    DEFAULT_BASE_URL = "https://api.openai.com/v1/chat/completions"

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        url = self.base_url or self.DEFAULT_BASE_URL
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        data = self._post(url, headers, payload)

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMProviderError("OpenAI-compatible response contained no choices.")

        text = (choices[0].get("message") or {}).get("content", "")
        if not text or not text.strip():
            raise LLMProviderError("OpenAI-compatible response contained no text output.")
        return text


class EchoProvider(BaseLLMProvider):
    """
    Offline test/demo provider. Makes NO network call.

    Returns a canned, valid JSON classification so the full pipeline can
    be exercised (routes, parsing, validation, frontend) without an API
    key or internet access. It does not classify anything intelligently
    and must never be used to produce evaluation numbers.
    """

    name = "echo"

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        return json.dumps({
            "category": "Unable to Classify",
            "confidence": 0.0,
            "reason": "Echo provider is a stub and does not perform real classification.",
        })


# Registry of available providers. Add a new adapter class above and one
# entry here to support another vendor.
PROVIDER_REGISTRY = {
    AnthropicProvider.name: AnthropicProvider,
    OpenAICompatibleProvider.name: OpenAICompatibleProvider,
    EchoProvider.name: EchoProvider,
}


def get_provider(provider_name: str, api_key: str, model: str,
                 base_url: str = "", timeout: int = 30) -> BaseLLMProvider:
    """
    Build the provider adapter for the configured provider name.

    Raises:
        LLMProviderError: If the provider name is not registered.
    """
    key = (provider_name or "").strip().lower()
    provider_class = PROVIDER_REGISTRY.get(key)
    if provider_class is None:
        raise LLMProviderError(
            f"Unknown LLM provider '{provider_name}'. "
            f"Supported providers: {sorted(PROVIDER_REGISTRY)}"
        )
    return provider_class(api_key=api_key, model=model, base_url=base_url, timeout=timeout)
