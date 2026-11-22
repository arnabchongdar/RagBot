"""Local LLM generator via Ollama (report §2.2).

The report specifies a local Mistral-7B-Instruct model. We run it through
Ollama (already installed with ``mistral`` pulled) instead of llama-cpp-python:
same model family, fully local/offline, no compiler or GGUF download, and
independent of the Python version. Generation options (temperature, top_p,
top_k, max_tokens) map directly onto the report's parameters.
"""

from __future__ import annotations

from dataclasses import dataclass


class LLMUnavailable(RuntimeError):
    """Raised when the Ollama server or model cannot be reached."""


@dataclass
class GenParams:
    temperature: float = 0.2
    top_p: float = 0.95
    top_k: int = 50
    max_tokens: int = 256
    num_ctx: int = 4096


class OllamaLLM:
    def __init__(
        self,
        model: str = "mistral",
        host: str = "http://localhost:11434",
        defaults: GenParams | None = None,
    ):
        import ollama

        self.model = model
        self.host = host
        self.defaults = defaults or GenParams()
        self._client = ollama.Client(host=host)

    # -- introspection ------------------------------------------------------
    def available_models(self) -> list[str]:
        try:
            resp = self._client.list()
        except Exception as e:  # connection error
            raise LLMUnavailable(
                f"Could not reach Ollama at {self.host}. Is it running? "
                f"Start it with `ollama serve`. Original error: {e}"
            ) from e
        models = getattr(resp, "models", None) or resp.get("models", [])
        names = []
        for m in models:
            name = getattr(m, "model", None) or (m.get("model") if isinstance(m, dict) else None)
            if name:
                names.append(name)
        return names

    def ensure_ready(self) -> None:
        names = self.available_models()
        base = self.model.split(":")[0]
        if not any(n == self.model or n.split(":")[0] == base for n in names):
            raise LLMUnavailable(
                f"Model '{self.model}' not found in Ollama. Available: {names or 'none'}. "
                f"Pull it with `ollama pull {self.model}`."
            )

    # -- generation ---------------------------------------------------------
    def _options(self, temperature, top_p, top_k, max_tokens) -> dict:
        d = self.defaults
        return {
            "temperature": d.temperature if temperature is None else temperature,
            "top_p": d.top_p if top_p is None else top_p,
            "top_k": d.top_k if top_k is None else top_k,
            "num_predict": d.max_tokens if max_tokens is None else max_tokens,
            "num_ctx": d.num_ctx,
        }

    def generate(
        self,
        user: str,
        system: str = "",
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        try:
            resp = self._client.chat(
                model=self.model,
                messages=messages,
                options=self._options(temperature, top_p, top_k, max_tokens),
            )
        except Exception as e:
            raise LLMUnavailable(
                f"Ollama generation failed ({self.host}, model={self.model}): {e}"
            ) from e
        # ollama returns an object or dict depending on version
        msg = getattr(resp, "message", None) or resp.get("message", {})
        content = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else "")
        return (content or "").strip()
