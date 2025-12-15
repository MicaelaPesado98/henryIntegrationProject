import os
import json
import requests
from typing import Optional, Mapping, Any

from langchain.llms.base import LLM
from pydantic import BaseModel, Extra


class GroqConfig(BaseModel):
    api_key: Optional[str]
    api_url: Optional[str]
    model: str = "groq-1"

    class Config:
        extra = Extra.ignore


class GroqLLM(LLM):
    """A lightweight Groq LLM wrapper for LangChain.

    NOTE: Groq API shapes may change; set `GROQ_API_URL` and `GROQ_API_KEY` env vars.
    Set `GROQ_API_URL` to the full base endpoint (e.g. https://api.groq.com/v1) if required.

    The wrapper sends POST requests with JSON: {"model": model, "prompt": prompt, "max_tokens": max_tokens}
    and expects the provider to return a JSON with a text/completion field. If the response shape differs,
    adjust `GroqLLM._call()` accordingly.
    """

    model: str = "groq-1"
    max_tokens: int = 512

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model}

    @property
    def _llm_type(self) -> str:
        return "groq"

    def _call(self, prompt: str, stop: Optional[list[str]] = None) -> str:
        api_key = os.environ.get("GROQ_API_KEY")
        api_url = os.environ.get("GROQ_API_URL", "https://api.groq.com/v1")

        if not api_key:
            raise ValueError("GROQ_API_KEY no está seteada en el entorno. Setea la variable y vuelve a intentar.")

        payload = {"model": self.model, "prompt": prompt, "max_tokens": self.max_tokens}
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        # The exact endpoint path may vary; user can set GROQ_API_URL including path if necessary.
        resp = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=30)
        try:
            resp.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Error al llamar a Groq API: {e} - response: {resp.text}")

        data = resp.json()

        # Heurística para extraer texto de la respuesta.
        # Ajusta según el formato real de la API de Groq (p. ej. data.choices[0].text)
        if isinstance(data, dict):
            # common shapes
            if "text" in data:
                return data["text"]
            if "completion" in data:
                return data["completion"]
            if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                first = data["choices"][0]
                if isinstance(first, dict) and "text" in first:
                    return first["text"]
                if isinstance(first, dict) and "message" in first:
                    # message: {role:..., content:...}
                    msg = first["message"]
                    if isinstance(msg, dict) and "content" in msg:
                        return msg["content"]

        # fallback: return the raw JSON string
        return json.dumps(data)