"""Configuração do agente, lida do ambiente (com suporte a arquivo .env na raiz do repositório).

Variáveis já definidas no ambiente têm precedência sobre o .env, então dá para trocar
provedor ou versão por linha de comando sem editar arquivo:

    LLM_PROVIDER=ollama python eval/run_eval.py --agent agent.sql_agent:generate_sql_v1
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)

VERSIONS = ("v1", "v2", "v3")


@dataclass(frozen=True)
class Settings:
    provider: str
    anthropic_model: str
    anthropic_effort: str
    ollama_model: str
    ollama_host: str
    version: str
    max_attempts: int

    @property
    def model(self) -> str:
        return self.anthropic_model if self.provider == "anthropic" else self.ollama_model


def get_settings() -> Settings:
    settings = Settings(
        provider=os.environ.get("LLM_PROVIDER", "anthropic").strip().lower(),
        anthropic_model=os.environ.get("ANTHROPIC_MODEL", "claude-opus-5-5").strip(),
        anthropic_effort=os.environ.get("ANTHROPIC_EFFORT", "low").strip().lower(),
        ollama_model=os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b").strip(),
        ollama_host=os.environ.get("OLLAMA_HOST", "http://localhost:11434").strip(),
        version=os.environ.get("AGENT_VERSION", "v3").strip().lower(),
        max_attempts=int(os.environ.get("MAX_ATTEMPTS", "3")),
    )
    if settings.provider not in ("anthropic", "ollama"):
        raise ValueError(f"LLM_PROVIDER inválido: {settings.provider!r} (use anthropic ou ollama)")
    if settings.version not in VERSIONS:
        raise ValueError(f"AGENT_VERSION inválida: {settings.version!r} (use {', '.join(VERSIONS)})")
    return settings
