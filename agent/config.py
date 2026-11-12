"""Configuração do agente, lida do ambiente (com suporte a arquivo .env na raiz do repositório).

Variáveis já definidas no ambiente têm precedência sobre o .env, então dá para trocar
modelo ou versão por linha de comando sem editar arquivo:

    OLLAMA_MODEL=llama3.1:8b python eval/run_eval.py --agent agent.sql_agent:generate_sql_v1
"""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)

VERSIONS = ("v1", "v2", "v3", "v4", "v5", "v6")


@dataclass(frozen=True)
class Settings:
    model: str
    host: str
    num_ctx: int
    version: str
    max_attempts: int


def get_settings() -> Settings:
    settings = Settings(
        model=os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:7b").strip(),
        host=os.environ.get("OLLAMA_HOST", "http://localhost:11434").strip(),
        num_ctx=int(os.environ.get("OLLAMA_NUM_CTX", "8192")),
        version=os.environ.get("AGENT_VERSION", "v3").strip().lower(),
        max_attempts=int(os.environ.get("MAX_ATTEMPTS", "3")),
    )
    if settings.version not in VERSIONS:
        raise ValueError(f"AGENT_VERSION inválida: {settings.version!r} (use {', '.join(VERSIONS)})")
    return settings
