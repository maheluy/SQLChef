"""Camada de provedor: a mesma interface para Claude (API) e Ollama (local).

Ambos devolvem LLMResponse com texto, tokens e custo, para que a avaliação compare
os provedores nas mesmas condições.
"""

from dataclasses import dataclass

from agent.config import Settings

# Preço em US$ por 1M de tokens (entrada, saída). Fonte: tabela de preços da Anthropic, nov/2026.
ANTHROPIC_PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}
CACHE_READ_MULT = 0.1    # leitura de cache custa 10% da entrada
CACHE_WRITE_MULT = 1.25  # escrita de cache (TTL 5 min) custa 125% da entrada


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0


class LLMError(RuntimeError):
    """O provedor não devolveu uma resposta utilizável."""


class AnthropicLLM:
    def __init__(self, model: str, effort: str):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort

    def complete(self, system: str, messages: list[dict]) -> LLMResponse:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            # schema + instruções são iguais em todas as perguntas: cache reduz custo e latência
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            output_config={"effort": self.effort},
        )
        if response.stop_reason == "refusal":
            raise LLMError("o modelo recusou a solicitação")
        text = "".join(b.text for b in response.content if b.type == "text")
        if response.stop_reason == "max_tokens" and not text:
            raise LLMError("resposta cortada por max_tokens")

        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        input_tokens = usage.input_tokens + cache_read + cache_write
        return LLMResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=self._cost(usage.input_tokens, cache_read, cache_write, usage.output_tokens),
        )

    def _cost(self, uncached, cache_read, cache_write, output):
        if self.model not in ANTHROPIC_PRICES:
            return 0.0  # modelo fora da tabela: sem estimativa (atualize ANTHROPIC_PRICES)
        price_in, price_out = ANTHROPIC_PRICES[self.model]
        billed_in = uncached + cache_read * CACHE_READ_MULT + cache_write * CACHE_WRITE_MULT
        return (billed_in * price_in + output * price_out) / 1_000_000


class OllamaLLM:
    def __init__(self, model: str, host: str):
        import ollama

        self.client = ollama.Client(host=host)
        self.model = model

    def complete(self, system: str, messages: list[dict]) -> LLMResponse:
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "system", "content": system}, *messages],
            # num_ctx: o padrão do Ollama (2048-4096) corta o schema + exemplos
            options={"temperature": 0, "num_ctx": 8192},
        )
        return LLMResponse(
            text=response.message.content or "",
            input_tokens=response.prompt_eval_count or 0,
            output_tokens=response.eval_count or 0,
            cost_usd=0.0,  # roda local; o custo comparável é a latência
        )


def get_llm(settings: Settings):
    if settings.provider == "anthropic":
        return AnthropicLLM(settings.anthropic_model, settings.anthropic_effort)
    return OllamaLLM(settings.ollama_model, settings.ollama_host)
