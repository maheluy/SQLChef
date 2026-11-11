"""Cliente do LLM local (Ollama).

Isolado aqui para que o grafo não dependa do provedor: trocar de modelo é só mudar
OLLAMA_MODEL, e os experimentos comparam modelos locais nas mesmas condições.
"""

from dataclasses import dataclass

from agent.config import Settings


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


class OllamaLLM:
    def __init__(self, model: str, host: str, num_ctx: int):
        import ollama

        self.client = ollama.Client(host=host)
        self.model = model
        self.num_ctx = num_ctx

    def complete(self, system: str, messages: list[dict]) -> LLMResponse:
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "system", "content": system}, *messages],
            # num_ctx: o padrão do Ollama (2048-4096) corta o schema + exemplos
            options={"temperature": 0, "num_ctx": self.num_ctx},
        )
        return LLMResponse(
            text=response.message.content or "",
            input_tokens=response.prompt_eval_count or 0,
            output_tokens=response.eval_count or 0,
        )


def get_llm(settings: Settings) -> OllamaLLM:
    return OllamaLLM(settings.model, settings.host, settings.num_ctx)
