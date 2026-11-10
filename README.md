# SQLChef
Agente text-to-SQL que responde em português a perguntas sobre dados reais de um marketplace brasileiro (dataset público da Olist), gerando, executando e corrigindo consultas SQL em um banco SQLite.

## Como rodar

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # escolha o provedor e preencha a chave
```

O agente funciona com dois provedores, escolhidos por `LLM_PROVIDER` no `.env` (ou no ambiente):

| Provedor | Configuração |
|---|---|
| `anthropic` (API) | `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` , `ANTHROPIC_EFFORT` |
| `ollama` (local) | instale o [Ollama](https://ollama.com/download), rode `ollama pull qwen2.5-coder:7b`; `OLLAMA_MODEL`, `OLLAMA_HOST` |

```bash
python -m agent.sql_agent "Quantos pedidos foram cancelados?"
LLM_PROVIDER=ollama python -m agent.sql_agent "Quantos pedidos foram cancelados?"
```

## Arquitetura

Grafo LangGraph em `agent/sql_agent.py`: **gerar** (LLM recebe schema e pergunta) → **executar** (`agent/db.py`: conexão SQLite somente leitura, só um `SELECT`, timeout) → se der erro ou resultado vazio, **corrigir** (o erro volta ao LLM) e gerar de novo, até `MAX_ATTEMPTS`.

Três versões para comparação:

| Versão | O que muda |
|---|---|
| `v1` | uma chamada, sem exemplos |
| `v2` | + exemplos few-shot (diferentes das perguntas de teste) |
| `v3` | + autocorreção com retry |

`agent/llm.py` dá a mesma interface aos dois provedores e registra tokens e custo (Claude: tabela de preços com cache do prompt; Ollama: custo zero, compara-se a latência).

## Avaliação

```bash
python eval/run_eval.py --validate-gold
LLM_PROVIDER=anthropic python eval/run_eval.py --agent agent.sql_agent:generate_sql_v1 --label anthropic_v1
LLM_PROVIDER=ollama    python eval/run_eval.py --agent agent.sql_agent:generate_sql_v3 --label ollama_v3
```

Resultados em `eval/results/<label>.csv` (por pergunta, com tentativas, tokens, custo e latência) e `<label>_summary.json` (acurácia por dificuldade e a configuração da rodada).
