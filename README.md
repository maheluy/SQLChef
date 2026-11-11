# SQLChef
Agente text-to-SQL que responde em português a perguntas sobre dados reais de um marketplace brasileiro (dataset público da Olist), gerando, executando e corrigindo consultas SQL em um banco SQLite.

## Como rodar

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

O LLM roda localmente no [Ollama](https://ollama.com/download), sem custo de API:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5-coder:7b
python -m agent.sql_agent "Quantos pedidos foram cancelados?"
```

Configuração no `.env` (ou no ambiente): `OLLAMA_MODEL`, `OLLAMA_HOST`, `OLLAMA_NUM_CTX`, `AGENT_VERSION`, `MAX_ATTEMPTS`.

## Arquitetura

Grafo LangGraph em `agent/sql_agent.py`: **gerar** (LLM recebe schema e pergunta) → **executar** (`agent/db.py`: conexão SQLite somente leitura, só um `SELECT`, timeout) → se der erro ou resultado vazio, **corrigir** (o erro volta ao LLM) e gerar de novo, até `MAX_ATTEMPTS`.

Três versões para comparação:

| Versão | O que muda |
|---|---|
| `v1` | uma chamada, sem exemplos |
| `v2` | + exemplos few-shot (diferentes das perguntas de teste) |
| `v3` | + autocorreção com retry |
| `v4` | + dica de schema na correção: se a coluna não existe, diz em quais tabelas ela está e mostra as colunas delas |

`agent/llm.py` isola o cliente do Ollama e registra os tokens de cada chamada; trocar de modelo é só mudar `OLLAMA_MODEL`.

## Avaliação

```bash
python eval/run_eval.py --validate-gold
python eval/run_eval.py --agent agent.sql_agent:generate_sql_v1 --label qwen7b_v1
OLLAMA_MODEL=llama3.1:8b python eval/run_eval.py --agent agent.sql_agent:generate_sql_v3 --label llama8b_v3
```

Resultados em `eval/results/<label>.csv` (por pergunta, com tentativas, tokens e latência) e `<label>_summary.json` (acurácia por dificuldade e a configuração da rodada).
