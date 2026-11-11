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

## Demo

```bash
streamlit run app.py
```

Interface em Streamlit sobre `answer()`: a pergunta vira SQL, a SQL roda no banco somente leitura e o modelo responde em português. Para cada resposta, a demo mostra a SQL usada, a tabela de resultado, um gráfico quando o resultado é rótulo + número, e as tentativas, a latência e os tokens. Na barra lateral dá para trocar a versão do agente (v1 a v4) e o modelo instalado no Ollama, e há perguntas de exemplo do conjunto de avaliação, incluindo uma que o 7B não resolve.

<!-- GIF: docs/demo.gif -->

## Arquitetura

Grafo LangGraph em `agent/sql_agent.py`: **gerar** (LLM recebe schema e pergunta) → **executar** (`agent/db.py`: conexão SQLite somente leitura, só um `SELECT`, timeout) → se der erro ou resultado vazio, **corrigir** (o erro volta ao LLM) e gerar de novo, até `MAX_ATTEMPTS`.

Quatro versões para comparação:

| Versão | O que muda |
|---|---|
| `v1` | uma chamada, sem exemplos |
| `v2` | + exemplos few-shot (diferentes das perguntas de teste) |
| `v3` | + autocorreção com retry |
| `v4` | + dica de schema na correção: se a coluna não existe, diz em quais tabelas ela está e mostra as colunas delas |

`agent/llm.py` isola o cliente do Ollama e registra os tokens de cada chamada; trocar de modelo é só mudar `OLLAMA_MODEL`.

## Avaliação

**Métrica: execution accuracy.** A SQL gerada e a SQL de referência são executadas no banco, e a resposta conta como certa se as duas devolvem o mesmo resultado. Não há LLM avaliando LLM.

Regras de comparação (`eval/run_eval.py`):
- nomes de colunas são ignorados;
- floats são arredondados a 2 casas;
- a ordem das linhas só importa nas perguntas marcadas `ordered` (rankings); nas demais, as linhas são comparadas como conjunto;
- a ordem das colunas só importa nas perguntas marcadas `strict_columns`;
- colunas a mais ou a menos contam como erro, porque a pergunta diz o que retornar.

**Conjunto de teste:** 60 perguntas em português (16 fáceis, 28 médias, 16 difíceis) em `eval/questions.json`, cada uma com a SQL de referência, em dois lotes:
- **Lote 1 (q01–q22):** usado para desenvolver as versões v2–v4.
- **Lote 2 (q23–q60):** escrito só a partir do schema, sem ver respostas do modelo. Serve para checar se as melhorias valem para perguntas que não influenciaram o desenvolvimento.

Os exemplos few-shot não repetem nenhuma pergunta de teste.

```bash
python eval/run_eval.py --validate-gold                     # as 60 SQLs de referência rodam e não vêm vazias
python eval/run_eval.py --agent agent.sql_agent:generate_sql_v3 --label qwen7b_v3_60
OLLAMA_MODEL=qwen2.5-coder:14b python eval/run_eval.py --agent agent.sql_agent:generate_sql_v3 --label qwen14b_v3_60
python eval/run_eval.py --agent agent.sql_agent:generate_sql_v3 --lote 2 --label so_lote2
```

Resultados em `eval/results/<label>.csv` (por pergunta: acerto, SQL gerada, tentativas, tokens, latência) e `<label>_summary.json` (acurácia por dificuldade e por lote, e a configuração da rodada).

## Resultados

Ollama local com temperatura 0 (rodar de novo dá as mesmas SQLs), GPU AMD Radeon (Navi 33, ~8 GB de VRAM).

| Modelo | Versão | Geral | Fácil | Média | Difícil | Latência média | Tokens/pergunta |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-coder:7b | v1 | 44/60 (73,3%) | 16/16 | 23/28 | 5/16 | 1,2 s | 1.486 |
| qwen2.5-coder:7b | v2 | 46/60 (76,7%) | 16/16 | 25/28 | 5/16 | 1,1 s | 1.980 |
| qwen2.5-coder:7b | v3 | 48/60 (80,0%) | 16/16 | 27/28 | 5/16 | 1,6 s | 2.452 |
| qwen2.5-coder:7b | v4 | 48/60 (80,0%) | 16/16 | 27/28 | 5/16 | 1,6 s | 2.381 |
| qwen2.5-coder:14b | v1 | 48/60 (80,0%) | 16/16 | 27/28 | 5/16 | 6,6 s | 1.287 |
| qwen2.5-coder:14b | v2 | 52/60 (86,7%) | 16/16 | 27/28 | 9/16 | 6,7 s | 1.980 |
| qwen2.5-coder:14b | v3 | 52/60 (86,7%) | 16/16 | 27/28 | 9/16 | 7,2 s | 2.130 |
| qwen2.5-coder:14b | v4 | 52/60 (86,7%) | 16/16 | 27/28 | 9/16 | 6,4 s | 2.100 |

O que os números mostram:
- **Exemplos e autocorreção ajudam, inclusive em perguntas inéditas.** No 7B, de v1 para v3 são 4 acertos a mais no total e 3 no lote 2. O ganho vem todo das perguntas médias (de 23 para 27 de 28); as fáceis já estavam no máximo.
- **Nenhuma técnica de prompt melhorou as perguntas difíceis.** No 7B ficaram em 5/16 da v1 à v4. A correção só reage quando a SQL dá erro ou volta vazia, e quase todas as falhas das difíceis rodam sem erro e devolvem um número errado (ver abaixo).
- **A dica de schema (v4) não mudou nenhum acerto.** Ela muda o comportamento do modelo (por exemplo, ele passa a adicionar o JOIN que faltava), mas o 7B não completa a correção.
- **O modelo maior foi o único que melhorou as difíceis** (de 5 para 9 de 16). O 14B acerta tudo que o 7B acerta e mais quatro perguntas difíceis. Também precisa bem menos de correção (2 perguntas com nova tentativa, contra 9), por isso usa menos tokens. O custo é a latência, cerca de 4,6 vezes maior.

## Análise de erros

Falhas restantes na v3, por tipo:

| Tipo de erro | Exemplo | 7B | 14B |
|---|---|---|---|
| Contar grupos sem subconsulta: `GROUP BY ... HAVING` com `COUNT` devolve uma linha por grupo em vez do total | "Quantos pedidos têm itens de mais de um vendedor?" (q22) | 2 | 2 |
| Média ou soma no nível errado: média por linha de pagamento em vez do total por pedido, ou JOIN que repete valores | "Qual estado tem o maior ticket médio?" (q57) | 2 | 2 |
| Confusão de schema: usa em `orders` uma coluna que está em `customers`, ou troca `customer_unique_id` (cliente real) por `customer_id` (muda a cada pedido) | "Quantos clientes únicos fizeram mais de um pedido?" (q19) | 2 | 2 |
| `order_item_id` usado como quantidade de itens (a coluna é a posição do item no pedido) | "Qual a quantidade média de itens por pedido?" (q43) | 2 | 1 |
| Lógica da pergunta lida errado: "tanto em 2017 quanto em 2018" vira `IN ('2017', '2018')`; condição ignorada; filtro no lugar da contagem; formato de resposta errado | q22, q50, q51, q52 | 4 | 0 |
| Divisão inteira: `COUNT(*) / COUNT(...)` no SQLite descarta as casas decimais | q43 (14B) | 0 | 1 |
| **Total** | | **12** | **8** |

Exemplo de erro ao contar grupos (q22, gerada pelo 14B):

```sql
-- gerada: devolve uma linha "1" para cada pedido que atende à condição
SELECT COUNT(DISTINCT order_id) FROM order_items
GROUP BY order_id HAVING COUNT(DISTINCT seller_id) > 1

-- correta: conta os grupos
SELECT COUNT(*) FROM (
  SELECT order_id FROM order_items
  GROUP BY order_id HAVING COUNT(DISTINCT seller_id) > 1
)
```

O que fica de lição:
1. **A maioria das falhas restantes são erros silenciosos:** a SQL roda e devolve um resultado plausível. Uma autocorreção baseada em erro de execução não tem como pegá-los. O próximo passo natural seria checar o resultado, e não só a execução: por exemplo, avisar quando "quantos?" devolve várias linhas, ou o modelo revisar a SQL antes de responder.
2. **Agregação é o ponto fraco nos dois tamanhos de modelo.** O 14B corrige os erros de leitura da pergunta, mas os erros de contar grupos e de nível de agregação continuam.
3. **A confusão de schema resiste à correção:** mesmo com a mensagem de erro e a dica de schema, o modelo volta a usar a coluna na tabela errada ou troca o identificador do cliente.

## Limitações

- **60 perguntas é pouco.** Cada pergunta vale 1,7 ponto percentual, então diferenças de uma ou duas perguntas entre versões não são conclusivas.
- **Uma rodada por configuração.** Com temperatura 0 o resultado é determinístico, mas não foi medido como ele varia com outras temperaturas.
- **O lote 2 também já foi inspecionado** para esta análise de erros. Uma técnica nova deveria ser medida em perguntas novas.
- **O 14B só foi rodado na v3.** Falta medir v1, v2 e v4 para separar quanto do ganho vem do tamanho do modelo e quanto vem das técnicas.
