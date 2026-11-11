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

Interface em Streamlit sobre `answer()`: a pergunta vira SQL, a SQL roda no banco somente leitura e o modelo responde em português. Para cada resposta, a demo mostra a SQL usada, a tabela de resultado, um gráfico quando o resultado é rótulo + número, e as tentativas, a latência e os tokens. Na barra lateral dá para trocar a versão do agente (v1 a v5) e o modelo instalado no Ollama, e há perguntas de exemplo do conjunto de avaliação, incluindo uma que o 7B não resolve.

<!-- GIF: docs/demo.gif -->

## Arquitetura

Grafo LangGraph em `agent/sql_agent.py`: **gerar** (LLM recebe schema e pergunta) → **executar** (`agent/db.py`: conexão SQLite somente leitura, só um `SELECT`, timeout) → se der erro ou resultado vazio (na v5, também resultado com formato suspeito), **corrigir** (o problema volta ao LLM) e gerar de novo, até `MAX_ATTEMPTS`.

Cinco versões para comparação, cada uma acumulando as anteriores:

| Versão | O que muda |
|---|---|
| `v1` | uma chamada, sem exemplos |
| `v2` | + exemplos few-shot (diferentes das perguntas de teste) |
| `v3` | + autocorreção com retry |
| `v4` | + dica de schema na correção: se a coluna não existe, diz em quais tabelas ela está e mostra as colunas delas |
| `v5` | + notas de semântica do schema no prompt (o que identifica cada linha, onde há várias linhas por pedido, que `order_item_id` é posição e não quantidade, divisão inteira) + checagem de formato: se a pergunta pede um único valor ("Quantos...") e voltaram várias linhas, o resultado vai para a correção mesmo sem erro |

`agent/llm.py` isola o cliente do Ollama e registra os tokens de cada chamada; trocar de modelo é só mudar `OLLAMA_MODEL`.

## Avaliação

**Métrica: execution accuracy.** A SQL gerada e a SQL de referência são executadas no banco, e a resposta conta como certa se as duas devolvem o mesmo resultado. Não há LLM avaliando LLM.

Regras de comparação (`eval/run_eval.py`):
- nomes de colunas são ignorados;
- floats são arredondados a 2 casas;
- a ordem das linhas só importa nas perguntas marcadas `ordered` (rankings); nas demais, as linhas são comparadas como conjunto;
- a ordem das colunas só importa nas perguntas marcadas `strict_columns`;
- colunas a mais ou a menos contam como erro, porque a pergunta diz o que retornar.

**Conjunto de teste:** 80 perguntas em português (21 fáceis, 37 médias, 22 difíceis) em `eval/questions.json`, cada uma com a SQL de referência, em três lotes:
- **Lote 1 (q01–q22):** usado para desenvolver as versões v2–v4.
- **Lote 2 (q23–q60):** escrito só a partir do schema, sem ver respostas do modelo. Serve para checar se as melhorias valem para perguntas que não influenciaram o desenvolvimento. Depois foi inspecionado para a análise de erros que originou a v5.
- **Lote 3 (q61–q80):** escrito antes de rodar a v5, para medi-la em perguntas que ela não viu.

Os exemplos few-shot não repetem nenhuma pergunta de teste.

```bash
python eval/run_eval.py --validate-gold                     # as 80 SQLs de referência rodam e não vêm vazias
python eval/run_eval.py --agent agent.sql_agent:generate_sql_v5 --label qwen7b_v5_80
OLLAMA_MODEL=qwen2.5-coder:14b python eval/run_eval.py --agent agent.sql_agent:generate_sql_v5 --label qwen14b_v5_80
python eval/run_eval.py --agent agent.sql_agent:generate_sql_v5 --lote 3 --label so_lote3
```

Resultados em `eval/results/<label>.csv` (por pergunta: acerto, SQL gerada, tentativas, tokens, latência) e `<label>_summary.json` (acurácia por dificuldade e por lote, e a configuração da rodada).

## Resultados

Ollama local com temperatura 0, GPU AMD Radeon (Navi 33, ~8 GB de VRAM). As colunas Fácil, Média e Difícil se referem aos lotes 1+2 (16, 28 e 16 perguntas); o lote 3 tem 5 fáceis, 9 médias e 6 difíceis. Latência e tokens são médias por pergunta.

| Modelo | Versão | Lotes 1+2 (60) | Fácil | Média | Difícil | Lote 3 (20, inédito) | Latência | Tokens |
|---|---|---|---|---|---|---|---|---|
| qwen2.5-coder:7b | v1 | 44/60 (73,3%) | 16/16 | 23/28 | 5/16 | 11/20 | 1,1 s | 1.481 |
| qwen2.5-coder:7b | v2 | 46/60 (76,7%) | 16/16 | 25/28 | 5/16 | 12/20 | 1,0 s | 1.975 |
| qwen2.5-coder:7b | v3 | 49/60 (81,7%) | 16/16 | 27/28 | 6/16 | 12/20 | 1,4 s | 2.439 |
| qwen2.5-coder:7b | v4 | 49/60 (81,7%) | 16/16 | 27/28 | 6/16 | 13/20 | 1,4 s | 2.390 |
| qwen2.5-coder:7b | v5 | 51/60 (85,0%) | 16/16 | 27/28 | 8/16 | 14/20 | 1,3 s | 2.825 |
| qwen2.5-coder:14b | v1 | 48/60 (80,0%) | 16/16 | 27/28 | 5/16 | 12/20 | 6,6 s | 1.483 |
| qwen2.5-coder:14b | v2 | 52/60 (86,7%) | 16/16 | 27/28 | 9/16 | 14/20 | 6,2 s | 1.976 |
| qwen2.5-coder:14b | v3 | 52/60 (86,7%) | 16/16 | 27/28 | 9/16 | 15/20 | 7,4 s | 2.115 |
| qwen2.5-coder:14b | v4 | 52/60 (86,7%) | 16/16 | 27/28 | 9/16 | 15/20 | 7,7 s | 2.092 |
| qwen2.5-coder:14b | **v5** | **55/60 (91,7%)** | 16/16 | 27/28 | **12/16** | **19/20** | 5,8 s | 2.306 |

Todas as linhas vêm de rodadas com as 80 perguntas (`eval/results/qwen{7b,14b}_v*_80`).

O que os números mostram:
- **Exemplos e autocorreção ajudam, inclusive em perguntas inéditas.** No 7B, de v1 para v3 são 5 acertos a mais nos lotes 1+2, quase todos nas perguntas médias (de 23 para 27 de 28); as fáceis já estavam no máximo.
- **No 7B, as técnicas de v2 a v4 quase não mexem nas difíceis** (de 5 para 6 de 16). A correção só reage quando a SQL dá erro ou volta vazia, e quase todas as falhas das difíceis rodam sem erro e devolvem um número errado (ver abaixo).
- **No 14B, o few-shot é que faz a diferença:** de v1 para v2 as difíceis vão de 5 para 9 de 16, e a autocorreção e a dica de schema não somam nada depois disso. O 14B precisa bem menos de correção, por isso usa menos tokens, mas a latência é cerca de 5 vezes maior.
- **A v5 é a primeira técnica que melhora as difíceis nos dois modelos:** de 6 para 8 de 16 no 7B e de 9 para 12 de 16 no 14B (lotes 1+2). Nos lotes 1+2, o 7B com a v5 chega perto do 14B com a v4 (51 contra 52 de 60), com a latência do 7B.
- **No lote 3, que a v5 não viu, o resultado depende do modelo.** O 14B vai de 15 para 19 de 20, sem perder nenhuma pergunta; o 7B vai só de 13 para 14. O ganho nas perguntas inéditas mostra que as notas não estão apenas decoradas para as falhas dos lotes 1 e 2, mas o 7B aproveita bem menos delas.
- **No 14B, o ganho vem das notas, não da checagem.** Só uma pergunta passou por nova tentativa na v5 (contra três na v4); todas as perguntas recuperadas acertam já na primeira geração. Por isso a v5 é até mais rápida que a v4 no 14B. As notas acrescentam ~400 tokens ao prompt.
- Com 20 perguntas, cada uma vale 5 pontos percentuais no lote 3, então o número é indicativo, não conclusivo.

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
1. **A maioria das falhas restantes são erros silenciosos:** a SQL roda e devolve um resultado plausível. Uma autocorreção baseada em erro de execução não tem como pegá-los. O próximo passo natural é checar o resultado, e não só a execução. Foi o que a v5 fez (abaixo).
2. **Agregação é o ponto fraco nos dois tamanhos de modelo.** O 14B corrige os erros de leitura da pergunta, mas os erros de contar grupos e de nível de agregação continuam.
3. **A confusão de schema resiste à correção:** mesmo com a mensagem de erro e a dica de schema, o modelo volta a usar a coluna na tabela errada ou troca o identificador do cliente.

### O que a v5 mudou

A v5 ataca as duas causas acima sem chamada extra ao LLM:
- **Notas de semântica do schema.** O `CREATE TABLE` mostra nomes e tipos, mas não diz que um pedido pode ter vários pagamentos, que estado e `customer_unique_id` só existem em `customers`, nem que `order_item_id` é posição. As notas dizem isso em oito linhas gerais, sem citar perguntas do teste.
- **Checagem de formato do resultado.** Se a pergunta começa com "Quantos" (sem "cada") ou o resultado é uma coluna numérica vinda de um `GROUP BY`, e voltaram várias linhas, o resultado vai para a correção com a instrução de envolver o `GROUP BY` em uma subconsulta. A regra não marca nenhuma das 80 respostas de referência.

**14B, em relação à v4:** ganhou sete perguntas e não perdeu nenhuma. Cinco são de contar grupos (q22, q60, q75, q76, q78), resolvidas já na primeira geração pela nota "envolva o `GROUP BY ... HAVING` em uma subconsulta". Na q76 a v4 usava `order_item_id = 3`; a v5 conta os itens com `COUNT(*)`, como diz a nota. Na q80 o frete passou a ser somado por pedido em uma subconsulta antes do JOIN, também como a nota pede. Na q50 a v4 errava nas três tentativas ao usar um alias fora da subconsulta; a v5 escreve outra consulta que roda, e não dá para atribuir o acerto a uma nota específica.

As notas que tratam de granularidade foram ignoradas mesmo pelo 14B. Ele continua errando:
- q43: divisão inteira, mesmo com a nota sobre `1.0 *`;
- q56: `order_item_id` usado como quantidade de itens;
- q57 e q77: média por linha de pagamento em vez do total por pedido;
- q54: JOIN de itens com pagamentos que duplica as somas.

Ou seja, as notas funcionaram quando dizem qual padrão de SQL usar (subconsulta com `HAVING`, `COUNT(*)` para itens, somar por pedido antes de juntar) e foram ignoradas quando o modelo precisa perceber sozinho que está no nível de agregação errado.

**7B, em relação à v4:** ganhou q17 e q19 (as notas resolvem a ligação com `customers`), q60 e q75 (a checagem de formato pega a contagem de grupos e o modelo corrige). Perdeu q51, mas essa pergunta oscila entre rodadas também na v3 e na v4 (ver Limitações).

Falhas do 7B na v5 no lote 3 (6 de 20), quase todas de leitura da pergunta, que as notas não cobrem. O 14B erra só a q77:

| Pergunta | O que o modelo fez |
|---|---|
| q65 "Quantos clientes únicos são do RS?" | `COUNT(customer_unique_id)` sem `DISTINCT` (falha em todas as versões) |
| q76 "Quantos pedidos têm exatamente 3 itens?" | contou os itens desses pedidos, não os pedidos |
| q77 valor médio pago em pedidos com avaliação nota 1 | JOIN com avaliações e média em função de janela, sem agregar por pedido |
| q78 categorias com pelo menos 1000 itens vendidos | contou produtos cadastrados, não itens vendidos |
| q79 percentual de clientes com mais de um pedido | ignorou "mais de um pedido" e devolveu 100% |
| q80 tempo de entrega quando o frete passa de 100 reais | ignorou a condição do frete |

No 7B, os erros de q76 a q80 mostram o limite de corrigir com regras: o modelo responde uma pergunta parecida com a feita. Pegar isso, e os erros de granularidade que sobram no 14B, exigiria comparar a SQL com a pergunta, por exemplo pedindo ao LLM que revise a consulta e o resultado antes de responder.

## Limitações

- **80 perguntas é pouco.** Nos lotes 1+2 cada pergunta vale 1,7 ponto percentual, e no lote 3, 5 pontos. Diferenças de uma ou duas perguntas entre versões não são conclusivas.
- **Temperatura 0 não garante resultado idêntico.** Repetir v1 a v4 nas 60 perguntas antigas deu os mesmos acertos no 14B; no 7B mudou uma pergunta (a q51, de errada para certa na v3 e na v4). Uma rodada por configuração não mede essa variação.
- **Os lotes 1 e 2 já foram inspecionados.** As notas da v5 foram escritas a partir dessas falhas, por isso o número que conta para a v5 é o do lote 3. O lote 3 agora também foi inspecionado; uma técnica nova precisaria de perguntas novas.
