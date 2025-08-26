# README — `agent.py`

Um **worker assíncrono** para processar uma lista de *inputs* (primeira coluna de um CSV), enviar cada item para a API do Toqan, **aguardar a resposta via polling** e salvar um JSON por item. Opcionalmente, o script **consolida os JSONs em um CSV** e **persiste os resultados no Databricks** via SQL Statements API.

---

## Requisitos

- Python 3.10+ (recomendado 3.11/3.12)
- Pacotes:
  - `aiohttp`
  - `python-dotenv` *(opcional, para carregar `.env` automaticamente)*

Instale com:

```bash
pip install aiohttp python-dotenv
```

---

## Arquivo `.env` (obrigatório para uso confortável)

Crie um `.env` na raiz do seu projeto (ou exporte como variáveis de ambiente no shell).  
Exemplo mínimo:

```env
# Chave da API Toqan
TOQAN_API_KEY=sk_xxx...

# Databricks (necessários somente se você usar --dbricks)
DATABRICKS_HOST=https://dbc-xxxx.cloud.databricks.com
DATABRICKS_TOKEN=dapi_xxx...
DATABRICKS_WAREHOUSE_ID=xxxx-xxxx-xxxx
DATABRICKS_TABLE=groceries_sandbox.toqan_answers
```

> O script tenta carregar o `.env` automaticamente ao iniciar (se `python-dotenv` estiver instalado).

---

## Entrada (CSV)

- O script lê **apenas a primeira coluna** do CSV informado em `--csv`.
- Cada linha é um *input* a ser enviado à API.
- O script **reescreve** o CSV para manter **apenas os itens pendentes** (*à prova de quedas*).

> **Dica:** tenha um backup do CSV original caso não queira que ele seja reescrito.

---

## Saída

- Por padrão, cria a pasta `outputs/YYYYMMDD_HHMMSS/`.
- Para cada *input*, grava um arquivo `JSON` com a resposta da API.
- Se `--collect` for usado, gera também um `answers.csv` consolidado.
- Se `--dbricks` for usado, faz `MERGE` na tabela informada.

---

## Uso básico

```bash
python agent.py \
  --csv toqan/input.csv \
  --api-key "$TOQAN_API_KEY"
```

Sem `--outdir`, a saída irá para `outputs/<timestamp>/`.

---

## Parâmetros

| Parâmetro              | Tipo/Default                  | Descrição |
|------------------------|-------------------------------|-----------|
| `--csv`                | **Obrigatório**               | Caminho do CSV de entrada (primeira coluna = `input`). |
| `--api-key`            | `os.getenv("TOQAN_API_KEY")`  | Chave da API Toqan. Pode vir do `.env`. |
| `--concurrency`        | `50`                          | Número de requisições simultâneas (semáforo assíncrono). |
| `--poll-interval`      | `30`                          | Intervalo (s) entre *polls* à API por resposta. |
| `--outdir`             | `outputs/<timestamp>`         | Pasta de saída dos JSONs/consolidados. |
| `--collect`            | *flag*                        | Se presente, consolida os JSONs em CSV ao final. |
| `--collect-out`        | `outdir/answers.csv`          | Caminho do CSV consolidado (se `--collect`). |
| `--collect-pattern`    | `"*.json"`                    | *Glob* para escolher quais JSONs consolidar. |
| `--collect-sep`        | `"."`                         | Separador ao achatar chaves do JSON (`a.b.c`). |
| `--dbricks`            | *flag*                        | Se presente, envia os resultados ao Databricks (MERGE). |
| `--dbricks-host`       | `os.getenv("DATABRICKS_HOST")` | URL do workspace (ex.: `https://dbc-xxxx.cloud.databricks.com`). |
| `--dbricks-token`      | `os.getenv("DATABRICKS_TOKEN")` | Token do Databricks. |
| `--dbricks-warehouse-id` | `os.getenv("DATABRICKS_WAREHOUSE_ID")` | Warehouse ID para Statements API. |
| `--dbricks-table`      | `os.getenv("DATABRICKS_TABLE")` | Tabela destino no formato `schema.tabela`. |
| `--dbricks-batch-size` | `1000`                        | Tamanho do sublote no envio ao Databricks. |
| `--agent`              | *opcional*                    | Nome do agente (aparece na coluna `agent`); se omitido, deriva da API key mascarada. |

---

## Exemplos

### 1) Rodar simples, saída padrão

```bash
python agent.py \
  --csv toqan/input.csv \
  --api-key "$TOQAN_API_KEY"
```

### 2) Definir pasta de saída e consolidar JSONs em CSV

```bash
python agent.py \
  --csv toqan/input.csv \
  --api-key "$TOQAN_API_KEY" \
  --outdir toqan/results_taxonomy2 \
  --collect
```

### 3) Consolidar com nome/regex próprios

```bash
python agent.py \
  --csv toqan/input.csv \
  --api-key "$TOQAN_API_KEY" \
  --outdir toqan/results \
  --collect \
  --collect-out toqan/results/minha_consolidacao.csv \
  --collect-pattern "*.resposta.json" \
  --collect-sep "."
```

### 4) Enviar para Databricks (MERGE)

> Requer variáveis do Databricks configuradas (via `.env` ou parâmetros).

```bash
python agent.py \
  --csv toqan/input.csv \
  --api-key "$TOQAN_API_KEY" \
  --dbricks \
  --dbricks-table groceries_sandbox.toqan_answers
```

Ou especificando tudo na linha de comando:

```bash
python agent.py \
  --csv toqan/input.csv \
  --api-key "$TOQAN_API_KEY" \
  --dbricks \
  --dbricks-host "https://dbc-xxxx.cloud.databricks.com" \
  --dbricks-token "$DATABRICKS_TOKEN" \
  --dbricks-warehouse-id "$DATABRICKS_WAREHOUSE_ID" \
  --dbricks-table "groceries_sandbox.toqan_answers" \
  --dbricks-batch-size 1000
```

### 5) Controlar concorrência, polling e nome do agente

```bash
python agent.py \
  --csv toqan/input.csv \
  --api-key "$TOQAN_API_KEY" \
  --concurrency 50 \
  --poll-interval 30 \
  --agent "taxonomy_v2"
```

---

## Comportamentos importantes

- **Skip automático:** se já existir `JSON` para um *input* na pasta de saída, ele **pula** esse item.
- **CSV pendente:** a cada *input* concluído, o CSV original é **reescrito** contendo apenas os itens **ainda não processados**.
- **Consolidação (`--collect`):** lê todos os JSONs conforme o *glob* e gera um CSV achatando o `answer` (com o separador definido).
- **Databricks (`--dbricks`):** executa um `MERGE` por sublotes (configurável), atualizando registros quando `created_at` novo > existente.

---

## Erros comuns & dicas

- **`Defina --api-key...`** → garanta `TOQAN_API_KEY` no `.env` ou passe `--api-key`.
- **Databricks faltando parâmetros** → use `.env` com `DATABRICKS_*` ou passe todos os `--dbricks-*`.
- **Timeouts/429** → reduza `--concurrency`, aumente `--poll-interval` ou rode novamente (*idempotente* graças ao *skip*).