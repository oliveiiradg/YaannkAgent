# YaannkAgent

Assistente pessoal que vive no WhatsApp e responde com base nas suas próprias anotações do Obsidian.

Você manda uma mensagem, ele busca no vault, processa com LLM e responde — tudo rodando na sua própria máquina.

## O que ele faz

- Consulta o vault do Obsidian via RAG híbrido (semântico + TF-IDF + reranker)
- Roteia automaticamente entre modelos locais (Ollama) e na nuvem (Kimi via OpenRouter) dependendo da complexidade da pergunta
- Registra gastos, lembretes e listas enviados pelo WhatsApp direto no vault
- Responde consultas financeiras por SQL em ~15ms, sem passar pelo LLM
- Funciona em grupo ou conversa individual

## Arquitetura

```
WhatsApp → Evolution API → Gateway (FastAPI)
                                ↓
                    Intent classifier + Query decomposer
                                ↓
                    RAG: MCP Obsidian + TF-IDF + bge-reranker
                                ↓
                    LLM Router → Ollama (local) ou Kimi (OpenRouter)
                                ↓
                          Resposta via WhatsApp
```

## Stack

- **Gateway**: Python + FastAPI
- **WhatsApp**: Evolution API (Baileys)
- **Vault**: Obsidian + MCP Connector
- **RAG**: RRF híbrido + bge-reranker-v2-m3 ONNX
- **LLMs**: Ollama (qwen2.5) local + Kimi K2.6/K2.7 via OpenRouter
- **Banco**: SQLite (conversas + gastos + telemetria)
- **Infra**: Ubuntu, systemd, OneDrive sync

## Rodando

```bash
git clone https://github.com/oliveiiradg/YaannkAgent
cd YaannkAgent/gateway
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# preencha o .env com suas credenciais (em produção: APP_ENV=production,
# WEBHOOK_SECRET forte e ALLOWED_NUMBERS são obrigatórios)
python -m app.main   # usa WEBHOOK_HOST/WEBHOOK_PORT do .env (default 127.0.0.1:5000)
```

Testes (inclui a suíte de segurança):

```bash
pip install -r requirements-dev.txt
pytest
```

O reranker precisa ser baixado separadamente:

```bash
# baixar de onnx-community/bge-reranker-v2-m3-ONNX no Hugging Face
# colocar em ~/YaannkAgent/models/bge-reranker-v2-m3/
```

## Configuração

O projeto tem dois arquivos de configuração separados:

**Gateway** (`gateway/.env`) — variáveis do assistente em si:
- Copie `gateway/.env.example` para `gateway/.env` e preencha

**Evolution API** (`deploy/evolution/.env`) — variáveis do Docker:
- Copie `deploy/evolution/.env.example` para `deploy/evolution/.env` e preencha
- Suba os containers: `docker compose -f deploy/evolution/docker-compose.yml up -d`

## Segurança

### Modelo de ameaça

O repositório é público. Assuma que o atacante lê todo o código, alcança
qualquer porta exposta e pode fabricar payloads. O objetivo do desenho abaixo
é que **nada disso dê acesso ao Vault nem permita acionar o pipeline como se
fosse o dono**.

### Autenticação do webhook

`POST /webhook` é a única entrada. A ordem é sempre:

```
requisição HTTP → limite de corpo → WEBHOOK_SECRET → validação do payload
→ extração do remetente → ALLOWED_NUMBERS → pipeline
```

- **`WEBHOOK_SECRET`** autentica a *origem* da requisição. É comparado em tempo
  constante (`hmac.compare_digest`) no middleware, antes do parsing do corpo.
  Erro de auth devolve `401` genérico (não revela se o segredo estava ausente
  ou errado) e nunca loga o valor.
  Gere com `openssl rand -hex 32` e configure o **mesmo** valor no header do
  webhook na Evolution API.
- **`ALLOWED_NUMBERS` / `ALLOWED_LIDS`** são *autorização de usuário*, não
  autenticação. O número dentro do JSON nunca é tratado como identidade — sem
  o `WEBHOOK_SECRET` válido a requisição não passa, então não dá para "entrar"
  só mandando `{"sender": "<meu numero>"}`.
- **Anti-replay**: eventos com `messageTimestamp` mais antigo que
  `REPLAY_MAX_SKEW_SECONDS` (300s) são rejeitados; `data.key.id` é deduplicado
  em memória.
- **Limites**: corpo do POST (`WEBHOOK_MAX_BODY_BYTES`, 256 KB) e texto da
  mensagem (`MESSAGE_MAX_CHARS`, 4000) — acima disso a requisição é cortada.

### Produção (`APP_ENV=production`)

O boot **falha** (`ConfigError`) se, em produção:

- `WEBHOOK_SECRET` estiver ausente ou tiver menos de 16 caracteres;
- não houver `ALLOWED_NUMBERS` nem `ALLOWED_LIDS`.

Em produção também: `/docs`, `/redoc` e `/openapi.json` desligados; logs de
conteúdo de mensagem/resposta desabilitados (`LOG_MESSAGE_CONTENT` ignorado).

### Rede / portas

| Serviço | Onde deve escutar |
|---|---|
| Gateway (FastAPI) | `WEBHOOK_HOST` — default `127.0.0.1`. Só amplie se a Evolution rodar noutra máquina/container, e sempre atrás de firewall. |
| Evolution API | `127.0.0.1:8080` (docker-compose já fixa o bind no loopback). Para acesso externo, proxy reverso com TLS + auth. |
| Postgres (Evolution) | rede interna do Compose, sem `ports:`. |
| Ollama | `127.0.0.1:11434` (ou LAN confiável). **Nunca** exposto. |
| MCP do Obsidian | `127.0.0.1:27200`, autenticado por `OBSIDIAN_MCP_TOKEN`. **Nunca** exposto. |

### Filesystem / Vault

- Todo acesso a arquivo do Vault passa por `safe_vault_path()`, que resolve o
  caminho e recusa qualquer coisa fora da raiz (`../`, caminho absoluto,
  symlink que escapa, byte nulo).
- Não há endpoint que leia arquivo arbitrário; o pipeline só lê `.md` sob
  `VAULT_PATH`.

### Logs — o que **não** é registrado

Conteúdo do Vault, contexto do RAG, texto de mensagens/respostas, valores e
descrições de gastos, nomes de arquivos do Vault em nível `INFO`,
`WEBHOOK_SECRET`, `EVOLUTION_API_KEY`, `OBSIDIAN_MCP_TOKEN`, `KIMI_API_KEY`,
headers de autenticação, payloads completos. Os logs de RAG guardam só
contagens (`N docs, M chars`). Caminhos de nota vão só em `DEBUG`.

### Prompt injection

Documento malicioso no Vault é tratado como **dado**: o system prompt do
Bloco 3 instrui a ignorar instruções embutidas em trechos recuperados, a não
revelar o prompt e a não descrever outros arquivos do Vault fora do escopo da
pergunta. Isso reduz o impacto, não é uma garantia — mantenha o MCP com o
mínimo de ferramentas.

### Rotação de segredos

1. Gere o novo valor (`openssl rand -hex 32` para o webhook; painéis do
   provedor para `KIMI_API_KEY` / Evolution / MCP).
2. Atualize `gateway/.env` (e `deploy/evolution/.env` para a Evolution).
3. Atualize o header do webhook na Evolution API com o novo `WEBHOOK_SECRET`.
4. Reinicie `yaannk-gateway` (e `yaannk-evolution` se aplicável).
5. Revogue o valor antigo no provedor.

### Higiene do repositório

- `.env`, `*.db`/`*.sqlite`, `*.log`, `data/`, chaves (`*.pem`, `*.key`) e o
  Vault ficam fora do Git (ver `.gitignore`).
- `.env.example` documenta as variáveis **sem valores reais**.
- Rode a suíte de segurança antes de publicar: `pytest` (ver `tests/`).

```
        INTERNET
           │
           ▼
      Evolution API
           │
    X-Webhook-Secret
           │
           ▼
  ┌─────────────────┐
  │ FastAPI Gateway │
  └────────┬────────┘
           │
     sender allowlist
           │
           ▼
     Intent / RAG
           │
  ┌────────┴────────┐
  ▼                 ▼
Ollama             Kimi
(local)         (OpenRouter)
                    │
                    ▼
            Obsidian Vault
```

## Variáveis de ambiente

Veja `.env.example` para a lista completa. O essencial:

```
APP_ENV=production          # development | production | test
WEBHOOK_SECRET=             # obrigatório em production (openssl rand -hex 32)
WEBHOOK_HOST=127.0.0.1      # amplie só se a Evolution rodar noutra máquina
ALLOWED_NUMBERS=            # obrigatório em production (DDI+DDD+número, csv)
EVOLUTION_API_URL=
EVOLUTION_API_KEY=
EVOLUTION_INSTANCE_NAME=
OBSIDIAN_MCP_TOKEN=
VAULT_PATH=
KIMI_API_KEY=              # opcional, para os tiers 1 e 2 do router
OLLAMA_DESKTOP_URL=        # opcional, nó GPU externo (LAN confiável)
```

## Benchmark

O projeto tem uma suite de 68 casos cobrindo parsing, RAG, consultas financeiras e conversação:

```bash
venv/bin/python scripts/benchmark.py           # determinístico (~0s)
venv/bin/python scripts/benchmark.py --llm     # pipeline real (~$0.10)
```