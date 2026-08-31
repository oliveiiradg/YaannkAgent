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
# preencha o .env com suas credenciais
uvicorn app.main:app --host 0.0.0.0 --port 5000
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

- Nunca exponha o gateway diretamente na internet pública — rode atrás de firewall/NAT.
- Configure `WEBHOOK_SECRET` para autenticar requisições da Evolution API.
- Mantenha o Ollama dentro da rede local — nunca exponha a porta 11434 publicamente.
- Nunca suba arquivos `.env`, bancos SQLite ou logs no repositório.
- O vault do Obsidian é mantido fora do repositório intencionalmente.

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
EVOLUTION_API_URL=
EVOLUTION_API_KEY=
EVOLUTION_INSTANCE_NAME=
OBSIDIAN_MCP_TOKEN=
VAULT_PATH=
KIMI_API_KEY=        # opcional, para os tiers 1 e 2 do router
OLLAMA_DESKTOP_URL=  # opcional, nó GPU externo
```

## Benchmark

O projeto tem uma suite de 68 casos cobrindo parsing, RAG, consultas financeiras e conversação:

```bash
venv/bin/python scripts/benchmark.py           # determinístico (~0s)
venv/bin/python scripts/benchmark.py --llm     # pipeline real (~$0.10)
```