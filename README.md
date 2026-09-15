# YaannkAgent

Assistente pessoal via WhatsApp com acesso ao vault do Obsidian. Atende o Douglas e a Bia em finanças (contas fixas e gastos), agenda de clientes, listas e consultas ao conteúdo das notas — tudo rodando numa máquina própria.

## O que ele faz

- **Grupo do casal**: consulta e registra contas fixas e gastos, marca conta como paga, adiciona e remove contas, mantém lista de compras e datas importantes
- **Privado da Bia**: marca, desmarca e consulta atendimentos de clientes na agenda
- **Privado do Douglas**: responde perguntas sobre qualquer nota do vault, lê, cria e edita notas
- **Proativo**: todo dia às 08h avisa no grupo as contas que vencem nos próximos dias e manda para a Bia a agenda do dia; no dia 01 cria a nota de contas do mês novo

## Arquitetura

```
WhatsApp ─► Evolution API ─► POST /webhook (gateway FastAPI)
                                   │
                                   ▼
                  agente (Kimi K2.6 + tool calling nativo)
                     │                              │
                     ▼                              ▼
          MCP do Obsidian                 ferramentas de domínio
   (buscar, ler, criar, editar,     (contas, gastos, agenda — somam e
    mover e apagar notas)            mantêm o formato das notas)
                     │                              │
                     └──────────────► vault ◄───────┘
                                   │
                                   ▼
                        resposta ─► Evolution API ─► WhatsApp


n8n (Schedule) ─► orquestrador ─► subworkflows ─► GET/POST /proactive/* ─► Evolution API
```

- **Agente** (`app/agents/yaannk_agent.py`): recebe as últimas mensagens da conversa com o nome de quem falou, decide quais ferramentas chamar e responde no formato do WhatsApp. Tem limite de passos e de tempo, e responde com o que já levantou quando o prazo aperta.
- **Ferramentas de domínio**: contas do mês ordenadas por vencimento com totais já somados, gastos por período (por pessoa e categoria), registro de gasto, agenda de clientes. Dado exato não fica a cargo do modelo.
- **Proteções de escrita**: nada fora do vault, nada em `.obsidian/`, e reescrita de nota de contas ou gastos é recusada se quebrar o formato lido pelo sistema.
- **Proativo** (n8n): `yaannk_daily_orchestrator` (08h) chama `yaannk_contas_alert` e `yaannk_agenda_bia`; `yaannk_monthly_orchestrator` (dia 01) chama `yaannk_create_next_month`. Os endpoints `/proactive/*` são determinísticos — sem modelo de linguagem.

## Stack

- **Linguagem / API**: Python 3, FastAPI, Uvicorn
- **Modelo**: Kimi K2.6 via OpenRouter (API compatível com OpenAI)
- **Vault**: Obsidian + MCP Connector (porta 27200)
- **WhatsApp**: Evolution API
- **Automação**: n8n
- **Banco**: SQLite (histórico de conversa e telemetria de chamadas ao modelo)
- **Infra**: Linux, systemd, Docker (Evolution API e n8n)

## Estrutura

```
app/
  main.py                 app FastAPI, middleware de autenticação do webhook, /health
  config.py               leitura e validação do .env
  routes/webhook.py       POST /webhook — identidade, allowlist, histórico, envio
  routes/proactive.py     /proactive/* — endpoints chamados pelo n8n
  agents/yaannk_agent.py  agente com ferramentas (caminho principal)
  agents/agent_vida.py    pipeline anterior (rollback)
  services/bills.py       contas fixas: leitura, marcar paga, adicionar, remover, mês novo
  services/agenda_bia.py  agenda de clientes
  services/vault_writer.py escrita no vault via MCP
  services/llm/           providers de modelo e telemetria
tests/                    pytest (agente, contas, endpoints proativos, segurança do webhook)
deploy/evolution/         docker-compose da Evolution API
```

## Como rodar

```bash
git clone https://github.com/oliveiiradg/YaannkAgent
cd YaannkAgent
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # preencher (ver "Variáveis de ambiente")
```

Em produção o gateway roda como serviço do usuário no systemd, escutando só na bridge do Docker (`172.17.0.1:5000`), que é por onde a Evolution API e o n8n o alcançam:

```bash
systemctl --user start yaannk-gateway.service
systemctl --user status yaannk-gateway.service
curl http://172.17.0.1:5000/health
```

O unit executa `python -m uvicorn app.main:app --host 172.17.0.1 --port 5000` a partir da pasta do projeto.

A Evolution API sobe pelo compose em `deploy/evolution/` (copie `deploy/evolution/.env.example` para `deploy/evolution/.env` antes):

```bash
docker compose -f deploy/evolution/docker-compose.yml up -d
```

No webhook da instância na Evolution, configure a URL `http://172.17.0.1:5000/webhook`, o evento `MESSAGES_UPSERT` e o header `x-webhook-secret` com o valor de `WEBHOOK_SECRET`.

### Testes

```bash
pip install -r requirements-dev.txt
pytest
```

Os testes não tocam vault, MCP, modelo nem Evolution reais: o vault aponta para um diretório temporário e as chamadas externas são substituídas.

## Variáveis de ambiente

Copie `.env.example` para `.env` e preencha. O arquivo documenta todas as variáveis, sem valores reais. Os grupos principais:

- **Ambiente**: `APP_ENV`
- **Evolution API e webhook**: `EVOLUTION_*`, `WEBHOOK_*`, `ALLOWED_NUMBERS`, `KNOWN_NAMES`, `GROUP_JID`, `DOUGLAS_JID`, `BIA_JID`
- **Endpoints proativos**: `PROACTIVE_TOKEN`
- **Modelo**: `KIMI_*`, `LLM_*`
- **Agente**: `AGENT_ENABLED`, `AGENT_MAX_STEPS`, `AGENT_TIMEOUT_S`, `AGENT_REASONING`
- **Vault**: `VAULT_PATH`, `OBSIDIAN_MCP_URL`, `OBSIDIAN_MCP_TOKEN`

## Rollback

O pipeline anterior (fast-paths determinísticos → classificação de intenção → busca no vault) continua no código. Para voltar a ele:

```bash
# no .env
AGENT_ENABLED=false

systemctl --user restart yaannk-gateway.service
```

## Segurança

O repositório é público: assuma que qualquer pessoa lê o código e pode fabricar requisições. O desenho abaixo impede que isso dê acesso ao vault ou acione o assistente como se fosse o dono.

### Entradas autenticadas

| Entrada | Autenticação | Sem credencial |
|---|---|---|
| `POST /webhook` | header `x-webhook-secret` = `WEBHOOK_SECRET` | `401` |
| `/proactive/*` | header `x-proactive-token` = `PROACTIVE_TOKEN` | `401` |
| `GET /health` | pública, não revela nada | — |

- Segredos comparados em tempo constante (`hmac.compare_digest`); a resposta de erro não diz se o valor estava ausente ou errado, e o valor nunca vai para o log.
- `ALLOWED_NUMBERS` / `ALLOWED_LIDS` são **autorização** de usuário, não autenticação: o número dentro do payload só vale depois do segredo do webhook.
- **Anti-replay**: eventos com `messageTimestamp` mais antigo que `REPLAY_MAX_SKEW_SECONDS` são rejeitados, e `data.key.id` é deduplicado.
- **Limites**: corpo do POST (`WEBHOOK_MAX_BODY_BYTES`) e texto da mensagem (`MESSAGE_MAX_CHARS`).

### Produção (`APP_ENV=production`)

O boot falha se, em produção:

- `WEBHOOK_SECRET` estiver ausente ou tiver menos de 16 caracteres;
- `PROACTIVE_TOKEN` estiver ausente ou tiver menos de 16 caracteres;
- não houver `ALLOWED_NUMBERS` nem `ALLOWED_LIDS`.

Em produção também: `/docs`, `/redoc` e `/openapi.json` desligados, e log de conteúdo de mensagem desabilitado.

### Rede

| Serviço | Onde escuta |
|---|---|
| Gateway | `172.17.0.1:5000` (bridge do Docker) — não responde em `localhost` nem na rede local |
| Evolution API | `127.0.0.1:8080` |
| n8n | `127.0.0.1:5678` |
| MCP do Obsidian | `127.0.0.1:27200`, autenticado por `OBSIDIAN_MCP_TOKEN` |

### Vault

- Leitura direta do disco passa por `safe_vault_path()`, que recusa caminho fora da raiz (`../`, absoluto, symlink que escapa).
- O agente não escreve fora do vault nem em `.obsidian/`, e não tem acesso a busca-e-substituição em massa, comandos do Obsidian nem requisições para a internet.

### Logs

Não são registrados: segredos e tokens, headers de autenticação, texto de mensagens e respostas (em produção), conteúdo de notas passado para escrita. Chamadas de ferramenta do agente aparecem com os argumentos, sem o conteúdo das notas.

### Rotação de segredos

1. Gere o novo valor: `openssl rand -hex 32`.
2. Atualize o `.env` do gateway.
3. `WEBHOOK_SECRET`: atualize o header do webhook da instância na Evolution. `PROACTIVE_TOKEN`: atualize o header dos nós HTTP do n8n que chamam `/proactive/*` e publique os workflows de novo.
4. `systemctl --user restart yaannk-gateway.service`.

## Benchmark

`scripts/benchmark.py` tem 68 casos do pipeline anterior (parsing, busca no vault, consultas financeiras, conversação):

```bash
venv/bin/python scripts/benchmark.py           # determinístico
venv/bin/python scripts/benchmark.py --llm     # pipeline real (custo de API)
```
