# YaannkAgent Gateway

## Início de sessão
1. Ler o STATE.md do vault: `01 - Projetos/Pessoal/YaannkAgent/YaannkAgent - STATE.md`
2. Checar o gateway: `systemctl --user status yaannk-gateway.service`
3. Checar o git: `git log --oneline -5 && git status`

## Fim de sessão
- Executar `/yaannk-session-end` (obrigatório)

## Antes de alterar qualquer `.py`
- Rodar `pytest`. Os 8 testes que falham são esperados (fixtures desatualizadas).

## Nunca
- Usar regex para detectar intenção
- Usar `reply_equals` para validar resposta de LLM
- Fazer commit (nem `git add`/`git commit`/`git push`). O Douglas commita manualmente quando achar necessário — isso vale inclusive dentro de `/yaannk-session-end`
- Mencionar Claude/AI nos commits
- Criar fixtures com path `03 - Vida/`

## Sempre
- Documentar decisões em `Decisões/D-NN` no vault
- Atualizar o STATE.md
- Testar no WhatsApp antes de commitar

## Dados em produção
1. A agenda da Bia tem dados reais de clientes. Qualquer mudança em `agenda_bia.py` ou no código que escreve no vault exige teste com fixture antes de tocar em produção.
2. Antes de qualquer mudança em código que escreve no vault, fazer backup do arquivo afetado.
3. `AGENT_ENABLED=false` é o rollback. Confirmar que funciona antes de cada sessão de desenvolvimento pesado.
4. Nunca testar escrita diretamente no vault de produção. Nos testes, apontar para um vault temporário: `GASTOS_ROOT_OVERRIDE` (gastos) ou `VAULT_PATH` temporário (o `tests/conftest.py` já usa `/tmp/yaannk-test-vault`), com o MCP mockado.

## Arquitetura (D-11, substitui D-10 com `AGENT_ENABLED=true`)
- Toda mensagem vai para `app/agents/yaannk_agent.py`: Kimi com tool calling nativo, conversa com autor, ferramentas do MCP do Obsidian + ferramentas de domínio.
- Dado exato (contas, gastos, agenda) sai das ferramentas de domínio, que já somam e mantêm o formato das notas. Não criar fast-path nem regex de intenção novos.
- Capacidade nova = ferramenta nova em `_FERRAMENTAS_DOMINIO` (ou liberar ferramenta do MCP em `_MCP_TOOLS`), não if/regex no pipeline.
- Flags: `AGENT_ENABLED`, `AGENT_MAX_STEPS`, `AGENT_TIMEOUT_S`, `AGENT_REASONING` (off/low/medium/high/on).
- Rollback: `AGENT_ENABLED=false` volta ao pipeline antigo (D-10: fast-paths → orquestrador → ReAct), que continua no código.

## Stack
- Gateway: porta 5000
- MCP: porta 27200
- LLM: Kimi K2.6 via OpenRouter
- Ollama local

## Skills
- `/yaannk-session-end` — obrigatório no fim de sessão
- `/yaannk-clean-code`
- `/yaannk-logging`
- `/yaannk-deploy`
- `/fastapi-clean-architecture`
- `/async-testing-expert`
- `/brazilian-financial-integration`

## Vault paths
- Gastos: `02 - Áreas/Finanças/Gastos-AAAA-MM.md`
- Contas: `02 - Áreas/Finanças/Contas-AAAA-MM.md`
- STATE: `01 - Projetos/Pessoal/YaannkAgent/YaannkAgent - STATE.md`
- Decisões: `01 - Projetos/Pessoal/YaannkAgent/Decisões/D-NN - Título.md`
