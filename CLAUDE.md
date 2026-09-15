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
- Mencionar Claude/AI nos commits
- Criar fixtures com path `03 - Vida/`

## Sempre
- Documentar decisões em `Decisões/D-NN` no vault
- Atualizar o STATE.md
- Testar no WhatsApp antes de commitar

## Arquitetura (D-10)
- Kimi é o cérebro.
- Fast-paths só para dado puro: `_try_balance`, `_try_expenses`, `_try_query_agenda`, `_try_shopping_list`, `_try_important_dates`, `_try_bills`.
- ReAct para qualquer intenção em linguagem natural.

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
