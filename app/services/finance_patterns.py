"""Padrões de detecção de consulta financeira, compartilhados entre o
`intent_classifier` (classificação de skill) e o `llm/router` (escolha de tier).

São duas regexes com propósitos diferentes — ver os comentários de cada uma.
"""

import re

# --- consulta de agregação, DEPENDENTE de contexto ---------------------------
# Usada pelo llm/router DEPOIS que a skill já é vida_casal (base "conversation").
# Como o contexto já garante o domínio "casal/finanças", pode ser larga — casa
# "quanto" sozinho. NÃO usar isto sem esse gate (sequestraria "quanto custa X").
FINANCIAL_QUERY_RE = re.compile(
    r"\b(quanto|qual\s+o\s+total|total\s+(?:de|gasto)|m[eé]dia|somat[oó]rio|"
    r"gastamos|gastei|gastaram|quanto\s+custou)\b",
    re.IGNORECASE,
)

# --- consulta de agregação, AUTOSSUFICIENTE ---------------------------------
# Usada pelo intent_classifier como short-circuit, ANTES de qualquer contexto.
# Estreita de propósito: "quanto" sozinho não basta — exige o radical de
# "gastar" (ou "custou" / "total do mês") por perto, para não sequestrar
# perguntas técnicas ("quanto custa uma GPU", "qual o total de endpoints").
# --- comando explícito de balanço -------------------------------------------
# `@yaannk balanço` / `balanço` / `balanco do mês`. Em GRUPO o webhook já removeu
# o gatilho antes do pipeline (`_activate_in_group`), então o texto chega como
# "balanço"; em conversa individual não há strip e o `@yaannk` pode vir junto —
# daí o prefixo opcional. Ancorado no início: é comando, não menção.
BALANCE_COMMAND_RE = re.compile(
    r"^\s*(?:@yaannk\s*)?balan[çc]o\b", re.IGNORECASE
)

FINANCIAL_QUERY_STRICT_RE = re.compile(
    r"\b("
    r"quanto\s+(?:\w+\s+){0,3}?gast\w+|"       # quanto (a gente / eu / nós) gastamos
    r"quanto\s+(?:\w+\s+){0,3}?custou|"
    r"total\s+(?:d[eo]s?\s+)?gast\w+|"          # total de/dos gastos
    r"total\s+gasto|"
    r"total\s+do\s+m[êe]s|"
    r"m[eé]dia\s+de\s+gast\w+|"
    r"somat[óo]rio\s+de\s+gast\w+|"
    r"quem\s+(?:gastou|gasta|pagou|paga)\s+mais|"  # "quem gastou mais?"
    r"gastamos|gastaram"                        # formas inequívocas, isoladas
    r")",
    re.IGNORECASE,
)
