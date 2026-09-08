"""Skills por domínio.

Uma "skill" aqui não é mágica: é um dicionário com 4 campos que o pipeline ativa
conforme o domínio da pergunta:
  - keywords: termos que disparam a skill (matching simples em intent_classifier)
  - priority_folder: pasta do vault que ganha prioridade na busca (Bloco 2)
  - prompt_injection: instrução extra somada ao system prompt do Bloco 3
  - excluded_subfolders: subpastas de priority_folder descartadas do pool do
    RAG (Bug 10, Sessão 17) — documentos de planejamento/spec dentro do
    escopo certo mas de fase errada (rascunho, não é o estado real do
    código) competindo no RRF/reranker contra a documentação real

Ajuste `priority_folder` para a estrutura do seu vault.
"""

SKILLS: dict[str, dict] = {
    "yaannk_tecnico": {
        "keywords": [
            "yaannk", "gateway", "rag", "pipeline", "reranker",
            "ollama", "evolution api", "webhook", "fastapi",
            "vault_search", "systemd", "docker",
            # Sessão 22 cont.: "quais são os tiers de roteamento?" não casava
            # nenhuma keyword (caía em `default`) e nunca chegava ao RAG
            # técnico. Jargão do LLM Router — inequívoco no vault do YaannkAgent.
            "tier", "tiers", "roteamento", "router", "orquestrador",
        ],
        "priority_folder": "01 - Projetos/Pessoal/YaannkAgent",
        "prompt_injection": (
            "Cite o arquivo de origem. Se for sobre código, copie o trecho "
            "exato se disponível."
        ),
        "excluded_subfolders": [
            "01 - Projetos/Pessoal/YaannkAgent/Especificação - Arquitetura Multi-Agentes",
            "01 - Projetos/Pessoal/YaannkAgent/Sessões",
            # Caminho fantasma no índice semântico do MCP — a pasta não existe
            # mais em disco (renomeada para "Especificação - Arquitetura
            # Multi-Agentes"), mas o índice do plugin do Obsidian não foi
            # reindexado e ainda devolve candidatos com esse prefixo (achado
            # na Sessão 17, diagnóstico do Bug 10).
            "01 - Projetos/Pessoal/YaannkAgent/SDD - Arquitetura Multi-Agentes",
        ],
    },
    "pessoas": {
        "keywords": ["quem é", "quem faz", "responsável", "contato"],
        "priority_folder": "02 - Áreas/Pessoas",
        "prompt_injection": (
            "Responda com: nome, cargo/papel e contexto de relacionamento "
            "com o usuário."
        ),
        "excluded_subfolders": [],
    },
    "pendencias": {
        "keywords": [
            "pendente", "falta", "próximo passo", "o que falta",
            "o que está aberto", "status", "próxima ação",
        ],
        "priority_folder": "",
        "prompt_injection": (
            "Liste em ordem de prioridade. Máximo 7 itens. Cite a nota de "
            "origem de cada um."
        ),
        "excluded_subfolders": [],
    },
    "vida_casal": {
        "keywords": [
            "gasto", "gastei", "gastamos", "gastaram", "gastam", "gasta",
            "total do mês", "total do mes", "total gasto", "total de gastos",
            "mercado", "conta", "conta de", "financas",
            "finanças", "lista", "compras", "data", "aniversario", "aniversário",
            "lembrete", "lembra", "pagamento", "paguei", "aluguel", "salário",
            "salario", "boleto", "fatura", "condomínio", "condominio",
        ],
        "priority_folder": "03 - Vida",
        "prompt_injection": (
            "Você está ajudando o casal a organizar a vida em comum. "
            "Seja prático e direto. Quando alguém registrar uma informação "
            "(gasto, data, lembrete), confirme o que foi salvo e onde."
        ),
        "excluded_subfolders": [],
    },
    # --- Fase D (Sessão 20) — pastas confirmadas por busca no vault, não
    # assumidas: `05 - Faculdade/` e `02 - Áreas/Trabalho/` existem.
    "conhecimento": {
        "keywords": [
            "faculdade", "semestre", "matéria", "materia", "aula", "prova",
            "trabalho da facul", "hackathon", "curso", "certificação",
            "certificacao", "estácio", "estacio", "cronograma", "leitura",
        ],
        "priority_folder": "05 - Faculdade",
        "prompt_injection": (
            "Contexto de estudos: faculdade, cursos e certificações. "
            "Cite a nota de origem. Se houver prazo ou data, destaque."
        ),
        "excluded_subfolders": [],
    },
    "carreira": {
        "keywords": [
            "currículo", "curriculo", "cv", "carreira", "vaga", "entrevista",
            "perfil profissional", "linkedin", "competência", "competencia",
            "trajetória", "trajetoria", "senioridade", "salário de mercado",
        ],
        "priority_folder": "02 - Áreas/Trabalho",
        "prompt_injection": (
            "Contexto de carreira: perfil profissional, currículo, "
            "competências e trajetória. Seja concreto e cite a nota de origem."
        ),
        "excluded_subfolders": [],
    },
    "default": {
        "keywords": [],
        "priority_folder": "",
        "prompt_injection": "",
        "excluded_subfolders": [],
    },
}


def get_skill(skill_name: str) -> dict:
    """Retorna a config da skill; cai em 'default' se o nome não existir."""
    return SKILLS.get(skill_name, SKILLS["default"])
