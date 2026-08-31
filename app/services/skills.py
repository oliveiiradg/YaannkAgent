"""Skills por domínio.

Uma "skill" aqui não é mágica: é um dicionário com 3 campos que o pipeline ativa
conforme o domínio da pergunta:
  - keywords: termos que disparam a skill (matching simples em intent_classifier)
  - priority_folder: pasta do vault que ganha prioridade na busca (Bloco 2)
  - prompt_injection: instrução extra somada ao system prompt do Bloco 3

Ajuste `priority_folder` para a estrutura do seu vault.
"""

SKILLS: dict[str, dict] = {
    "yaannk_tecnico": {
        "keywords": [
            "yaannk", "gateway", "rag", "pipeline", "reranker",
            "ollama", "evolution api", "webhook", "fastapi",
            "vault_search", "systemd", "docker",
        ],
        "priority_folder": "01 - Projetos/Pessoal/YaannkAgent",
        "prompt_injection": (
            "Cite o arquivo de origem. Se for sobre código, copie o trecho "
            "exato se disponível."
        ),
    },
    "pessoas": {
        "keywords": ["quem é", "quem faz", "responsável", "contato"],
        "priority_folder": "02 - Áreas/Pessoas",
        "prompt_injection": (
            "Responda com: nome, cargo/papel e contexto de relacionamento "
            "com o usuário."
        ),
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
    },
    "default": {
        "keywords": [],
        "priority_folder": "",
        "prompt_injection": "",
    },
}


def get_skill(skill_name: str) -> dict:
    """Retorna a config da skill; cai em 'default' se o nome não existir."""
    return SKILLS.get(skill_name, SKILLS["default"])
