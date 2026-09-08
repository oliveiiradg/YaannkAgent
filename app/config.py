import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

# Ambientes reconhecidos. `production` liga as validações rígidas (secret do
# webhook obrigatório, allowlist obrigatória, /docs desligada por padrão).
_ENVS = ("development", "production", "test")


class ConfigError(RuntimeError):
    """Configuração inválida — a aplicação não deve subir assim."""


@dataclass(frozen=True)
class Settings:
    app_env: str
    evolution_api_url: str
    evolution_api_key: str
    evolution_instance_name: str
    webhook_host: str
    webhook_port: int
    webhook_secret: str
    webhook_max_body_bytes: int
    message_max_chars: int
    replay_max_skew_seconds: int
    docs_enabled: bool
    log_message_content: bool
    allowed_numbers: list[str]
    lid_map: dict[str, str]
    allowed_lids: list[str]
    known_names: dict[str, str]
    owner_name: str
    partner_name: str
    extra_stopwords: list[str]
    yaannk_number: str
    yaannk_lid: str
    llm_default_provider: str
    llm_telemetry: bool
    llm_router_enabled: bool
    kimi_api_key: str
    kimi_base_url: str
    kimi_model_tier1: str
    kimi_model_tier2: str
    kimi_min_max_tokens: int
    kimi_price_in_per_mtok: float
    kimi_price_out_per_mtok: float
    ollama_url: str
    ollama_desktop_url: str
    ollama_model: str
    vault_path: str
    obsidian_mcp_url: str
    obsidian_mcp_token: str
    reranker_model_dir: str
    backlink_weight: float
    agentic_rag_enabled: bool
    agentic_rag_max_searches: int
    orchestrator_enabled: bool
    orchestrator_timeout_s: float


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("true", "1", "yes", "on")


def _load_settings() -> Settings:
    app_env = os.environ.get("APP_ENV", "development").strip().lower()
    if app_env not in _ENVS:
        raise ConfigError(
            f"APP_ENV inválido: {app_env!r} (use um de {', '.join(_ENVS)})"
        )
    is_prod = app_env == "production"

    webhook_secret = os.environ.get("WEBHOOK_SECRET", "").strip()
    allowed_numbers = [
        n.strip()
        for n in os.environ.get("ALLOWED_NUMBERS", "").split(",")
        if n.strip()
    ]
    allowed_lids = [
        l.strip()
        for l in os.environ.get("ALLOWED_LIDS", "").split(",")
        if l.strip()
    ]

    # --- validações rígidas de produção ---------------------------------
    # O webhook é a única porta de entrada do sistema; sem segredo, qualquer
    # um que alcance a porta consegue acionar o pipeline como se fosse o dono.
    if is_prod and not webhook_secret:
        raise ConfigError(
            "WEBHOOK_SECRET é obrigatório quando APP_ENV=production "
            "(gere um valor forte e configure o MESMO nos headers do webhook "
            "da Evolution API)."
        )
    if is_prod and len(webhook_secret) < 16:
        raise ConfigError(
            "WEBHOOK_SECRET fraco (< 16 caracteres) em APP_ENV=production."
        )
    if is_prod and not allowed_numbers and not allowed_lids:
        raise ConfigError(
            "ALLOWED_NUMBERS (ou ALLOWED_LIDS) é obrigatório em "
            "APP_ENV=production — sem allowlist ninguém deveria ser atendido."
        )

    return Settings(
        app_env=app_env,
        evolution_api_url=os.environ["EVOLUTION_API_URL"].rstrip("/"),
        evolution_api_key=os.environ["EVOLUTION_API_KEY"],
        evolution_instance_name=os.environ["EVOLUTION_INSTANCE_NAME"],
        # Interface de bind do uvicorn. Default seguro: só loopback. Quando a
        # Evolution API roda em container Docker e precisa alcançar o gateway
        # no host, use o IP da bridge do Docker (ex: 172.17.0.1) ou 0.0.0.0
        # ATRÁS de firewall — nunca 0.0.0.0 exposto à internet.
        webhook_host=os.environ.get("WEBHOOK_HOST", "127.0.0.1").strip(),
        webhook_port=int(os.environ.get("WEBHOOK_PORT", "5000")),
        # Se preenchido, o gateway exige o header `X-Webhook-Secret` com esse
        # valor em toda requisição do webhook (configure o mesmo nos headers do
        # webhook da Evolution API). Obrigatório em production (ver acima).
        webhook_secret=webhook_secret,
        # Corpo máximo aceito no POST /webhook (bytes). Payload real da Evolution
        # fica bem abaixo disso; o limite corta floods/DoS triviais.
        webhook_max_body_bytes=int(
            os.environ.get("WEBHOOK_MAX_BODY_BYTES", str(256 * 1024))
        ),
        # Tamanho máximo do texto da mensagem processado (chars). Acima disso a
        # mensagem é ignorada — protege o pipeline/LLM de entradas gigantes.
        message_max_chars=int(os.environ.get("MESSAGE_MAX_CHARS", "4000")),
        # Janela (s) de tolerância do `messageTimestamp` — eventos mais antigos
        # que isso são rejeitados (proteção anti-replay). 0 desliga a checagem.
        replay_max_skew_seconds=int(
            os.environ.get("REPLAY_MAX_SKEW_SECONDS", "300")
        ),
        # OpenAPI/Swagger. Default: ligado só em development.
        docs_enabled=_env_flag("DOCS_ENABLED", default=app_env == "development"),
        # Loga conteúdo de mensagem/resposta (debug). NUNCA em produção.
        log_message_content=_env_flag("LOG_MESSAGE_CONTENT", default=False)
        and not is_prod,
        allowed_numbers=allowed_numbers,
        lid_map={
            k.strip(): v.strip()
            for pair in os.environ.get("LID_MAP", "").split(",")
            if ":" in pair
            for k, v in [pair.split(":", 1)]
            if k.strip() and v.strip()
        },
        allowed_lids=allowed_lids,
        known_names={
            k.strip(): v.strip()
            for pair in os.environ.get("KNOWN_NAMES", "").split(",")
            if ":" in pair
            for k, v in [pair.split(":", 1)]
            if k.strip() and v.strip()
        },
        # Nomes que o assistente usa nos prompts. Vazios → fraseado genérico
        # ("o usuário" / "seu par"). Não afeta KNOWN_NAMES (que é por número).
        owner_name=os.environ.get("OWNER_NAME", "").strip(),
        partner_name=os.environ.get("PARTNER_NAME", "").strip(),
        # Stopwords extras do TF-IDF (csv) — tipicamente nomes próprios que
        # aparecem em quase todo arquivo do vault e não discriminam.
        extra_stopwords=[
            w.strip().lower()
            for w in os.environ.get("EXTRA_STOPWORDS", "").split(",")
            if w.strip()
        ],
        # Número do próprio Yaannk (só dígitos), usado para detectar menção
        # via contato nos grupos. Se vazio, cai no campo `sender` do payload.
        yaannk_number="".join(
            c for c in os.environ.get("YAANNK_NUMBER", "") if c.isdigit()
        ),
        # Contas novas mencionam por LID: `contextInfo.mentionedJid` traz
        # `<lid>@lid` em vez do número. LID do próprio Yaannk (só dígitos).
        yaannk_lid="".join(
            c for c in os.environ.get("YAANNK_LID", "") if c.isdigit()
        ),
        # Provider de LLM usado por padrão enquanto o LLM Router (Fase 6) não
        # existe. Fase 2: só "ollama". Fase 3 acrescenta "kimi".
        llm_default_provider=os.environ.get("LLM_DEFAULT_PROVIDER", "ollama").strip(),
        # Telemetria de LLM (Fase 4): grava uma linha em `llm_calls` por chamada.
        llm_telemetry=os.environ.get("LLM_TELEMETRY", "true").strip().lower()
        not in ("false", "0", "no", ""),
        # LLM Router (Fase 5): escolhe tier/provider do Bloco 3 por
        # intenção+complexidade. Desligado = Bloco 3 vai para o provider default.
        llm_router_enabled=os.environ.get("LLM_ROUTER_ENABLED", "false").strip().lower()
        in ("true", "1", "yes"),
        # Kimi (Moonshot) — API compatível com OpenAI. Fase 3. Tier 1 = default
        # de API; Tier 2 = heavy. Sem KIMI_API_KEY o provider não é registrado
        # e o Yaannk segue só com o Ollama.
        kimi_api_key=os.environ.get("KIMI_API_KEY", "").strip(),
        kimi_base_url=os.environ.get(
            "KIMI_BASE_URL", "https://api.moonshot.cn/v1"
        ).rstrip("/"),
        kimi_model_tier1=os.environ.get("KIMI_MODEL_TIER1", "kimi-k2.6").strip(),
        kimi_model_tier2=os.environ.get("KIMI_MODEL_TIER2", "kimi-k2.7-code").strip(),
        # K2.6 é modelo de raciocínio: gasta tokens de "reasoning" antes do
        # texto. Piso de max_tokens para não devolver resposta vazia (o Router
        # da Fase 5 pode passar valores maiores).
        kimi_min_max_tokens=int(os.environ.get("KIMI_MIN_MAX_TOKENS", "1000")),
        # Preço p/ estimativa de custo — USD por 1M tokens (Kimi via OpenRouter).
        kimi_price_in_per_mtok=float(os.environ.get("KIMI_PRICE_IN_PER_MTOK", "0.95")),
        kimi_price_out_per_mtok=float(os.environ.get("KIMI_PRICE_OUT_PER_MTOK", "4.00")),
        ollama_url=os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
        ollama_desktop_url=os.environ.get("OLLAMA_DESKTOP_URL", "").rstrip("/"),
        ollama_model=os.environ.get("OLLAMA_MODEL", "qwen2.5:3b"),
        vault_path=os.environ["VAULT_PATH"],
        obsidian_mcp_url=os.environ.get(
            "OBSIDIAN_MCP_URL", "http://127.0.0.1:27200/mcp"
        ),
        obsidian_mcp_token=os.environ.get("OBSIDIAN_MCP_TOKEN", ""),
        reranker_model_dir=os.environ.get(
            "RERANKER_MODEL_DIR",
            os.path.expanduser("~/YaannkAgent/models/bge-reranker-v2-m3"),
        ),
        backlink_weight=float(os.environ.get("BACKLINK_WEIGHT", "0.5")),
        # Agentic RAG: depois da 1ª busca, o Kimi julga se o contexto
        # responde a pergunta e, se não, reformula a query e busca de novo.
        # Desligado por default — o baseline do benchmark foi medido sem ele.
        agentic_rag_enabled=_env_flag("AGENTIC_RAG_ENABLED", default=False),
        # Teto de buscas por pergunta, INCLUINDO a inicial (3 = 1 + até 2
        # refinamentos). 1 desliga o loop na prática.
        agentic_rag_max_searches=int(
            os.environ.get("AGENTIC_RAG_MAX_SEARCHES", "3")
        ),
        # Orquestrador (Fase A): Kimi substitui classify_intent() no
        # roteamento de intenção. Desligado = pipeline chama classify_intent()
        # direto, sem instanciar o orquestrador (ver app/services/orchestrator.py).
        orchestrator_enabled=_env_flag("ORCHESTRATOR_ENABLED", default=True),
        # Segundos antes de desistir do Kimi e cair no keyword matching.
        # 3s (default original) causava fallback quase sempre — a telemetria
        # real do Bloco 3 mostra o Kimi via OpenRouter levando 4-18s em boa
        # parte das chamadas (achado Sessão 17). 15s cobre a cauda observada.
        orchestrator_timeout_s=float(os.environ.get("ORCHESTRATOR_TIMEOUT_S", "15")),
    )


settings = _load_settings()
