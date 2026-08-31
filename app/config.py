import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    evolution_api_url: str
    evolution_api_key: str
    evolution_instance_name: str
    webhook_port: int
    webhook_secret: str
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


def _load_settings() -> Settings:
    return Settings(
        evolution_api_url=os.environ["EVOLUTION_API_URL"].rstrip("/"),
        evolution_api_key=os.environ["EVOLUTION_API_KEY"],
        evolution_instance_name=os.environ["EVOLUTION_INSTANCE_NAME"],
        webhook_port=int(os.environ.get("WEBHOOK_PORT", "5000")),
        # Se preenchido, o gateway exige o header `X-Webhook-Secret` com esse
        # valor em toda requisição do webhook (configure o mesmo nos headers do
        # webhook da Evolution API). Vazio = sem autenticação (retrocompatível).
        webhook_secret=os.environ.get("WEBHOOK_SECRET", "").strip(),
        allowed_numbers=[
            n.strip()
            for n in os.environ.get("ALLOWED_NUMBERS", "").split(",")
            if n.strip()
        ],
        lid_map={
            k.strip(): v.strip()
            for pair in os.environ.get("LID_MAP", "").split(",")
            if ":" in pair
            for k, v in [pair.split(":", 1)]
            if k.strip() and v.strip()
        },
        allowed_lids=[
            l.strip()
            for l in os.environ.get("ALLOWED_LIDS", "").split(",")
            if l.strip()
        ],
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
    )


settings = _load_settings()
