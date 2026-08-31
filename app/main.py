import logging

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.routes.webhook import router as webhook_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

# OpenAPI/Swagger só quando explicitamente habilitado (default: dev sim,
# produção não) — não expõe o mapa de rotas/schema num deploy público.
_docs = settings.docs_enabled
app = FastAPI(
    title="Yaannk Gateway",
    docs_url="/docs" if _docs else None,
    redoc_url="/redoc" if _docs else None,
    openapi_url="/openapi.json" if _docs else None,
)


@app.middleware("http")
async def _webhook_guard(request: Request, call_next):
    """Barreira antes do parsing do corpo no POST /webhook:
    1) corta payloads acima do limite (floods/DoS trivial);
    2) exige o segredo do webhook (comparação em tempo constante).
    A resposta de auth é genérica de propósito — não revela se o segredo
    estava ausente ou incorreto."""
    if request.url.path == "/webhook" and request.method == "POST":
        import hmac

        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > settings.webhook_max_body_bytes:
            return JSONResponse(status_code=413, content={"detail": "payload too large"})

        if settings.webhook_secret:
            got = request.headers.get("x-webhook-secret") or ""
            if not hmac.compare_digest(got, settings.webhook_secret):
                client = request.client.host if request.client else "?"
                logger.warning("Webhook rejeitado: autenticação inválida (de %s)", client)
                return JSONResponse(status_code=401, content={"detail": "unauthorized"})

    return await call_next(request)


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
    """Nunca devolve stack trace/detalhes internos ao cliente."""
    logger.exception("Erro não tratado em %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "internal error"})


app.include_router(webhook_router)


@app.get("/health")
async def health() -> dict:
    # Deliberadamente sem versão, uptime, config ou estado de dependências.
    return {"status": "ok"}


def run() -> None:
    uvicorn.run(
        "app.main:app", host=settings.webhook_host, port=settings.webhook_port
    )


if __name__ == "__main__":
    run()
