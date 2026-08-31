import logging

import uvicorn
from fastapi import FastAPI

from app.config import settings
from app.routes.webhook import router as webhook_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

app = FastAPI(title="Yaannk Gateway")
app.include_router(webhook_router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


def run() -> None:
    uvicorn.run("app.main:app", host="0.0.0.0", port=settings.webhook_port)


if __name__ == "__main__":
    run()
