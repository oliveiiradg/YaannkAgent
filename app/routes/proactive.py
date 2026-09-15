"""Endpoints do módulo proativo (Fase E/V4) — lembretes, confirmação de
pagamento e criação da nota do mês seguinte.

Toda rota exige o header `x-proactive-token` (ver `_exige_token`). Sem LLM,
sem RAG: lógica 100% determinística sobre
`app/services/bills.py` (mesma fonte — a nota `Contas - AAAA-MM.md` do vault).
`bills-due`/`bills-message`/`create-next-month` são pensados pra ser chamados
por um agendador externo (cron, n8n) — quem manda a mensagem pro grupo do
WhatsApp é o n8n, não o gateway (nenhum desses endpoints chama a Evolution
API; devolvem `message` pronta pra o workflow repassar). `bills.mark_bill_paid()`
é chamado tanto por esse endpoint quanto pelo ReAct do Agent Vida (D-10) —
sem round-trip HTTP — quando o usuário confirma pagamento pelo WhatsApp.

`bia-agenda-today`/`bia-agenda-today-message` (V5) seguem o mesmo contrato:
o gateway só devolve dado/mensagem pronta, quem envia pro WhatsApp privado
da Bia (BIA_JID) é o n8n.
"""

import datetime
import hmac
import logging

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel

from app.config import settings
from app.services.agenda_bia import format_agenda_dia_message, get_bia_agenda_today
from app.services.bills import (
    create_next_month_note,
    format_bills_due_message,
    get_bills_due,
    mark_bill_paid,
)

logger = logging.getLogger(__name__)


def _exige_token(x_proactive_token: str = Header(default="")) -> None:
    """Mesmo esquema do segredo do webhook: token fixo no header, comparado em
    tempo constante. Sem PROACTIVE_TOKEN configurado (dev/test) as rotas ficam
    abertas; em production a config não sobe sem ele."""
    if settings.proactive_token and not hmac.compare_digest(
        x_proactive_token, settings.proactive_token
    ):
        logger.warning("proactive: requisição recusada (token ausente ou inválido)")
        raise HTTPException(status_code=401, detail="unauthorized")


router = APIRouter(prefix="/proactive", tags=["proactive"], dependencies=[Depends(_exige_token)])


class MarkPaidRequest(BaseModel):
    bill_name: str
    month: str  # "AAAA-MM"


@router.get("/bills-due")
async def bills_due(days_ahead: int = Query(default=3, ge=0, le=90)) -> dict:
    bills = get_bills_due(days_ahead)
    return {"has_due_bills": bool(bills), "bills": bills}


@router.get("/bills-message")
async def bills_message(days_ahead: int = Query(default=3, ge=0, le=90)) -> dict:
    bills = get_bills_due(days_ahead)
    return {"message": format_bills_due_message(bills)}


@router.post("/mark-paid")
async def mark_paid(body: MarkPaidRequest) -> dict:
    return await mark_bill_paid(body.bill_name, body.month)


@router.post("/create-next-month")
async def create_next_month(force: bool = Query(default=False)) -> dict:
    hoje = datetime.date.today()
    resultado = await create_next_month_note(hoje.year, hoje.month, force=force)
    return {
        "success": resultado["success"],
        "month": resultado.get("month"),
        "message": resultado["message"],
    }


@router.get("/bia-agenda-today")
async def bia_agenda_today() -> dict:
    appointments = get_bia_agenda_today()
    return {"has_appointments": bool(appointments), "appointments": appointments}


@router.get("/bia-agenda-today-message")
async def bia_agenda_today_message() -> dict:
    appointments = get_bia_agenda_today()
    # `has_appointments` aqui também: o subworkflow yaannk_agenda_bia decide
    # se envia com uma chamada só, sem precisar de /bia-agenda-today antes.
    return {
        "has_appointments": bool(appointments),
        "message": format_agenda_dia_message(appointments, datetime.date.today()),
    }
