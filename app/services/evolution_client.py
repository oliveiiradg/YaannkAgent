import httpx

from app.config import settings


async def send_message(number: str, text: str) -> None:
    url = f"{settings.evolution_api_url}/message/sendText/{settings.evolution_instance_name}"
    headers = {
        "Content-Type": "application/json",
        "apikey": settings.evolution_api_key,
    }
    body = {"number": number, "text": text}

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(url, json=body, headers=headers)
        response.raise_for_status()
