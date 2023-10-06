import httpx


def get_json(uri: str) -> dict:
    response = httpx.get(uri)
    return {}