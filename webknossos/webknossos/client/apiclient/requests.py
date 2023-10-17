import httpx


def get_json(uri: str) -> dict:
    # todo set mimetype
    response = httpx.get(uri)
    return response.json()
