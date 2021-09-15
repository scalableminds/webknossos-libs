import os
from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from rich.prompt import Prompt

from .defaults import DEFAULT_WEBKNOSSOS_URL
from .generated import Client as GeneratedClient

load_dotenv()


@lru_cache(maxsize=None)
def _ask_for_token(webknossos_url: str) -> str:
    # TODO  pylint: disable=fixme
    # -validate token and ask again if necessary
    # -ask if the token should be saved in some .env file
    return Prompt.ask(
        f"\nPlease enter your webknossos token as shown on {webknossos_url}/auth/token ",
        password=True,
    )


@lru_cache(maxsize=None)
def _get_generated_client(
    webknossos_url: str = DEFAULT_WEBKNOSSOS_URL,
    *,
    token: Optional[str] = None,
    enforce_token: bool = False,
) -> GeneratedClient:
    """Generates a client which might contain an x-auth-token header.
    The token is taken from one of the following sources, using the first one matching:
    - function argument
    - environment variable
    - user prompt (only if enforce_token is set)
    """
    if token is None and "WK_TOKEN" in os.environ:
        token = os.environ["WK_TOKEN"]
    if token is None and enforce_token:
        token = _ask_for_token(webknossos_url)

    if token is None:
        return GeneratedClient(base_url=webknossos_url)
    else:
        return GeneratedClient(
            base_url=webknossos_url,
            headers={"X-Auth-Token": token},
        )
