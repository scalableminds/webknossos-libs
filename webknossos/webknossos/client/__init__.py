import os
from functools import lru_cache

from dotenv import load_dotenv
from rich.prompt import Prompt

from .defaults import DEFAULT_WEBKNOSSOS_URL
from .generated import Client as GeneratedClient

load_dotenv()


@lru_cache(maxsize=None)
def _get_token(webknossos_url: str) -> str:
    if "WK_TOKEN" in os.environ:
        return os.environ["WK_TOKEN"]
    else:
        # TODO  pylint: disable=fixme
        # -validate token and ask again if necessary
        # -ask if the token should be saved in some .env file
        return Prompt.ask(
            f"\nPlease enter your webknossos token as shown on {webknossos_url}/auth/token ",
            password=True,
        )


@lru_cache(maxsize=None)
def _get_generated_client(
    webknossos_url: str = DEFAULT_WEBKNOSSOS_URL, with_token: bool = False
) -> GeneratedClient:
    if with_token:
        return GeneratedClient(
            base_url=webknossos_url,
            headers={"X-Auth-Token": _get_token(webknossos_url)},
        )
    else:
        return GeneratedClient(base_url=webknossos_url)
