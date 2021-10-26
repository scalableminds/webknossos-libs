"""
# Authentication & Server Context

When interacting with a webKnossos server, you might need to
specify your user token to authenticate yourself.
You can copy your token from

[https://webknossos.org/auth/token](https://webknossos.org/auth/token)

Using the same methods, you can also specify which organization
you want to use by default if you are member of multiple ones,
or specify the webknossos-server if your are not using
the default [webknossos.org](https://webknossos.org) instance.

There are the following four options to specify which server context to use:

1. Specifying the context in code using the `webknossos_context`
   contextmanager in a `with` statement:

    ```python
    with webknossos_context(token="my_webknossos_token"):
       # code that interacts with webknossos
    ```

    For more information about the `with` statement and contextmanagers,
    please see [this tutorial](https://realpython.com/python-with-statement).

2. You may specify your settings as environment variables `WK_TOKEN`, `WK_ORG`, `WK_URL`:

    ```shell
    WK_TOKEN="my_webknossos_token" python my-script.py
    ```
3. You can also specify those environment variables in a `.env` file
   in your working directory.
   Environment variables set in the command line take precedence.

    ```shell
    # content of .env
    WK_TOKEN="my_webknossos_token"
    WK_ORG="…"
    WK_URL="…"
    ```

4. If nothing else is specified and authentication is needed,
   you are asked interactively for a token, which is used for
   subsequent interactions in the same python run as well.
"""


import os
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Iterator, Optional

import attr
import httpx
from dotenv import load_dotenv
from rich.prompt import Prompt

from webknossos.client._defaults import DEFAULT_WEBKNOSSOS_URL
from webknossos.client._generated import Client as GeneratedClient

load_dotenv()


@lru_cache(maxsize=None)
def _cached_ask_for_token(webknossos_url: str) -> str:
    # TODO  pylint: disable=fixme
    # -validate token and ask again if necessary
    # -ask if the token should be saved in some .env file
    # -reset invalid tokens
    #  (e.g. use cachetools for explicit cache managment:
    #  https://cachetools.readthedocs.io/en/stable/#memoizing-decorators)
    return Prompt.ask(
        f"\nPlease enter your webknossos token as shown on {webknossos_url}/auth/token ",
        password=True,
    )


@lru_cache(maxsize=None)
def _cached_get_default_org(webknossos_url: str, token: str) -> str:
    response = httpx.get(
        f"{webknossos_url}/api/organizations/default",
        headers={"X-Auth-Token": token},
    )
    response.raise_for_status()
    default_organization = response.json()
    assert default_organization is not None
    return default_organization["name"]


# TODO reset invalid tokens e.g. using cachetools  pylint: disable=fixme
@lru_cache(maxsize=None)
def _cached_get_datastore_token(webknossos_url: str, token: str) -> str:
    response = httpx.post(
        f"{webknossos_url}/api/userToken/generate",
        headers={"X-Auth-Token": token},
    )
    response.raise_for_status()
    return response.json()["token"]


@lru_cache(maxsize=None)
def _cached__get_generated_client(
    webknossos_url: str,
    token: Optional[str],
) -> GeneratedClient:
    """Generates a client which might contain an x-auth-token header."""
    if token is None:
        return GeneratedClient(base_url=webknossos_url)
    else:
        return GeneratedClient(
            base_url=webknossos_url,
            headers={"X-Auth-Token": token},
        )


@attr.frozen
class _WebknossosContext:
    url: str = os.environ.get("WK_URL", default=DEFAULT_WEBKNOSSOS_URL)
    token: Optional[str] = os.environ.get("WK_TOKEN", default=None)
    _organization: Optional[str] = os.environ.get("WK_ORG", default=None)

    # all properties are cached outside to allow re-usability
    # if same context is instantiated twice
    @property
    def required_token(self) -> str:
        if self.token is None:
            return _cached_ask_for_token(self.url)
        else:
            return self.token

    @property
    def organization(self) -> str:
        if self._organization is None:
            return _cached_get_default_org(self.url, self.required_token)
        else:
            return self._organization

    @property
    def datastore_token(self) -> str:
        return _cached_get_datastore_token(self.url, self.required_token)

    @property
    def generated_client(self) -> GeneratedClient:
        return _cached__get_generated_client(self.url, self.token)

    @property
    def generated_auth_client(self) -> GeneratedClient:
        return _cached__get_generated_client(self.url, self.required_token)


_webknossos_context_var: ContextVar[_WebknossosContext] = ContextVar(
    "_webknossos_context_var", default=_WebknossosContext()
)


@contextmanager
def webknossos_context(
    url: str = DEFAULT_WEBKNOSSOS_URL,
    token: Optional[str] = None,
    organization: Optional[str] = None,
) -> Iterator[None]:
    context_var_token = _webknossos_context_var.set(
        _WebknossosContext(url, token, organization)
    )
    try:
        yield
    finally:
        _webknossos_context_var.reset(context_var_token)


def _get_context() -> _WebknossosContext:
    return _webknossos_context_var.get()


def _get_generated_client(enforce_auth: bool = False) -> GeneratedClient:
    if enforce_auth:
        return _get_context().generated_auth_client
    else:
        return _get_context().generated_client
