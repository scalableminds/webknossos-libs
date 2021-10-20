import os
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Iterator, Optional

import attr
import httpx
from dotenv import load_dotenv
from rich.prompt import Prompt

from webknossos.client.defaults import DEFAULT_WEBKNOSSOS_URL
from webknossos.client.generated import Client as GeneratedClient

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
def _cached_get_generated_client(
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
class WebknossosContext:
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
        return _cached_get_generated_client(self.url, self.token)

    @property
    def generated_auth_client(self) -> GeneratedClient:
        return _cached_get_generated_client(self.url, self.required_token)


_webknossos_context_var: ContextVar[WebknossosContext] = ContextVar(
    "_webknossos_context_var", default=WebknossosContext()
)


@contextmanager
def webknossos_context(
    url: str = DEFAULT_WEBKNOSSOS_URL,
    token: Optional[str] = None,
    organization: Optional[str] = None,
) -> Iterator[None]:
    context_var_token = _webknossos_context_var.set(
        WebknossosContext(url, token, organization)
    )
    try:
        yield
    finally:
        _webknossos_context_var.reset(context_var_token)


def get_context() -> WebknossosContext:
    return _webknossos_context_var.get()


def get_generated_client(enforce_auth: bool = False) -> GeneratedClient:
    if enforce_auth:
        return get_context().generated_auth_client
    else:
        return get_context().generated_client
