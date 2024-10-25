"""
# Authentication & Server Context

When interacting with a WEBKNOSSOS server, you might need to
specify your user token to authenticate yourself.
You can copy your token from

[https://webknossos.org/auth/token](https://webknossos.org/auth/token)

Using the same methods, you can also specify the webknossos-server if you
are not using the default [webknossos.org](https://webknossos.org) instance,
as well as a timeout for network requests (default is 30 minutes).

There are the following four options to specify which server context to use:

1. Specifying the context in code using the `webknossos_context`
   contextmanager in a `with` statement:

    ```python
    with webknossos_context(token="my_webknossos_token"):
       # code that interacts with webknossos
    ```

    For more information about the `with` statement and contextmanagers,
    please see [this tutorial](https://realpython.com/python-with-statement).

2. You may specify your settings as environment variables `WK_TOKEN` and `WK_URL`:

    ```shell
    WK_TOKEN="my_webknossos_token" python my-script.py
    ```
3. You can also specify those environment variables in a `.env` file
   in your working directory.
   Environment variables set in the command line take precedence.

    ```shell
    # content of .env
    WK_TOKEN="my_webknossos_token"
    WK_URL="…"
    WK_TIMEOUT="3600"  # in seconds
    ```

4. If nothing else is specified and authentication is needed,
   you are asked interactively for a token, which is used for
   subsequent interactions in the same python run as well.
"""

import os
from contextlib import ContextDecorator
from contextvars import ContextVar, Token
from functools import lru_cache
from typing import Any, List, Optional

import attr
from dotenv import load_dotenv
from rich.prompt import Prompt

from ._defaults import DEFAULT_HTTP_TIMEOUT, DEFAULT_WEBKNOSSOS_URL
from .api_client import DatastoreApiClient, WkApiClient

load_dotenv()


@lru_cache(maxsize=None)
def _cached_ask_for_token(webknossos_url: str) -> str:
    # TODO # noqa: FIX002 Line contains TODO
    # -validate token and ask again if necessary
    # -ask if the token should be saved in some .env file
    # -reset invalid tokens
    #  (e.g. use cachetools for explicit cache management:
    #  https://cachetools.readthedocs.io/en/stable/#memoizing-decorators)
    return Prompt.ask(
        f"\nPlease enter your webknossos token as shown on {webknossos_url}/auth/token ",
        password=True,
    )


@lru_cache(maxsize=None)
def _cached_get_org(context: "_WebknossosContext") -> str:
    current_api_user = context.api_client_with_auth.user_current()
    return current_api_user.organization


# TODO reset invalid tokens e.g. using cachetools # noqa: FIX002 Line contains TODO
@lru_cache(maxsize=None)
def _cached_get_datastore_token(context: "_WebknossosContext") -> str:
    api_datastore_token = context.api_client_with_auth.token_generate_for_data_store()
    return api_datastore_token.token


@lru_cache(maxsize=None)
def _cached__get_api_client(
    webknossos_url: str,
    token: Optional[str],
    timeout: int,
) -> WkApiClient:
    """Generates a client which might contain an x-auth-token header."""
    if token is None:
        return WkApiClient(base_wk_url=webknossos_url, timeout_seconds=timeout)
    return WkApiClient(
        base_wk_url=webknossos_url,
        headers={"X-Auth-Token": token},
        timeout_seconds=timeout,
    )


def _clear_all_context_caches() -> None:
    _cached_ask_for_token.cache_clear()
    _cached_get_org.cache_clear()
    _cached_get_datastore_token.cache_clear()
    _cached__get_api_client.cache_clear()


@attr.frozen
class _WebknossosContext:
    url: str = os.environ.get("WK_URL", default=DEFAULT_WEBKNOSSOS_URL).rstrip("/")
    token: Optional[str] = os.environ.get("WK_TOKEN", default=None)
    timeout: int = int(os.environ.get("WK_TIMEOUT", default=DEFAULT_HTTP_TIMEOUT))

    # all properties are cached outside to allow re-usability
    # if same context is instantiated twice
    @property
    def required_token(self) -> str:
        if self.token is None:
            token = _cached_ask_for_token(self.url)
            # We replace the current context, but leave all previous ones as-is.
            # Any opened contextmanagers will still close correctly, as the stored
            # tokens still point to the correct predecessors.
            _webknossos_context_var.set(
                _WebknossosContext(self.url, token, self.timeout)
            )
            return token
        else:
            return self.token

    @property
    def organization_id(self) -> str:
        return _cached_get_org(self)

    @property
    def datastore_token(self) -> Optional[str]:
        if self.token is None:
            return None
        else:
            return _cached_get_datastore_token(self)

    @property
    def datastore_required_token(self) -> str:
        return _cached_get_datastore_token(self)

    @property
    def api_client(self) -> WkApiClient:
        return _cached__get_api_client(self.url, self.token, self.timeout)

    @property
    def api_client_with_auth(self) -> WkApiClient:
        return _cached__get_api_client(self.url, self.required_token, self.timeout)

    def get_datastore_api_client(self, datastore_url: str) -> DatastoreApiClient:
        return DatastoreApiClient(
            datastore_base_url=datastore_url, timeout_seconds=self.timeout
        )


_webknossos_context_var: ContextVar[_WebknossosContext] = ContextVar(
    "_webknossos_context_var", default=_WebknossosContext()
)


def _get_context() -> _WebknossosContext:
    return _webknossos_context_var.get()


class webknossos_context(ContextDecorator):
    """"""

    def __init__(
        self,
        url: Optional[str] = None,
        token: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """Creates a new WEBKNOSSOS server context manager.

        Can be used as a context manager with 'with' or as a decorator.

        Args:
            url: Base URL for WEBKNOSSOS server, defaults to https://webknossos.org.
                Taken from previous context if not specified.
            token: Authentication token from https://webknossos.org/auth/token.
                Must be specified explicitly.
            timeout: Network request timeout in seconds, defaults to 1800 (30 min).
                Taken from previous context if not specified.

        Examples:
            Using as context manager:
                ```
                with webknossos_context(token="my_webknossos_token"):
                    # code that interacts with webknossos
                    ds.download(...)
                ```

            Using as decorator:
                ```
                @webknossos_context(token="my_webknossos_token")
                def my_func():
                    # code that interacts with webknossos
                    ...
                ```

        Note:
            The url and timeout parameters will use values from the previous context
            (e.g. environment variables) if not specified explicitly. The token
            parameter must always be set explicitly.
        """
        self._url = _get_context().url if url is None else url.rstrip("/")
        self._token = token
        self._timeout = _get_context().timeout if timeout is None else timeout
        self._context_var_token_stack: List[Token[_WebknossosContext]] = []

    def __enter__(self) -> None:
        context_var_token = _webknossos_context_var.set(
            _WebknossosContext(self._url, self._token, self._timeout)
        )
        self._context_var_token_stack.append(context_var_token)

    def __exit__(self, *exc: Any) -> None:
        _webknossos_context_var.reset(self._context_var_token_stack.pop())


def _get_api_client(enforce_auth: bool = False) -> WkApiClient:
    if enforce_auth:
        return _get_context().api_client_with_auth
    else:
        return _get_context().api_client
