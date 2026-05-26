"""
# Authentication & Server Context

When interacting with a WEBKNOSSOS server, you might need to
specify your user token to authenticate yourself.
You can copy your token from

[https://webknossos.org/account/token](https://webknossos.org/account/token)

Using the same methods, you can also specify the webknossos-server if you
are not using the default [webknossos.org](https://webknossos.org) instance,
as well as a timeout for network requests (default is 30 minutes).

There are the following four options to specify which server context to use:

1. Calling `login` to set credentials for the whole session:

    ```python
    import webknossos as wk
    wk.login(token="my_webknossos_token")
    # all subsequent interactions use this token
    ```

2. Specifying the context in code using the `webknossos_context`
   contextmanager in a `with` statement (useful for scoped overrides):

    ```python
    with webknossos_context(token="my_webknossos_token"):
       # code that interacts with webknossos
    ```

    For more information about the `with` statement and contextmanagers,
    please see [this tutorial](https://realpython.com/python-with-statement).

3. You may specify your settings as environment variables `WK_TOKEN` and `WK_URL`:

    ```shell
    WK_TOKEN="my_webknossos_token" python my-script.py
    ```
4. You can also specify those environment variables in a `.env` file
   in your working directory.
   Environment variables set in the command line take precedence.

    ```shell
    # content of .env
    WK_TOKEN="my_webknossos_token"
    WK_URL="…"
    WK_TIMEOUT="3600"  # in seconds
    ```

5. If nothing else is specified and authentication is needed,
   you are asked interactively for a token, which is used for
   subsequent interactions in the same python run as well.
"""

import os
from contextlib import ContextDecorator
from contextvars import ContextVar, Token
from dataclasses import dataclass
from functools import cache, cached_property
from typing import Any

import httpx
from dotenv import load_dotenv

from ..ssl_context import SSL_CONTEXT
from ._defaults import DEFAULT_HTTP_TIMEOUT, DEFAULT_WEBKNOSSOS_URL
from .api_client import (
    DatastoreApiClient,
    DatastoreApiClientV13,
    TracingStoreApiClient,
    WkApiClient,
    WkApiClientV13,
)

load_dotenv()


@cache
def _cached_detect_api_version(wk_url: str, timeout: int) -> int:
    """Queries /api/buildinfo to determine the server's current API version."""
    response = httpx.get(f"{wk_url}/api/buildinfo", timeout=timeout, verify=SSL_CONTEXT)
    data = response.json()
    current = data.get("httpApiVersioning", {}).get("currentApiVersion")
    if current is None or not isinstance(current, int):
        raise RuntimeError("Could not determine current API version.")
    if current not in (13, 14):
        raise RuntimeError(f"Unsupported API version: {current}")
    return current


def _clear_all_context_caches() -> None:
    _cached_detect_api_version.cache_clear()
    del _get_context().organization_id
    del _get_context().api_client


@dataclass(kw_only=True)
class _WebknossosContext:
    url: str = os.environ.get("WK_URL", default=DEFAULT_WEBKNOSSOS_URL).rstrip("/")
    token: str | None = os.environ.get("WK_TOKEN", default=None)
    timeout: int = int(os.environ.get("WK_TIMEOUT", default=DEFAULT_HTTP_TIMEOUT))
    _api_version: int | None = None

    # all properties are cached outside to allow re-usability
    # if same context is instantiated twice

    @property
    def api_version(self) -> int:
        if self._api_version is None:
            self._api_version = _cached_detect_api_version(self.url, self.timeout)
        return self._api_version

    @cached_property
    def organization_id(self) -> str:
        return self.api_client.user_current().organization

    @cached_property
    def api_client(self) -> WkApiClient:
        cls = WkApiClientV13 if self.api_version == 13 else WkApiClient
        return cls(
            base_wk_url=self.url,
            headers={} if self.token is None else {"X-Auth-Token": self.token},
            timeout_seconds=self.timeout,
        )

    def get_datastore_api_client(self, datastore_url: str) -> DatastoreApiClient:
        cls = DatastoreApiClientV13 if self.api_version == 13 else DatastoreApiClient
        return cls(
            datastore_base_url=datastore_url,
            headers={} if self.token is None else {"X-Auth-Token": self.token},
            timeout_seconds=self.timeout,
        )

    def get_tracingstore_api_client(self) -> TracingStoreApiClient:
        api_tracingstore = self.api_client.tracing_store()
        return TracingStoreApiClient(
            base_url=api_tracingstore.url,
            headers={} if self.token is None else {"X-Auth-Token": self.token},
            timeout_seconds=self.timeout,
        )


_webknossos_context_var: ContextVar[_WebknossosContext] = ContextVar(
    "_webknossos_context_var", default=_WebknossosContext()
)


def _get_context() -> _WebknossosContext:
    return _webknossos_context_var.get()


def login(
    *,
    url: str | None = None,
    token: str | None = None,
    timeout: int | None = None,
    api_version: int | None = None,
) -> None:
    """Log into a WEBKNOSSOS server, changing the global context for the entire session.

    Unlike `webknossos_context`, this is not scoped to a `with` block — the change
    persists for the remainder of the Python session (or until `login` is called again).

    Args:
        url: Base URL for the WEBKNOSSOS server. Defaults to the current context URL
            (https://webknossos.org if not previously set).
        token: Authentication token from https://webknossos.org/account/token.
            If not provided, you will be prompted interactively.
        timeout: Network request timeout in seconds. Defaults to the current context timeout.
        api_version: WEBKNOSSOS API version to use (13 or 14). Defaults to the current
            context version (14 if not previously set).

    Example:
        ```python
        import webknossos as wk
        wk.login(token="my_token")
        ds = wk.RemoteDataset.open("my_dataset")
        ```
    """
    current = _get_context()
    resolved_url = current.url if url is None else url.rstrip("/")
    resolved_timeout = current.timeout if timeout is None else timeout
    resolved_api_version = current.api_version if api_version is None else api_version
    if token is None:
        token = input(
            f"Please enter your WEBKNOSSOS token for {resolved_url}\n"
            f"(You can find it at {resolved_url}/account/token): "
        )
    _webknossos_context_var.set(
        _WebknossosContext(
            url=resolved_url,
            token=token,
            timeout=resolved_timeout,
            _api_version=resolved_api_version,
        )
    )


class webknossos_context(ContextDecorator):
    """"""

    def __init__(
        self,
        url: str | None = None,
        token: str | None = None,
        timeout: int | None = None,
        api_version: int | None = None,
    ) -> None:
        """Creates a new WEBKNOSSOS server context manager.

        Can be used as a context manager with 'with' or as a decorator.

        Args:
            url: Base URL for WEBKNOSSOS server, defaults to https://webknossos.org.
                Taken from previous context if not specified.
            token: Authentication token from https://webknossos.org/account/token.
                Must be specified explicitly.
            timeout: Network request timeout in seconds, defaults to 1800 (30 min).
                Taken from previous context if not specified.
            api_version: WEBKNOSSOS API version to use (13 or 14). Defaults to the
                previous context version (14 if not previously set).

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
        current = _get_context()
        self._url = current.url if url is None else url.rstrip("/")
        self._token = token
        self._timeout = current.timeout if timeout is None else timeout
        self._api_version = current.api_version if api_version is None else api_version
        self._context_var_token_stack: list[Token[_WebknossosContext]] = []

    def __enter__(self) -> None:
        context_var_token = _webknossos_context_var.set(
            _WebknossosContext(
                url=self._url,
                token=self._token,
                timeout=self._timeout,
                _api_version=self._api_version,
            )
        )
        self._context_var_token_stack.append(context_var_token)

    def __exit__(self, *exc: Any) -> None:
        _webknossos_context_var.reset(self._context_var_token_stack.pop())


def _get_api_client() -> WkApiClient:
    return _get_context().api_client
