"""Read-only fsspec filesystem for Globus collections.

Paths have the form `globus://<collection_id>/<absolute path on the collection>`.
Directory listings and metadata come from the Globus Transfer API, file contents
are read via ranged HTTPS requests from the collection's HTTPS server.

Requires the `globus` extra (`pip install webknossos[globus]`).

Credentials are passed explicitly via storage options, never read from global state:

- `transfer_authorizer`: Used for the Transfer API. Needs the scope
  `urn:globus:auth:scope:transfer.api.globus.org:all`. For mapped collections on
  Globus Connect Server v5, the scope additionally needs the collection's
  `data_access` dependency, see `transfer_scope(collection_id)`.
- `https_authorizer`: Used for reading files. Needs the scope
  `https://auth.globus.org/scopes/<collection_id>/https`, see `https_scope(collection_id)`.

Both accept a `globus_sdk` authorizer (e.g. `RefreshTokenAuthorizer` or
`ClientCredentialsAuthorizer`, so that tokens are refreshed during long copies)
or a callable that returns an access token.

```python
import globus_sdk
from upath import UPath
import webknossos as wk

path = UPath(
    "globus://<collection_id>/path/to/dataset",
    transfer_authorizer=globus_sdk.RefreshTokenAuthorizer(...),
    https_authorizer=globus_sdk.RefreshTokenAuthorizer(...),
)
ds = wk.Dataset.open(path)
```

Note that globus_sdk authorizers can not be pickled. When paths need to be sent to
other processes, pass picklable callables that return access tokens instead.
"""

import logging
import threading
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, NoReturn

from fsspec.spec import AbstractBufferedFile, AbstractFileSystem
from fsspec.utils import stringify_path
from upath.implementations.cloud import CloudPath

if TYPE_CHECKING:
    import globus_sdk
    import requests

logger = logging.getLogger(__name__)

PROTOCOL = "globus"

_TRANSFER_SCOPE = "urn:globus:auth:scope:transfer.api.globus.org:all"
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
# Globus Connect Server reports failing storage backends as 404 with one of these in the body.
_BACKEND_FAULT_MARKERS = ("ENDPOINT_ERROR", "GCS Manager Internal Error")


def transfer_scope(collection_id: str) -> str:
    """Returns the Transfer API scope including the `data_access` dependency of the collection,
    which is required for mapped collections on Globus Connect Server v5."""
    return f"{_TRANSFER_SCOPE}[*https://auth.globus.org/scopes/{collection_id}/data_access]"


def https_scope(collection_id: str) -> str:
    """Returns the scope that is required for reading files from the collection via HTTPS."""
    return f"https://auth.globus.org/scopes/{collection_id}/https"


class GlobusPermissionError(PermissionError):
    pass


def _read_only(*_args: Any, **_kwargs: Any) -> NoReturn:
    raise NotImplementedError("globus:// paths are read-only.")


def _authorization_header(token_source: Any) -> str | None:
    if token_source is None:
        return None
    if callable(token_source):
        return f"Bearer {token_source()}"
    return token_source.get_authorization_header()


def _split_path(path: str) -> tuple[str, str]:
    collection_id, _, sub_path = path.partition("/")
    return collection_id, "/" + sub_path


def _parse_mtime(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _make_transfer_client(token_source: Any) -> "globus_sdk.TransferClient":
    import globus_sdk
    from globus_sdk.authorizers import GlobusAuthorizer

    authorizer: GlobusAuthorizer | None
    if token_source is None or isinstance(token_source, GlobusAuthorizer):
        authorizer = token_source
    else:

        class _CallableAuthorizer(GlobusAuthorizer):
            def get_authorization_header(self) -> str:
                return f"Bearer {token_source()}"

        authorizer = _CallableAuthorizer()
    return globus_sdk.TransferClient(authorizer=authorizer)


class GlobusFileSystem(AbstractFileSystem):
    """Read-only fsspec filesystem for Globus collections.

    Args:
        transfer_authorizer: globus_sdk authorizer or token callable for the Transfer API.
        https_authorizer: globus_sdk authorizer or token callable for the collection's HTTPS scope.
        transfer_client: Use this `globus_sdk.TransferClient` instead of creating one from `transfer_authorizer`.
        https_server: Base URL of the collection's HTTPS server. Looked up via the Transfer API if not set.
        local_user: Local user to map to on GCS v5 mapped collections.
        retries: Number of retries for transient HTTPS errors.
        ls_page_size: Number of entries per Transfer API listing request.
    """

    protocol = PROTOCOL
    root_marker = ""

    def __init__(
        self,
        *,
        transfer_authorizer: Any = None,
        https_authorizer: Any = None,
        transfer_client: "globus_sdk.TransferClient | None" = None,
        https_server: str | None = None,
        local_user: str | None = None,
        retries: int = 5,
        ls_page_size: int = 100_000,
        timeout: float = 60,
        **kwargs: Any,
    ) -> None:
        try:
            import globus_sdk  # noqa: F401
        except ImportError as e:
            from webknossos.utils import WkImportError

            raise WkImportError("globus-sdk", extras="globus") from e

        super().__init__(**kwargs)
        self._transfer_client = transfer_client or _make_transfer_client(
            transfer_authorizer
        )
        self._https_authorizer = https_authorizer
        self._https_server_override = https_server
        self._https_servers: dict[str, str] = {}
        self._local_user = local_user
        self._retries = retries
        self._ls_page_size = ls_page_size
        self._timeout = timeout
        self._lock = threading.Lock()
        self._thread_local = threading.local()

    @classmethod
    def _strip_protocol(cls, path: Any) -> Any:
        if isinstance(path, list):
            return [cls._strip_protocol(p) for p in path]
        path = stringify_path(path)
        if path.startswith(f"{PROTOCOL}://"):
            path = path[len(f"{PROTOCOL}://") :]
        return path.rstrip("/")

    # Transfer API

    def _transfer_error(
        self, error: "globus_sdk.TransferAPIError", path: str
    ) -> Exception:
        collection_id, sub_path = _split_path(path)
        code = error.code or ""
        if "NotDirectory" in code:
            return NotADirectoryError(path)
        if error.http_status == 404 or code.endswith("NotFound"):
            return FileNotFoundError(path)
        if error.http_status in (401, 403) or code in (
            "ConsentRequired",
            "PermissionDenied",
            "EndpointPermissionDenied",
            "AuthenticationFailed",
        ):
            return GlobusPermissionError(
                f"No access to {sub_path} on Globus collection {collection_id} "
                f"via the Transfer API ({error.http_status} {code}: {error.message}). "
                f"Make sure the transfer token has the scope {_TRANSFER_SCOPE} and, "
                f"for mapped collections on Globus Connect Server v5, its data_access "
                f"dependency: {transfer_scope(collection_id)}."
            )
        return error

    def _list_dir(self, path: str) -> list[dict[str, Any]]:
        import globus_sdk

        collection_id, sub_path = _split_path(path)
        kwargs: dict[str, Any] = {}
        if self._local_user is not None:
            kwargs["local_user"] = self._local_user
        entries = []
        offset = 0
        while True:
            try:
                response = self._transfer_client.operation_ls(
                    collection_id,
                    path=sub_path.rstrip("/") + "/",
                    limit=self._ls_page_size,
                    offset=offset,
                    **kwargs,
                )
            except globus_sdk.TransferAPIError as e:
                raise self._transfer_error(e, path) from e
            page = response["DATA"]
            entries.extend(page)
            if len(page) < self._ls_page_size:
                break
            offset += len(page)

        base = path.rstrip("/")
        return [
            {
                "name": f"{base}/{entry['name']}",
                "size": entry.get("size") or 0,
                "type": "directory" if entry.get("type") == "dir" else "file",
                "mtime": _parse_mtime(entry.get("last_modified")),
            }
            for entry in entries
        ]

    def ls(self, path: str, detail: bool = True, **_kwargs: Any) -> list[Any]:
        path = self._strip_protocol(path)
        listing = self.dircache.get(path)
        if listing is None:
            try:
                listing = self._list_dir(path)
            except NotADirectoryError:
                listing = [self.info(path)]
            else:
                self.dircache[path] = listing
        if detail:
            return listing
        return [entry["name"] for entry in listing]

    def info(self, path: str, **_kwargs: Any) -> dict[str, Any]:
        path = self._strip_protocol(path)
        collection_id, sub_path = _split_path(path)
        if sub_path == "/":
            return {"name": collection_id, "size": 0, "type": "directory"}
        parent = self._parent(path)
        for entry in self.ls(parent, detail=True):
            if entry["name"] == path:
                return entry
        raise FileNotFoundError(path)

    # Unlike the defaults, these don't hide permission and network errors.

    def exists(self, path: str, **_kwargs: Any) -> bool:
        try:
            self.info(path)
            return True
        except (FileNotFoundError, NotADirectoryError):
            return False

    def isdir(self, path: str) -> bool:
        try:
            return self.info(path)["type"] == "directory"
        except (FileNotFoundError, NotADirectoryError):
            return False

    def isfile(self, path: str) -> bool:
        try:
            return self.info(path)["type"] == "file"
        except (FileNotFoundError, NotADirectoryError):
            return False

    def modified(self, path: str) -> datetime:
        mtime = self.info(path).get("mtime")
        if mtime is None:
            raise NotImplementedError(f"No modification time available for {path}.")
        return datetime.fromtimestamp(mtime)

    def https_server(self, collection_id: str) -> str:
        if self._https_server_override is not None:
            return self._https_server_override.rstrip("/")
        with self._lock:
            if collection_id not in self._https_servers:
                import globus_sdk

                try:
                    collection = self._transfer_client.get_endpoint(collection_id)
                except globus_sdk.TransferAPIError as e:
                    raise self._transfer_error(e, collection_id) from e
                https_server = collection.get("https_server")
                if not https_server:
                    raise ValueError(
                        f"Globus collection {collection_id} does not support HTTPS access."
                    )
                self._https_servers[collection_id] = https_server.rstrip("/")
            return self._https_servers[collection_id]

    # HTTPS

    def _session(self, renew: bool = False) -> "requests.Session":
        import requests

        session = getattr(self._thread_local, "session", None)
        if session is not None and renew:
            session.close()
            session = None
        if session is None:
            session = requests.Session()
            self._thread_local.session = session
        return session

    def _fetch_range(self, path: str, start: int, end: int) -> bytes:
        """Reads bytes [start, end) of a file."""
        import requests

        if start >= end:
            return b""
        collection_id, sub_path = _split_path(path)
        url = self.https_server(collection_id) + requests.utils.quote(sub_path)
        refreshed_auth = False
        renew_session = False
        for attempt in range(self._retries + 1):
            headers = {"Range": f"bytes={start}-{end - 1}"}
            if (auth := _authorization_header(self._https_authorizer)) is not None:
                headers["Authorization"] = auth
            try:
                response = self._session(renew=renew_session).get(
                    url, headers=headers, timeout=self._timeout
                )
                renew_session = False
                status = response.status_code
                if status == 206:
                    return response.content
                if status == 200:
                    return response.content[start:end]
                if status == 416:
                    return b""
                if status == 401 and not refreshed_auth:
                    handler = getattr(
                        self._https_authorizer, "handle_missing_authorization", None
                    )
                    if handler is not None and handler():
                        refreshed_auth = True
                        continue
                if status in (401, 403):
                    raise GlobusPermissionError(
                        f"No access to {sub_path} on Globus collection {collection_id} "
                        f"via HTTPS ({status}). Make sure the https token has the scope "
                        f"{https_scope(collection_id)} and the identity may read this path."
                    )
                if status == 404 and not any(
                    marker in response.text for marker in _BACKEND_FAULT_MARKERS
                ):
                    raise FileNotFoundError(path)
                if status != 404 and status not in _RETRYABLE_STATUS_CODES:
                    response.raise_for_status()
                    raise OSError(f"Unexpected status {status} when reading {path}.")
                error: Exception = OSError(
                    f"Transient error {status} when reading {path}: {response.text[:200]}"
                )
            except (
                requests.ConnectionError,
                requests.Timeout,
                requests.exceptions.ChunkedEncodingError,
            ) as e:
                error = e
            if attempt == self._retries:
                raise error
            # A fresh connection may hit a healthy backend.
            renew_session = True
            delay = min(0.5 * 2**attempt, 10)
            logger.warning(f"Retrying read of {path} in {delay}s after error: {error}")
            time.sleep(delay)
        raise AssertionError("unreachable")

    def _open(
        self,
        path: str,
        mode: str = "rb",
        block_size: int | None = None,
        autocommit: bool = True,  # noqa: ARG002 files are read-only
        cache_options: dict | None = None,
        **kwargs: Any,
    ) -> "GlobusFile":
        if mode != "rb":
            _read_only()
        info = self.info(path)
        if info["type"] == "directory":
            raise IsADirectoryError(path)
        return GlobusFile(
            self,
            path,
            mode=mode,
            block_size=block_size or "default",
            cache_options=cache_options,
            size=info["size"],
            **kwargs,
        )

    def cat_file(
        self,
        path: str,
        start: int | None = None,
        end: int | None = None,
        **_kwargs: Any,
    ) -> bytes:
        path = self._strip_protocol(path)
        size = self.info(path)["size"]
        start = 0 if start is None else (start if start >= 0 else max(size + start, 0))
        end = size if end is None else (end if end >= 0 else max(size + end, 0))
        return self._fetch_range(path, start, min(end, size))

    mkdir = makedirs = rmdir = rm_file = _rm = rm = mv = cp_file = put_file = (
        pipe_file
    ) = touch = _read_only


class GlobusFile(AbstractBufferedFile):
    fs: GlobusFileSystem

    def _fetch_range(self, start: int, end: int) -> bytes:
        return self.fs._fetch_range(self.path, start, end)


class GlobusPath(CloudPath):
    __slots__ = ()


def register_globus_protocol(*, clobber: bool = False) -> None:
    """Registers `GlobusFileSystem` and `GlobusPath` for `globus://` paths.

    An existing `globus` backend from another package is kept unless `clobber` is True.
    """
    import fsspec
    import upath.registry
    from fsspec.registry import known_implementations, registry

    existing = registry.get(PROTOCOL) or (
        known_implementations.get(PROTOCOL, {}).get("class")
    )
    own = (GlobusFileSystem, f"{GlobusFileSystem.__module__}.GlobusFileSystem")
    if existing is not None and existing not in own and not clobber:
        logger.info(
            "Keeping the existing fsspec backend %s for globus:// paths. "
            "Call webknossos.globus_fs.register_globus_protocol(clobber=True) to replace it.",
            existing,
        )
        return

    # Registering the class itself lets upath derive the path flavour from it.
    fsspec.register_implementation(PROTOCOL, GlobusFileSystem, clobber=True)
    upath.registry.register_implementation(PROTOCOL, GlobusPath, clobber=True)
    # Drops lookups of globus:// cached before this registration.
    cache_clear = getattr(upath.registry.get_upath_class, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()
    try:
        from upath._flavour import WrappedFileSystemFlavour

        # Treat the collection id like a bucket name.
        WrappedFileSystemFlavour.protocol_config["netloc_is_anchor"].add(PROTOCOL)
        WrappedFileSystemFlavour.protocol_config["supports_empty_parts"].add(PROTOCOL)
    except (ImportError, AttributeError, KeyError):
        logger.warning(
            "globus:// paths are not supported with this universal-pathlib version. "
            "Please upgrade universal-pathlib."
        )
