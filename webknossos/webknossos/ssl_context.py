import copyreg
import ssl
from collections.abc import Callable
from typing import Any

import certifi


def _create_sslcontext() -> ssl.SSLContext:
    cafile = certifi.where()
    ssl_context = ssl.create_default_context(cafile=cafile)
    ssl_context.cafile = cafile  # type: ignore
    return ssl_context


def _save_sslcontext(
    obj: ssl.SSLContext,
) -> tuple[Callable[[Any, Any], ssl.SSLContext], tuple[ssl._SSLMethod, str | None]]:
    cafile = getattr(obj, "cafile", None)
    return _rebuild_sslcontext, (obj.protocol, cafile)


def _rebuild_sslcontext(protocol: ssl._SSLMethod, cafile: str | None) -> ssl.SSLContext:
    ssl_context = ssl.SSLContext(protocol)
    if cafile is not None:
        ssl_context.load_verify_locations(cafile=cafile)
        ssl_context.cafile = cafile  # type: ignore
    return ssl_context


SSL_CONTEXT = _create_sslcontext()
copyreg.pickle(ssl.SSLContext, _save_sslcontext)
