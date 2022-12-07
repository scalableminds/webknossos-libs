import logging
import re
from urllib.parse import urlparse

from webknossos.client._generated.api.default import short_link_by_key
from webknossos.client.context import _get_generated_client, webknossos_context

logger = logging.getLogger(__name__)


_SHORT_LINK_REGEX = re.compile(r"^/links/(?P<short_link_key>.+)$")


def resolve_short_link(url: str) -> str:
    try:
        parts = urlparse(url)
        if parts.scheme not in ["http", "https"]:
            return url
        if parts.netloc == "":
            return url

        match = re.match(_SHORT_LINK_REGEX, parts.path)
        if match is None:
            return url

        webknossos_url = f"{parts.scheme}://{parts.netloc}"
        short_link_key = match.group("short_link_key")
        with webknossos_context(url=webknossos_url):
            client = _get_generated_client()
            response = short_link_by_key.sync(
                key=short_link_key,
                client=client,
            )
        assert response is not None, f"Could not resolve short link {url}."
        return response.long_link
    except Exception as e:
        logger.warning(f"Got an error during short link resolution of link {url}: {e}")
        return url
