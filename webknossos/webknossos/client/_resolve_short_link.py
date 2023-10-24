import logging
import re
from urllib.parse import urlparse

from .context import _get_api_client, webknossos_context

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
            client = _get_api_client()
            return client.short_link_by_key(key=short_link_key).long_link
    except Exception as e:
        logger.warning(f"Got an error during short link resolution of link {url}: {e}")
        return url
