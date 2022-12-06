import re

from webknossos.client._generated.api.default import short_link_by_key
from webknossos.client.context import _get_generated_client, webknossos_context

_SHORT_LINK_REGEX = re.compile(
    r"^(?P<webknossos_url>https?://.*)/links/" + r"(?P<short_link_key>.*)"
)


def resolve_short_link(url: str) -> str:
    match = re.match(_SHORT_LINK_REGEX, url)
    if match is None:
        return url
    else:
        webknossos_url = match.group("webknossos_url")
        short_link_key = match.group("short_link_key")
        with webknossos_context(url=webknossos_url):
            client = _get_generated_client()
            response = short_link_by_key.sync(
                key=short_link_key,
                client=client,
            )
        assert response is not None, f"Could not resolve short link {url}."
        return response.long_link
