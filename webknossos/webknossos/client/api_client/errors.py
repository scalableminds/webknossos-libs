import httpx


class ApiClientError(Exception):
    def message_for_response_body(self, response: httpx.Response) -> str:
        response_limit_chars = 2000

        response_str = response.content.decode("utf-8")
        shortened_label = (
            f" (showing first {response_limit_chars} of {len(response_str)} characters)"
            if (len(response_str) > response_limit_chars)
            else ""
        )
        return f"Got response status {response.status_code} with body{shortened_label}: {response_str[0:response_limit_chars]}"

    check_version_hint = """If this is unexpected, please double-check your WEBKNOSSOS URL and credentials.
If the error persists, it might be caused by a version mismatch of the python client and the WEBKNOSSOS server API version.
See https://github.com/scalableminds/webknossos-libs/releases for current releases."""

    def request_label(self, response: httpx.Response) -> str:
        if response.request is None:
            return "a WEBKNOSSOS API request"
        return f"a {response.request.method} request for URL {response.request.url}"


class UnexpectedStatusError(ApiClientError):
    def __init__(self, response: httpx.Response):
        msg = f"""An error occurred while performing {self.request_label(response)}.
{self.check_version_hint}
{self.message_for_response_body(response)}
"""
        super().__init__(msg)


class CannotHandleResponseError(ApiClientError):
    def __init__(self, response: httpx.Response):
        msg = f"""An error occurred while processing the response to {self.request_label(response)}.
{self.check_version_hint}
{self.message_for_response_body(response)}
"""

        super().__init__(msg)
