import attr


@attr.frozen
class Tracingstore:
    """Tracingstore class for interactions with the tracing store."""

    name: str
    url: str

    @classmethod
    def get_tracingstore(
        cls,
    ) -> "Tracingstore":
        """Get the tracingstore for current webknossos url.


        Returns:
            Tracingstore object

        Examples:
            ```
            # Get a list of all datastores that allow dataset uploads
            tracingstore = Tracingstore.get_tracingstore()
            ```
        """

        from ..client.context import _get_context

        context = _get_context()
        api_tracingstore = context.api_client_with_auth.tracingstore()
        return cls(
            api_tracingstore.name,
            api_tracingstore.url,
        )
