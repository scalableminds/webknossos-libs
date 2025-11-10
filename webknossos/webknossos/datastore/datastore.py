import attr


@attr.frozen
class Datastore:
    """Datastore class for managing datastores."""

    name: str
    url: str
    allows_upload: bool

    @classmethod
    def get_datastores(
        cls,
        allows_upload: bool | None = None,
    ) -> list["Datastore"]:
        """Get a list of all datastore URLs.

        Args:
            allows_upload: Optional filter for datastores that allow or deny dataset uploads (`None` for all)

        Returns:
            List of datastore URLs

        Examples:
            ```
            # Get a list of all datastores that allow dataset uploads
            datastores = Dataset.get_upload_datastores()
            ```
        """

        from ..client.context import _get_api_client

        client = _get_api_client()
        if allows_upload is None:
            return [
                cls(datastore.name, datastore.url, datastore.allows_upload)
                for datastore in client.datastore_list()
            ]
        return [
            cls(datastore.name, datastore.url, datastore.allows_upload)
            for datastore in client.datastore_list()
            if allows_upload == datastore.allows_upload
        ]

    @classmethod
    def get_upload_url(cls) -> str:
        """Get URL for uploading a dataset.

        Returns:
            Upload URL

        Examples:
            ```
            # Get the upload URL for the first datastore that allows dataset uploads
            upload_url = Datastore.get_upload_url()
            ```
        """

        return cls.get_datastores(allows_upload=True)[0].url
