from functools import lru_cache
from pathlib import Path
from shutil import make_archive
from tempfile import NamedTemporaryFile
from time import gmtime, strftime
from uuid import uuid4

import httpx
from rich.progress import Progress

from webknossos.client._generated.api.default import datastore_list
from webknossos.client._resumable import Resumable
from webknossos.client.context import _get_context, _WebknossosContext
from webknossos.dataset import Dataset


@lru_cache(maxsize=None)
def _cached_get_upload_datastore(context: _WebknossosContext) -> str:
    datastores = datastore_list.sync(client=context.generated_auth_client)
    assert datastores is not None
    for datastore in datastores:
        if datastore.allows_upload:
            assert isinstance(datastore.url, str)
            return datastore.url
    raise ValueError("No datastore found where datasets can be uploaded.")


def upload_dataset(dataset: Dataset) -> str:
    context = _get_context()
    # replicates https://github.com/scalableminds/webknossos/blob/master/frontend/javascripts/admin/dataset/dataset_upload_view.js
    with NamedTemporaryFile(suffix=".zip") as tmp_zip_file:
        zip_path = Path(tmp_zip_file.name)
        make_archive(str(zip_path.with_suffix("")), "zip", dataset.path)
        time_str = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        upload_id = f"{time_str}__{uuid4()}"
        datastore_token = context.datastore_token
        datastore_url = _cached_get_upload_datastore(context)
        for _ in range(5):
            try:
                httpx.post(
                    f"{datastore_url}/data/datasets/reserveUpload?token={datastore_token}",
                    params={"token": datastore_token},
                    json={
                        "uploadId": upload_id,
                        "organization": context.organization,
                        "name": dataset.name,
                        "totalFileCount": 1,
                        "initialTeams": [],
                    },
                    timeout=60,
                ).raise_for_status()
                break
            except httpx.HTTPStatusError as e:
                http_error = e
        else:
            raise http_error
        with Progress() as progress:
            with Resumable(
                f"{datastore_url}/data/datasets?token={datastore_token}",
                simultaneous_uploads=1,
                query={
                    "owningOrganization": context.organization,
                    "name": dataset.name,
                    "totalFileCount": 1,
                },
                chunk_size=10 * 1024 * 1024,  # 10 MiB
                generate_unique_identifier=lambda path: f"{upload_id}/{path.name}",
                test_chunks=False,
                permanent_errors=[400, 403, 404, 409, 415, 500, 501],
                client=httpx.Client(timeout=None),
            ) as session:
                file = session.add_file(zip_path)
                progress_task = progress.add_task(
                    "Dataset Upload", total=len(file.chunks)
                )
                file.chunk_completed.register(lambda _: progress.advance(progress_task))
    for _ in range(5):
        try:
            httpx.post(
                f"{datastore_url}/data/datasets/finishUpload?token={datastore_token}",
                params={"token": datastore_token},
                json={
                    "uploadId": upload_id,
                    "organization": context.organization,
                    "name": dataset.name,
                    "needsConversion": False,
                },
                timeout=None,
            ).raise_for_status()
            break
        except httpx.HTTPStatusError as e:
            http_error = e
    else:
        raise http_error

    return f"{context.url}/datasets/{context.organization}/{dataset.name}/view"
