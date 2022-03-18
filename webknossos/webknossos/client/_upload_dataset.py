import os
import warnings
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory
from time import gmtime, strftime
from typing import Dict, Iterator, List, NamedTuple, Optional, Tuple
from uuid import uuid4

import httpx
from rich.progress import Progress

from webknossos.client._generated.api.datastore import (
    dataset_finish_upload,
    dataset_reserve_upload,
)
from webknossos.client._generated.api.default import (
    datastore_list,
    new_dataset_name_is_valid,
)
from webknossos.client._resumable import Resumable
from webknossos.client.context import _get_context, _WebknossosContext
from webknossos.dataset import Dataset

DEFAULT_SIMULTANEOUS_UPLOADS = 5
MAXIMUM_RETRY_COUNT = 5


class LayerToLink(NamedTuple):
    dataset_name: str
    layer_name: str
    new_layer_name: Optional[str] = None
    organization_id: Optional[
        str
    ] = None  # defaults to the user's organization before uploading

    def as_json(self) -> Dict[str, Optional[str]]:
        context = _get_context()
        return {
            "dataSetName": self.dataset_name,
            "layerName": self.layer_name,
            "newLayerName": self.new_layer_name,
            "organizationName": self.organization_id or context.organization_id,
        }


@lru_cache(maxsize=None)
def _cached_get_upload_datastore(context: _WebknossosContext) -> str:
    datastores = datastore_list.sync(client=context.generated_auth_client)
    assert datastores is not None
    for datastore in datastores:
        if datastore.allows_upload:
            assert isinstance(datastore.url, str)
            return datastore.url
    raise ValueError("No datastore found where datasets can be uploaded.")


def _walk(
    path: Path,
    base_path: Optional[Path] = None,
) -> Iterator[Tuple[Path, Path, int]]:
    if base_path is None:
        base_path = path
    if path.is_dir():
        for p in path.iterdir():
            yield from _walk(p, base_path)
    else:
        yield (path.resolve(), path.relative_to(base_path), path.stat().st_size)


def upload_dataset(
    dataset: Dataset,
    new_dataset_name: Optional[str] = None,
    layers_to_link: Optional[List[LayerToLink]] = None,
    jobs: Optional[int] = None,
) -> str:
    from webknossos.client._generated.models import (
        DatasetFinishUploadJsonBody,
        DatasetReserveUploadJsonBody,
    )

    if new_dataset_name is None:
        new_dataset_name = dataset.name
    if layers_to_link is None:
        layers_to_link = []
    context = _get_context()
    layer_names_to_link = set(i.new_layer_name or i.layer_name for i in layers_to_link)
    if len(layer_names_to_link.intersection(dataset.layers.keys())) > 0:
        warnings.warn(
            "Excluding the following layers from upload, since they will be linked: "
            + f"{layer_names_to_link.intersection(dataset.layers.keys())}"
        )
        with TemporaryDirectory() as tmpdir:
            tmp_ds = dataset.shallow_copy_dataset(
                tmpdir, name=dataset.name, layers_to_ignore=layer_names_to_link
            )
            return upload_dataset(
                tmp_ds,
                new_dataset_name=new_dataset_name,
                layers_to_link=layers_to_link,
                jobs=jobs,
            )

    file_infos = list(_walk(dataset.path))
    total_file_size = sum(size for _, _, size in file_infos)
    # replicates https://github.com/scalableminds/webknossos/blob/master/frontend/javascripts/admin/dataset/dataset_upload_view.js
    time_str = strftime("%Y-%m-%dT%H-%M-%S", gmtime())
    upload_id = f"{time_str}__{uuid4()}"
    datastore_token = context.datastore_required_token
    datastore_url = _cached_get_upload_datastore(context)
    datastore_client = _get_context().get_generated_datastore_client(datastore_url)
    simultaneous_uploads = jobs if jobs is not None else DEFAULT_SIMULTANEOUS_UPLOADS
    if "PYTEST_CURRENT_TEST" in os.environ:
        simultaneous_uploads = 1
    response = new_dataset_name_is_valid.sync_detailed(
        organization_name=context.organization_id,
        data_set_name=new_dataset_name,
        client=context.generated_auth_client,
    )
    assert (
        response.status_code == 200
    ), f"Dataset name {context.organization_id}/{new_dataset_name} does not seem to be valid: {response}"
    for _ in range(MAXIMUM_RETRY_COUNT):
        response = dataset_reserve_upload.sync_detailed(
            client=datastore_client,
            token=datastore_token,
            json_body=DatasetReserveUploadJsonBody.from_dict(
                {
                    "uploadId": upload_id,
                    "organization": context.organization_id,
                    "name": new_dataset_name,
                    "totalFileCount": len(file_infos),
                    "initialTeams": [],
                    "layersToLink": [layer.as_json() for layer in layers_to_link],
                }
            ),
        )
        if response.status_code == 200:
            break
    else:
        assert response.status_code == 200, response
    with Progress() as progress:
        with Resumable(
            f"{datastore_url}/data/datasets?token={datastore_token}",
            simultaneous_uploads=simultaneous_uploads,
            query={
                "owningOrganization": context.organization_id,
                "name": new_dataset_name,
                "totalFileCount": len(file_infos),
            },
            chunk_size=100 * 1024 * 1024,  # 100 MiB
            generate_unique_identifier=lambda _, relative_path: f"{upload_id}/{relative_path}",
            test_chunks=False,
            permanent_errors=[400, 403, 404, 409, 415, 500, 501],
            client=httpx.Client(timeout=None),
        ) as session:
            progress_task = progress.add_task("Dataset Upload", total=total_file_size)
            for file_path, relative_path, _ in file_infos:
                resumable_file = session.add_file(file_path, relative_path)
                resumable_file.chunk_completed.register(
                    lambda chunk: progress.advance(progress_task, chunk.size)
                )
    for _ in range(MAXIMUM_RETRY_COUNT):
        response = dataset_finish_upload.sync_detailed(
            client=datastore_client.with_timeout(None),  # type: ignore[arg-type]
            token=datastore_token,
            json_body=DatasetFinishUploadJsonBody.from_dict(
                {
                    "uploadId": upload_id,
                    "organization": context.organization_id,
                    "name": new_dataset_name,
                    "needsConversion": False,
                    "layersToLink": [layer.as_json() for layer in layers_to_link],
                }
            ),
        )
        if response.status_code == 200:
            break
    else:
        assert response.status_code == 200, response

    return f"{context.url}/datasets/{context.organization_id}/{new_dataset_name}/view"
