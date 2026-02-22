import os
import warnings
from collections.abc import Iterator
from functools import cache
from tempfile import TemporaryDirectory
from time import gmtime, strftime
from uuid import uuid4

import httpx
from upath import UPath

from .. import LayerToLink
from ..dataset import Dataset
from ..datastore import Datastore
from ..utils import get_rich_progress
from ._resumable import Resumable
from .api_client.models import (
    ApiDatasetUploadInformation,
    ApiReserveDatasetUploadInformation,
)
from .context import _get_context, _WebknossosContext, webknossos_context

DEFAULT_SIMULTANEOUS_UPLOADS = 5
MAXIMUM_RETRY_COUNT = 4


@cache
def _cached_get_upload_datastore(context: _WebknossosContext) -> str:
    with webknossos_context(context.url, context.token):
        return Datastore.get_upload_url()


def _walk(
    path: UPath,
    base_path: UPath | None = None,
) -> Iterator[tuple[UPath, UPath, int]]:
    if base_path is None:
        base_path = path
    if path.is_dir():
        for p in path.iterdir():
            yield from _walk(p, base_path)
    else:
        yield (path.resolve(), path.relative_to(base_path), path.stat().st_size)


def upload_dataset(
    dataset: Dataset,
    new_dataset_name: str | None = None,
    layers_to_link: list[LayerToLink] | None = None,
    jobs: int | None = None,
    datastore_url: str | None = None,
    folder_id: str | None = None,
) -> str:
    if new_dataset_name is None:
        new_dataset_name = dataset.name
    if layers_to_link is None:
        layers_to_link = []
    context = _get_context()
    layer_names_to_link = set(i.new_layer_name or i.layer_name for i in layers_to_link)
    if len(layer_names_to_link.intersection(dataset.layers.keys())) > 0:
        warnings.warn(
            "[INFO] Excluding the following layers from upload, since they will be linked: "
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
    datastore_url = datastore_url or _cached_get_upload_datastore(context)
    datastore_api_client = context.get_datastore_api_client(datastore_url)
    simultaneous_uploads = jobs if jobs is not None else DEFAULT_SIMULTANEOUS_UPLOADS
    if "PYTEST_CURRENT_TEST" in os.environ:
        simultaneous_uploads = 1
    is_valid_new_name_response = context.api_client.dataset_is_valid_new_name(
        dataset_name=new_dataset_name
    )
    if not is_valid_new_name_response.is_valid:
        problems_str = ""
        if is_valid_new_name_response.errors is not None:
            problems_str = f" Problems: {is_valid_new_name_response.errors}"
        raise Exception(
            f"Dataset name {context.organization_id}/{new_dataset_name} is not a valid new dataset name.{problems_str}"
        )

    datastore_api_client.dataset_reserve_upload(
        reserve_upload_information=ApiReserveDatasetUploadInformation(
            upload_id,
            new_dataset_name,
            context.organization_id,
            total_file_count=len(file_infos),
            total_file_size_in_bytes=total_file_size,
            layers_to_link=[
                layer._as_api_linked_layer_identifier() for layer in layers_to_link
            ],
            folder_id=folder_id,
            initial_teams=[],
        ),
        retry_count=MAXIMUM_RETRY_COUNT,
    )
    with get_rich_progress() as progress:
        with Resumable(
            f"{datastore_url}/data/datasets",
            simultaneous_uploads=simultaneous_uploads,
            query={
                "owningOrganization": context.organization_id,
                "name": new_dataset_name,
                "totalFileCount": len(file_infos),
            },
            headers={"X-Auth-Token": context.token},
            chunk_size=100 * 1024 * 1024,  # 100 MiB
            generate_unique_identifier=lambda _, relative_path: (
                f"{upload_id}/{relative_path.as_posix()}"
            ),
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

    dataset_id = datastore_api_client.dataset_finish_upload(
        upload_information=ApiDatasetUploadInformation(upload_id),
        retry_count=MAXIMUM_RETRY_COUNT,
    )

    return dataset_id
