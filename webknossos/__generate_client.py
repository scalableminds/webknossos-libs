import calendar
import json
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Iterable, Tuple

import httpx
from inducoapi import build_openapi
from openapi_python_client import (
    Config,
    MetaType,
    Project,
    _get_project_for_url_or_path,
)
from openapi_python_client.cli import handle_errors

from webknossos.client import _get_generated_client
from webknossos.utils import snake_to_camel_case

SCHEMA_URL = "https://converter.swagger.io/api/convert?url=https%3A%2F%2Fwebknossos.org%2Fswagger.json"


def assert_valid_schema(openapi_schema: Dict) -> None:
    assert openapi_schema["openapi"].startswith("3.0.")


def generate_client(openapi_schema: Dict) -> None:
    assert_valid_schema(openapi_schema)
    with NamedTemporaryFile("w", suffix=".json") as schema_file:
        schema_file.write(json.dumps(openapi_schema))
        generator_config = Config(
            project_name_override="webknossos/client/generated",
            package_name_override=".",
        )
        generator_project = _get_project_for_url_or_path(
            url=None,
            path=Path(schema_file.name),
            meta=MetaType.POETRY,
            config=generator_config,
        )
        assert isinstance(generator_project, Project)
        errors = generator_project.update()
        # handle_errors(errors)  # prints warnings


def add_api_prefix_for_non_data_paths(openapi_schema: Dict) -> None:
    """The current webKnossos backend does not include the
    /api prefix into the different backend paths.
    Howevery, the /data prefix for datastore paths is included.
    This adds the missing /api prefixes in the openapi_schema (in-place)."""
    assert_valid_schema(openapi_schema)
    paths = openapi_schema["paths"]
    for path in list(paths.keys()):
        assert path[0] == "/", f"{path} must start with /"
        if path.startswith("/data/"):
            continue
        paths[f"/api{path}"] = paths.pop(path)


def iterate_request_ids_with_responses() -> Iterable[Tuple[str, bytes]]:
    from webknossos.client.generated.api.default import (
        annotation_info,
        build_info,
        dataset_info,
        datastore_list,
    )

    d = datetime.utcnow()
    unixtime = calendar.timegm(d.utctimetuple())
    client = _get_generated_client(enforce_token=True)

    annotation_info_response = annotation_info.sync_detailed(
        typ="Explorational",
        id="6114d9410100009f0096c640",
        client=client,
        timestamp=unixtime,
    )
    assert annotation_info_response.status_code == 200
    yield "annotationInfo", annotation_info_response.content

    dataset_info_response = dataset_info.sync_detailed(
        organization_name="scalable_minds",
        data_set_name="l4dense_motta_et_al_demo",
        client=client,
    )
    assert dataset_info_response.status_code == 200
    yield "datasetInfo", dataset_info_response.content

    for api_endpoint in [datastore_list, build_info]:
        api_endpoint_name = api_endpoint.__name__.split(".")[-1]
        api_endpoint_name = snake_to_camel_case(api_endpoint_name)

        response = api_endpoint.sync_detailed(client=client)
        assert response.status_code == 200
        yield api_endpoint_name, response.content


def set_response_schema_by_example(
    openapi_schema: Dict,
    example_response: bytes,
    operation_id: str,
    method: str = "get",
) -> None:
    recorded_schema = build_openapi(
        method=method,
        path="/placeholder",
        resp_code="200",
        response=example_response,
    )
    assert_valid_schema(recorded_schema)
    recorded_response_schema = recorded_schema["paths"]["/placeholder"][method][
        "responses"
    ]["200"]["content"]

    # Update openapi_schema in-place
    request_schema = [
        path_method
        for path in openapi_schema["paths"].values()
        for path_method in path.values()
        if path_method["operationId"] == operation_id
    ][0]
    request_schema["responses"]["200"]["content"] = recorded_response_schema


def bootstrap_response_schemas(openapi_schema: Dict) -> None:
    """Inserts the response schemas into openapi_schema (in-place),
    as recorded by example requests."""
    assert_valid_schema(openapi_schema)
    for operation_id, response in iterate_request_ids_with_responses():
        set_response_schema_by_example(
            openapi_schema, example_response=response, operation_id=operation_id
        )


if __name__ == "__main__":
    response = httpx.get(SCHEMA_URL)
    response.raise_for_status()
    schema = json.loads(response.text)
    add_api_prefix_for_non_data_paths(schema)
    generate_client(schema)
    bootstrap_response_schemas(schema)
    generate_client(schema)
