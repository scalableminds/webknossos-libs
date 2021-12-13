#! /usr/bin/env -S poetry run python

import calendar
import json
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterable, Tuple

import httpx
from inducoapi import build_openapi
from openapi_python_client import (
    Config,
    MetaType,
    Project,
    _get_project_for_url_or_path,
)

from webknossos.client.context import _get_generated_client
from webknossos.utils import snake_to_camel_case

SCHEMA_URL = "https://converter.swagger.io/api/convert?url=https%3A%2F%2Fmaster.webknossos.xyz%2Fswagger.json"


def assert_valid_schema(openapi_schema: Dict) -> None:
    assert openapi_schema["openapi"].startswith("3.0.")


def generate_client(openapi_schema: Dict) -> None:
    assert_valid_schema(openapi_schema)
    with NamedTemporaryFile("w", suffix=".json") as schema_file:
        json.dump(openapi_schema, schema_file)
        schema_file.flush()
        generator_config = Config(
            project_name_override="webknossos/client/_generated",
            package_name_override=".",
        )
        generator_project = _get_project_for_url_or_path(
            url=None,
            path=Path(schema_file.name),
            meta=MetaType.POETRY,
            config=generator_config,
        )
        assert isinstance(generator_project, Project), generator_project.detail
        _errors = generator_project.update()  # pylint: disable=no-member
        # from openapi_python_client.cli import handle_errors
        # handle_errors(_errors)  # prints warnings


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
    from webknossos.client._generated.api.default import (
        annotation_info,
        build_info,
        current_user_info,
        dataset_info,
        datastore_list,
        generate_token_for_data_store,
        user_list,
        user_logged_time,
    )

    d = datetime.utcnow()
    unixtime = calendar.timegm(d.utctimetuple())
    client = _get_generated_client(enforce_auth=True)

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

    for api_endpoint in [
        datastore_list,
        build_info,
        current_user_info,
        generate_token_for_data_store,
        user_list,
    ]:
        api_endpoint_name = api_endpoint.__name__.split(".")[-1]
        api_endpoint_name = snake_to_camel_case(api_endpoint_name)

        api_endpoint_response = api_endpoint.sync_detailed(client=client)
        assert api_endpoint_response.status_code == 200
        yield api_endpoint_name, api_endpoint_response.content

        if api_endpoint == current_user_info:
            user_id = json.loads(api_endpoint_response.content)["id"]

    user_logged_time_response = user_logged_time.sync_detailed(
        id=user_id,
        client=client,
    )
    assert user_logged_time_response.status_code == 200
    yield "userLoggedTime", user_logged_time_response.content


FIELDS_WITH_VARYING_CONTENT = ["adminViewConfiguration", "novelUserExperienceInfos"]


def make_properties_required(x: Any) -> None:
    if isinstance(x, dict):
        for key, value in x.items():
            # do not recurse into objects where the contents might be varying
            if key in FIELDS_WITH_VARYING_CONTENT:
                continue
            make_properties_required(value)
    elif isinstance(x, list):
        for i in x:
            make_properties_required(i)

    if isinstance(x, dict) and "properties" in x:
        properties = x["properties"]
        if isinstance(properties, dict) and len(properties) > 0:
            assert "required" not in x
            x["required"] = list(
                property
                for property in properties.keys()
                if property not in FIELDS_WITH_VARYING_CONTENT
            )


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
    make_properties_required(recorded_response_schema)
    request_schema["responses"]["200"]["content"] = recorded_response_schema


def fix_request_body(openapi_schema: Dict) -> None:
    assert_valid_schema(openapi_schema)
    for path_val in openapi_schema["paths"].values():
        for method_val in path_val.values():
            if "requestBody" in method_val:
                method_val["requestBody"]["content"] = {
                    "application/json": {"schema": {"type": "object"}}
                }


def bootstrap_response_schemas(openapi_schema: Dict) -> None:
    """Inserts the response schemas into openapi_schema (in-place),
    as recorded by example requests."""
    assert_valid_schema(openapi_schema)
    for operation_id, example_response in iterate_request_ids_with_responses():
        set_response_schema_by_example(
            openapi_schema, example_response=example_response, operation_id=operation_id
        )


if __name__ == "__main__":
    response = httpx.get(SCHEMA_URL)
    response.raise_for_status()
    schema = json.loads(response.text)
    add_api_prefix_for_non_data_paths(schema)
    generate_client(schema)
    fix_request_body(schema)
    bootstrap_response_schemas(schema)
    generate_client(schema)
