import calendar
import json
import os
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

WK_URL = os.environ["WK_URL"]
assert WK_URL == "http://localhost:9000", (
    f"The wrong WK_URL is configured, got {WK_URL}, expected http://localhost:9000. "
    + "Are you running this script via ./generate_client.sh?"
)
WK_TOKEN = os.environ["WK_TOKEN"]
SCHEMA_URL = f"{WK_URL}/swagger.json"
CONVERTER_URL = "https://converter.swagger.io/api/convert"


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
    """Send requests to webKnossos and record the schema of their replies"""
    from webknossos.client._generated.api.default import (
        annotation_info,
        annotation_infos_by_task_id,
        build_info,
        current_user_info,
        dataset_info,
        datastore_list,
        generate_token_for_data_store,
        project_info_by_id,
        project_info_by_name,
        task_info,
        task_infos_by_project_id,
        user_info_by_id,
        user_list,
        user_logged_time,
    )
    from webknossos.client.context import _get_generated_client
    from webknossos.utils import snake_to_camel_case

    organization_name = "Organization_X"
    dataset_name = "e2006_knossos"
    task_id = "581367a82faeb37a008a5352"
    user_id = "570b9f4d2a7c0e4d008da6ef"
    project_id = "58135bfd2faeb3190181c057"
    project_name = "Test_Project"
    explorative_annotation_id = "58135c192faeb34c0081c05d"

    extract_200_response(
        httpx.post(
            url=f"{WK_URL}/data/triggers/checkInboxBlocking?token={WK_TOKEN}",
        )
    )
    response = httpx.get(
        url=f"{WK_URL}/api/datasets/{organization_name}/{dataset_name}",
        headers={"X-Auth-Token": f"{WK_TOKEN}"},
    )
    assert (
        response.status_code == 200 and response.json()["isActive"]
    ), f"You need to copy or link any dataset to binaryData/{organization_name}/{dataset_name}."

    d = datetime.utcnow()
    unixtime = calendar.timegm(d.utctimetuple())
    client = _get_generated_client(enforce_auth=True)

    yield (
        "annotationInfo",
        extract_200_response(
            annotation_info.sync_detailed(
                typ="Explorational",
                id=explorative_annotation_id,
                client=client,
                timestamp=unixtime,
            )
        ),
    )

    yield (
        "datasetInfo",
        extract_200_response(
            dataset_info.sync_detailed(
                organization_name=organization_name,
                data_set_name=dataset_name,
                client=client,
            )
        ),
    )

    yield (
        "taskInfo",
        extract_200_response(
            task_info.sync_detailed(
                id=task_id,
                client=client,
            ),
        ),
    )

    yield (
        "userInfoById",
        extract_200_response(
            user_info_by_id.sync_detailed(
                id=user_id,
                client=client,
            ),
        ),
    )

    yield (
        "projectInfoById",
        extract_200_response(
            project_info_by_id.sync_detailed(
                id=project_id,
                client=client,
            ),
        ),
    )

    yield (
        "projectInfoByName",
        extract_200_response(
            project_info_by_name.sync_detailed(name=project_name, client=client),
        ),
    )

    yield (
        "taskInfosByProjectId",
        extract_200_response(
            task_infos_by_project_id.sync_detailed(
                id=project_id,
                client=client,
            ),
        ),
    )

    yield (
        "annotationInfosByTaskId",
        extract_200_response(
            annotation_infos_by_task_id.sync_detailed(id=task_id, client=client),
        ),
    )

    yield (
        "userLoggedTime",
        extract_200_response(
            user_logged_time.sync_detailed(
                id=user_id,
                client=client,
            ),
        ),
    )

    for api_endpoint in [
        datastore_list,
        build_info,
        current_user_info,
        generate_token_for_data_store,
        user_list,
    ]:
        api_endpoint_name = api_endpoint.__name__.split(".")[-1]
        api_endpoint_name = snake_to_camel_case(api_endpoint_name)

        yield (
            api_endpoint_name,
            extract_200_response(api_endpoint.sync_detailed(client=client)),
        )


FIELDS_WITH_VARYING_CONTENT = [
    "experiences",
    "adminViewConfiguration",
    "novelUserExperienceInfos",
    "viewConfiguration",
    "defaultViewConfiguration",
]

OPTIONAL_FIELDS = [
    "adminViewConfiguration",
    "novelUserExperienceInfos",
    "viewConfiguration",
    "defaultViewConfiguration",
    # isSuperUser field was added 2022-03 and only optional for backwards-compatibility with wk,
    # it can be made non-optional when needed later:
    "isSuperUser",
]


def extract_200_response(response: Any) -> bytes:
    assert response.status_code == 200, response.content
    return response.content


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
                if property not in OPTIONAL_FIELDS
            )

            # Further corrections
            if "task" in properties:
                properties["task"]["nullable"] = True
            if "tracingTime" in properties:
                # is null during client-generation, but actually is Optional[int]
                properties["tracingTime"]["type"] = "integer"
                properties["tracingTime"]["nullable"] = True


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
    schema_response = httpx.get(SCHEMA_URL)
    schema_response.raise_for_status()
    converter_response = httpx.post(
        CONVERTER_URL,
        content=schema_response.text,
        headers={"content-type": "application/json"},
    )
    converter_response.raise_for_status()
    schema = json.loads(converter_response.text)
    add_api_prefix_for_non_data_paths(schema)
    generate_client(schema)
    fix_request_body(schema)
    bootstrap_response_schemas(schema)
    generate_client(schema)
