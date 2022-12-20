import calendar
import json
import os
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterable, List, Optional, Tuple

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


def remove_zarr_tagged_endpoints(openapi_schema: Dict) -> None:
    """This removes all endpoints tagged with zarr-streaming, in-place."""
    assert_valid_schema(openapi_schema)
    paths = openapi_schema["paths"]
    for path, path_value in list(paths.items()):
        for method, method_value in list(path_value.items()):
            if "zarr-streaming" in method_value.get("tags", []):
                del path_value[method]
        if len(path_value) == 0:
            del paths[path]


def iterate_request_ids_with_responses() -> Iterable[Tuple[str, bytes]]:
    """Send requests to webKnossos and record the schema of their replies"""
    from webknossos.client._generated.api.default import (
        annotation_info,
        annotation_infos_by_task_id,
        build_info,
        current_user_info,
        dataset_info,
        dataset_list,
        dataset_sharing_token,
        datastore_list,
        generate_token_for_data_store,
        project_info_by_id,
        project_info_by_name,
        short_link_by_key,
        task_info,
        task_infos_by_project_id,
        team_list,
        user_info_by_id,
        user_list,
        user_logged_time,
    )
    from webknossos.client.context import _get_generated_client
    from webknossos.utils import snake_to_camel_case

    organization_id = "Organization_X"
    dataset_name = "e2006_knossos"
    task_id = "581367a82faeb37a008a5352"
    user_id = "570b9f4d2a7c0e4d008da6ef"
    project_id = "58135bfd2faeb3190181c057"
    project_name = "Test_Project"
    explorative_annotation_id = "58135c192faeb34c0081c05d"

    response = httpx.get(
        url=f"{WK_URL}/api/datasets/{organization_id}/{dataset_name}",
        headers={"X-Auth-Token": WK_TOKEN},
    )
    assert (
        response.status_code == 200 and response.json()["isActive"]
    ), f"You need to copy or link any dataset to binaryData/{organization_id}/{dataset_name}."

    d = datetime.utcnow()
    unixtime = calendar.timegm(d.utctimetuple())
    client = _get_generated_client(enforce_auth=True)

    yield (
        "annotationInfo",
        extract_200_response(
            annotation_info.sync_detailed(
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
                organization_name=organization_id,
                data_set_name=dataset_name,
                client=client,
            )
        ),
    )

    yield (
        "datasetList",
        extract_200_response(
            dataset_list.sync_detailed(
                client=client,
            )
        ),
    )

    yield (
        "datasetSharingToken",
        extract_200_response(
            dataset_sharing_token.sync_detailed(
                organization_name=organization_id,
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
        "teamList",
        extract_200_response(
            team_list.sync_detailed(
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

    short_link_key = httpx.post(
        url=f"{WK_URL}/api/shortLinks",
        json=WK_URL,
        headers={"X-Auth-Token": WK_TOKEN},
    ).json()["key"]

    yield (
        "shortLinkByKey",
        extract_200_response(
            short_link_by_key.sync_detailed(
                key=short_link_key,
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
            extract_200_response(api_endpoint.sync_detailed(client=client)),  # type: ignore[attr-defined]
        )


# By default, any keys are optional and can have the type "Unset".
# Any keys with the following names are marked as required,
# and are therefore expected to be returned by the backend in
# related requests. Exceptions are specified in OPTIONAL_KEYS below.
REQUIRED_KEYS = {
    "id",
    "name",
    "token",
    ##### Datastore #####
    "allowsUpload",
    ##### User #####
    "email",
    "organization",
    "firstName",
    "lastName",
    "created",
    "lastActivity",
    # "teams", added 2022-07, optional for backwards-compatibility
    "experiences",
    "isActive",
    "isAdmin",
    "isDatasetManager",
    "loggedTime",
    "paymentInterval",
    "year",
    "month",
    "durationInSeconds",
    ##### Project #####
    "team",
    "teamName",
    # "owner", optional in annotations
    "priority",
    "paused",
    "expectedTime",
    ##### Task #####
    "projectId",
    "dataSet",
    "status",
    "open",
    "active",
    "finished",
    ##### Annotation #####
    # "owner", optional
    "description",
    "typ",
    "state",
    ##### Dataset #####
    "dataStore",
    "url",
    "dataSource",
    # "scale", not available for errors
    # "dataLayers", not available for errors
    "category",
    "elementClass",
    "boundingBox",
    "topLeft",
    "width",
    "height",
    "depth",
    "resolutions",
    "sharingToken",
    "displayName",
    "description",
    "tags",
    "isPublic",
    "allowedTeams",
    ##### Short links ####
    "longLink",
}

# Those key-pairs of (parent-key, child-key) mark exceptions
# for keys that are usually required, just not under the
# parent keys listed here:
OPTIONAL_KEYS = {
    ("annotationLayers", "name"),  # added 2022-07, optional for backwards-compatibility
    ("dataSource", "status"),  # sometimes part of the dataSource dict
}

# Anything in the following keys will not be marked as required,
# as those keys usually vary:
KEYS_WITH_VARYING_VALUES = {
    "experiences",
    "adminViewConfiguration",
    "novelUserExperienceInfos",
    "viewConfiguration",
    "defaultViewConfiguration",
}


def extract_200_response(response: Any) -> bytes:
    assert response.status_code == 200, response.content
    return response.content


def make_properties_required(  # pylint: disable=dangerous-default-value
    x: Any,
    parent_name: Optional[str] = None,
    handled_required_keys: List[str] = [],
) -> List[str]:
    if isinstance(x, dict):
        for key, value in x.items():
            # do not recurse into objects where the contents might be varying
            if key in KEYS_WITH_VARYING_VALUES:
                continue
            if key == "properties" and isinstance(value, dict):
                for property_key, property_value in x["properties"].items():
                    if property_key in KEYS_WITH_VARYING_VALUES:
                        continue
                    make_properties_required(
                        property_value,
                        parent_name=property_key,
                        handled_required_keys=handled_required_keys,
                    )
            else:
                make_properties_required(
                    value,
                    parent_name=parent_name,
                    handled_required_keys=handled_required_keys,
                )
    elif isinstance(x, list):
        for i in x:
            make_properties_required(
                i, parent_name=parent_name, handled_required_keys=handled_required_keys
            )

    if isinstance(x, dict) and "properties" in x:
        properties = x["properties"]
        if isinstance(properties, dict) and len(properties) > 0:
            assert "required" not in x
            required_properties = []
            for property_key in properties.keys():
                if property_key in REQUIRED_KEYS:
                    if (parent_name, property_key) not in OPTIONAL_KEYS:
                        required_properties.append(property_key)

            if len(required_properties) > 0:
                handled_required_keys += required_properties
                x["required"] = required_properties

            # Further corrections
            if "task" in properties:
                properties["task"]["nullable"] = True
            if "tracingTime" in properties:
                # is null during client-generation, but actually is Optional[int]
                properties["tracingTime"]["type"] = "integer"
                properties["tracingTime"]["nullable"] = True

    return handled_required_keys


def set_response_schema_by_example(
    openapi_schema: Dict,
    example_response: bytes,
    operation_id: str,
    method: str = "get",
) -> List[str]:
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
    handled_required_keys = make_properties_required(recorded_response_schema)
    request_schema["responses"]["200"]["content"] = recorded_response_schema
    return handled_required_keys


def fix_request_body(openapi_schema: Dict) -> None:
    """Applies fixes for request bodies in-place."""
    assert_valid_schema(openapi_schema)
    for path_val in openapi_schema["paths"].values():
        for method_val in path_val.values():
            if "requestBody" in method_val:
                if method_val.get("operationId") == "datasetUpdateTeams":
                    method_val["requestBody"]["content"] = {
                        "application/json": {
                            "schema": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                else:
                    method_val["requestBody"]["content"] = {
                        "application/json": {"schema": {"type": "object"}}
                    }


def bootstrap_response_schemas(openapi_schema: Dict) -> None:
    """Inserts the response schemas into openapi_schema (in-place),
    as recorded by example requests."""
    assert_valid_schema(openapi_schema)
    handled_required_keys = []
    for operation_id, example_response in iterate_request_ids_with_responses():
        handled_required_keys += set_response_schema_by_example(
            openapi_schema, example_response=example_response, operation_id=operation_id
        )
    left_over = REQUIRED_KEYS - set(handled_required_keys)
    assert (
        len(left_over) == 0
    ), f"Did not find all required keys, left over are {left_over}"


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
    remove_zarr_tagged_endpoints(schema)
    generate_client(schema)
    fix_request_body(schema)
    bootstrap_response_schemas(schema)
    generate_client(schema)
