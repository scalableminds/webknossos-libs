# webKnossos REST-API client

:warning: The code of this client is auto-generated! Please don't adapt manually, but re-run the code generation. :warning:

The client code is generated using [openapi-python-client](https://github.com/openapi-generators/openapi-python-client) (version 0.10.3, Python 3.7.3).

To re-generate,
1. [install openapi-python-client](https://github.com/openapi-generators/openapi-python-client#installation),
2. move into the `client` directory (`cd webknossos/webknossos/client`),
3. run `openapi-python-client generate --url https://raw.githubusercontent.com/openapi-generators/openapi-python-client/main/end_to_end_tests/openapi.json`,
4. move the relevant files into the correct folder:
   ```
   mv webknossos-client/webknossos_client/api .
   mv webknossos-client/webknossos_client/models .
   mv webknossos-client/webknossos_client/*.py .
   ```
5. verify that all dependencies from `webknossos-client/pyproject.toml` are in the top-level `pyproject.toml`, and
6. `rm -rf webknossos-client`.


4. move the relevant files into the correct folder:
   ```
   mv my-test-api-client/my_test_api_client/api .
   mv my-test-api-client/my_test_api_client/models .
   mv my-test-api-client/my_test_api_client/*.py .
   ```
5. verify that all dependencies from `my-test-api-client/pyproject.toml` are in the top-level `pyproject.toml`, and
6. `rm -rf my-test-api-client`
