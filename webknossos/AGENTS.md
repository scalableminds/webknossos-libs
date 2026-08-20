# Coding Agent Instructions

## Development

* When writing code comments, keep them short and concise, do not reference other code or previous behavior.
* Prefer keyword arguments over positional arguments, where there are more than 1-2 arguments and the ordering isn't abundantly clear.
* Inside our codebase, the NormalizedBoundingBox should be used, because it removes some ambiguity (regarding channels and axes). Just on user-facing interfaces should the BoundingBox and NDBoundingBox be used.

## Testing

* Run tests with `uv run test.py` (not `python -m pytest`). 
* However, running all tests takes several minutes and should be done by the CI. Try to pick a small subset of tests to run.

## Format, lint, typecheck

* After code changes, run `./precommit.sh`.

## Documentation

* API documentation is generated from docstrings in the code.
* For CLI commands, the docs are in `../docs/src/cli/`.
* New pages need to be registered in `../docs/mkdocs.yml`.
