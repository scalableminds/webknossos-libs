FROM python:3.12

COPY --from=ghcr.io/astral-sh/uv:0.4.20 /uv /bin/uv

RUN mkdir /webknossos
COPY webknossos/webknossos /webknossos/webknossos
COPY webknossos/uv.lock /webknossos/
COPY webknossos/pyproject.toml /webknossos/
COPY webknossos/README.md /webknossos/

RUN mkdir /cluster_tools
COPY cluster_tools/cluster_tools /cluster_tools/cluster_tools
COPY cluster_tools/uv.lock /cluster_tools/
COPY cluster_tools/pyproject.toml /cluster_tools/
COPY cluster_tools/README.md /cluster_tools/

WORKDIR /webknossos

RUN uv venv --system-site-packages
RUN uv sync --all-extras --frozen
