FROM python:3.10

COPY requirements.txt /

RUN pip install -r /requirements.txt

RUN mkdir /webknossos
COPY webknossos/webknossos /webknossos/webknossos
COPY webknossos/uv.lock /webknossos/
COPY webknossos/pyproject.toml /webknossos/
COPY webknossos/README.md /webknossos/

RUN mkdir /cluster_tools
COPY cluster_tools/cluster_tools /cluster_tools/cluster_tools
COPY cluster_tools/poetry.lock /cluster_tools/
COPY cluster_tools/pyproject.toml /cluster_tools/
COPY cluster_tools/README.md /cluster_tools/

WORKDIR /webknossos

RUN uv sync --all-extras --frozen
RUN poetry config virtualenvs.create false && poetry install --all-extras
