FROM python:3.7

RUN mkdir /app
WORKDIR /app

COPY wkcuber/README.md /app
COPY wkcuber/poetry.toml /app
COPY wkcuber/poetry.lock /app
COPY wkcuber/pyproject.toml /app

RUN pip install poetry

COPY wkcuber/wkcuber /app/wkcuber
COPY wkcuber/tests /app/tests

RUN mkdir /webknossos
COPY webknossos/webknossos /webknossos/webknossos
COPY webknossos/poetry.lock /webknossos/
COPY webknossos/pyproject.toml /webknossos/
COPY webknossos/README.md /webknossos/

RUN poetry config virtualenvs.create false --local
RUN poetry install --no-dev

ENTRYPOINT [ "python", "-m" ]
