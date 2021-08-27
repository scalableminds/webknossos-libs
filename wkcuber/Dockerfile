FROM python:3.7

RUN mkdir /app
WORKDIR /app

COPY README.md /app
COPY poetry.toml /app
COPY poetry.lock /app
COPY pyproject.toml /app

RUN pip install poetry

COPY wkcuber /app/wkcuber
COPY tests /app/tests

RUN poetry install --no-dev

ENTRYPOINT [ "python", "-m" ]
