FROM python:3.8

COPY poetry.lock /cluster_tools
COPY pyproject.toml /cluster_tools

WORKDIR /cluster_tools

RUN pip install poetry && poetry install

COPY . /cluster_tools
