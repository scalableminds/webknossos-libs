FROM python:3.8

RUN mkdir /cluster_tools
COPY poetry.lock /cluster_tools
COPY pyproject.toml /cluster_tools

WORKDIR /cluster_tools

RUN pip install poetry==1.6.1 && poetry install --all-extras

COPY . /cluster_tools

RUN poetry install --all-extras
