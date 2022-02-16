FROM python:3.8

RUN mkdir /cluster_tools
COPY poetry.lock /cluster_tools
COPY pyproject.toml /cluster_tools

WORKDIR /cluster_tools

RUN pip install poetry && poetry config virtualenvs.create false && poetry install

COPY . /cluster_tools

RUN poetry install
