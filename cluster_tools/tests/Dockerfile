FROM python:3.11

RUN mkdir /cluster_tools
COPY requirements.txt /cluster_tools
COPY poetry.lock /cluster_tools
COPY pyproject.toml /cluster_tools

WORKDIR /cluster_tools

RUN pip install -r requirements.txt && poetry config virtualenvs.create false && poetry install --all-extras

COPY . /cluster_tools

RUN poetry install --all-extras
