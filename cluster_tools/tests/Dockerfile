FROM python:3.11

COPY --from=ghcr.io/astral-sh/uv:0.4.20 /uv /bin/uv

RUN mkdir /cluster_tools
COPY uv.lock /cluster_tools
COPY pyproject.toml /cluster_tools

WORKDIR /cluster_tools

RUN uv venv --system-site-packages
RUN uv sync --all-extras --frozen

COPY . /cluster_tools
