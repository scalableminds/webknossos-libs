ARG PYTHON_VERSION="3.11"

FROM python:${PYTHON_VERSION}

COPY --from=ghcr.io/astral-sh/uv:0.5.26 /uv /bin/uv

RUN mkdir /cluster_tools
COPY . /cluster_tools

WORKDIR /cluster_tools

RUN uv export --all-extras --output-file requirements.txt && \
    uv pip sync --system requirements.txt && \
    uv cache clean

