FROM python:3.11

COPY --from=ghcr.io/astral-sh/uv:0.5.22 /uv /bin/uv

RUN mkdir /cluster_tools
COPY . /cluster_tools

WORKDIR /cluster_tools

RUN uv export --all-extras --output-file requirements.txt && \
    uv pip sync --system requirements.txt && \
    uv cache clean

