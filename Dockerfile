FROM python:3-slim

RUN mkdir /app
WORKDIR /app

COPY webknossos_cuber /app
COPY config.yml /app

ENTRYPOINT [ "python", "webknossos_cuber/cuber.py", "--config", "config.yml" ]
