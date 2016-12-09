FROM python:3-slim

RUN mkdir /app
WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY webknossos_cuber /app/webknossos_cuber
COPY config.yml /app

ENTRYPOINT [ "python", "webknossos_cuber/cuber.py", "--config", "config.yml" ]
