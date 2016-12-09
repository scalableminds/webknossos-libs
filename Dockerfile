FROM python:3-slim

RUN pip install numpy scipy pillow

RUN mkdir /app
WORKDIR /app

COPY cuber.py /app
COPY config.ini /app

ENTRYPOINT [ "python", "cuber.py", "--config", "config.ini" ]
