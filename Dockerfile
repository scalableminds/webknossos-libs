FROM python:3-slim

RUN mkdir /app
WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY wkcuber /app/wkcuber
COPY config.yml /app
COPY setup.py /app

RUN python setup.py install

ENTRYPOINT [ "python", "-m", "wkcuber" ]
