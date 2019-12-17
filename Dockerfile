FROM python:3.7

RUN mkdir /app
WORKDIR /app
COPY requirements.txt /app
COPY setup.py /app
COPY README.md /app
RUN pip install poetry

COPY wkcuber /app/wkcuber
COPY tests /app/tests

RUN poetry install --no-dev

ENTRYPOINT [ "python", "-m" ]
