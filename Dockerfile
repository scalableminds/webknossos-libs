FROM python:3.7

RUN mkdir /app
WORKDIR /app
COPY requirements.txt /app
COPY setup.py /app
COPY README.md /app
RUN pip install -r requirements.txt

COPY wkcuber /app/wkcuber
COPY tests /app/tests

COPY .git /app/.git
RUN python setup.py install
RUN rm -r /app/.git

ENTRYPOINT [ "python", "-m" ]
