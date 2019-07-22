FROM python:3.7

ENV LZ4_VERSION=1.8.1.2

RUN wget https://github.com/lz4/lz4/archive/v${LZ4_VERSION}.tar.gz -O liblz4.tar.gz && \
  tar -xvzf liblz4.tar.gz && \
  cd lz4-${LZ4_VERSION} && \
  make && \
  make install && \
  ldconfig

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
