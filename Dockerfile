FROM python:3

ENV RUSTUP_HOME=/usr/local/rustup \
    CARGO_HOME=/usr/local/cargo \
    PATH=/usr/local/cargo/bin:$PATH \
    LZ4_VERSION=1.8.1.2

RUN wget https://github.com/lz4/lz4/archive/v${LZ4_VERSION}.tar.gz -O liblz4.tar.gz && \
  tar -xvzf liblz4.tar.gz && \
  cd lz4-${LZ4_VERSION} && \
  make && \
  make install && \
  ldconfig

RUN set -eux; \
    \
# this "case" statement is generated via "update.sh"
    dpkgArch="$(dpkg --print-architecture)"; \
  case "${dpkgArch##*-}" in \
    amd64) rustArch='x86_64-unknown-linux-gnu'; rustupSha256='4b7a67cd971d713e0caef48b5754190aca19192d1863927a005c3432512b12dc' ;; \
    armhf) rustArch='armv7-unknown-linux-gnueabihf'; rustupSha256='622190c3f478a56563d45f6fbc1fab02d356b631c28a1beba2c3e4c68de3c14c' ;; \
    arm64) rustArch='aarch64-unknown-linux-gnu'; rustupSha256='a39d7643cdced9ad70a9927bbb0a861b579884f94793881b771d3a0f92c0ddd8' ;; \
    i386) rustArch='i686-unknown-linux-gnu'; rustupSha256='9e921fce4a2cc1f04095be6d623effdead0aab1261472e6933da9e6030330b90' ;; \
    *) echo >&2 "unsupported architecture: ${dpkgArch}"; exit 1 ;; \
  esac; \
    \
    url="https://static.rust-lang.org/rustup/archive/1.9.0/${rustArch}/rustup-init"; \
    wget "$url"; \
    echo "${rustupSha256} *rustup-init" | sha256sum -c -; \
    chmod +x rustup-init; \
    ./rustup-init -y --no-modify-path --default-toolchain 1.23.0; \
    rm rustup-init; \
    chmod -R a+w $RUSTUP_HOME $CARGO_HOME; \
    rustup --version; \
    cargo --version; \
    rustc --version;

RUN pip install \
  numpy \
  cffi \
  setuptools \
  pytest && \
  wget https://github.com/scalableminds/webknossos-wrap/archive/master.tar.gz -O webknossos-wrap.tar.gz && \
  tar -xzvf webknossos-wrap.tar.gz && \
  cd webknossos-wrap-master/python && \
  python setup.py install


RUN mkdir /app
WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY wkcuber /app/wkcuber
COPY config.yml /app
COPY setup.py /app

RUN python setup.py install

ENTRYPOINT [ "python", "-m" ]
