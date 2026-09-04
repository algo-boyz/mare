FROM ubuntu:25.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV GIT_TERMINAL_PROMPT=0

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git pkg-config \
    libprotobuf-dev protobuf-compiler \
    libgrpc++-dev protobuf-compiler-grpc \
    libssl-dev libcurl4-openssl-dev ca-certificates \
    libabsl-dev \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .
RUN rm -rf build CMakeCache.txt CMakeFiles

RUN git config --global http.version HTTP/1.1

RUN cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_CLICKHOUSE_CONSUMER=ON \
      -DWITH_OTEL=ON \
    && cmake --build build -j$(nproc) \
         --target nats_publisher ingest_server consumer clickhouse_consumer

# runtime
FROM ubuntu:25.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libprotobuf32t64 libgrpc++1.51t64 libssl3 ca-certificates \
    libabsl20230802 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /src/build/nats_publisher /app/
COPY --from=builder /src/build/ingest_server /app/
COPY --from=builder /src/build/consumer /app/
COPY --from=builder /src/build/clickhouse_consumer /app/

CMD ["/app/nats_publisher"]