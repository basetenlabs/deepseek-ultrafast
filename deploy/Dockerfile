FROM nvidia/cuda:13.0.1-devel-ubuntu24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates build-essential pkg-config libibverbs-dev librdmacm-dev \
    libopencv-dev clang-18 libclang-18-dev cmake && rm -rf /var/lib/apt/lists/*
ENV CLANG_PATH=/usr/bin/clang-18 LIBCLANG_PATH=/usr/lib/llvm-18/lib
ENV RUSTUP_HOME=/usr/local/rustup CARGO_HOME=/usr/local/cargo
COPY rust-toolchain.toml /src/rust-toolchain.toml
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal --default-toolchain none
ENV PATH=/usr/local/cargo/bin:/usr/lib/llvm-18/bin:$PATH
WORKDIR /src
RUN rustup show active-toolchain || rustup toolchain install
COPY Cargo.toml Cargo.lock ./
COPY crates crates
ARG DSUF_SM=sm_100a
ENV DSUF_SM=${DSUF_SM}
RUN --mount=type=cache,target=/usr/local/cargo/registry --mount=type=cache,target=/usr/local/cargo/git --mount=type=cache,id=deepseek-ultrafast-image-target-${DSUF_SM},target=/src/target \
    cargo build --locked --release -p deepseek-ultrafast-server -p deepseek-ultrafast-kv-transfer --bins && \
    cp target/release/serve target/release/kv-rdma-check /usr/local/bin/

FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04
ARG CUDNN_VERSION=9.14.0.64-1
RUN apt-get update && apt-get install -y --no-install-recommends tini ca-certificates libibverbs1 ibverbs-providers librdmacm1 \
    libcudnn9-cuda-13=${CUDNN_VERSION} libopencv-imgcodecs406 libopencv-imgproc406 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /usr/local/bin/serve /usr/local/bin/kv-rdma-check /usr/local/bin/
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
EXPOSE 8000 19500
ENTRYPOINT ["/usr/bin/tini", "--", "serve", "--host", "0.0.0.0", "--ckpt", "/models"]
CMD ["--gpus", "0,1,2,3", "--port", "8000"]
