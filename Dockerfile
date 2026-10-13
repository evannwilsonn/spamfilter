FROM rust:1.90-slim AS build
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY static ./static
COPY data ./data
RUN cargo build --release \
 && ./target/release/spamfilter train --bigrams --threshold 0.9 --out /app/model.json

FROM debian:bookworm-slim
WORKDIR /app
COPY --from=build /app/target/release/spamfilter /usr/local/bin/spamfilter
COPY --from=build /app/model.json /app/model.json
EXPOSE 3000
CMD ["spamfilter", "serve", "--model", "/app/model.json", "--addr", "0.0.0.0:3000"]
