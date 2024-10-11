FROM apache/answer as answer-builder

FROM golang:1.23-alpine AS golang-builder

COPY --from=answer-builder /usr/bin/answer /usr/bin/answer

RUN apk --no-cache add \
    build-base git bash nodejs npm go && \
    npm install -g pnpm@8.9.2

RUN answer build \
    --with github.com/apache/incubator-answer-plugins/storage-s3 \
    --with github.com/apache/incubator-answer-plugins/search-meilisearch \
    --output /usr/bin/new_answer

FROM alpine
LABEL maintainer="linkinstar@apache.org"

ARG TIMEZONE
ENV TIMEZONE=${TIMEZONE:-"Asia/Tokyo"}

RUN apk update \
    && apk --no-cache add \
    bash \
    ca-certificates \
    curl \
    dumb-init \
    gettext \
    openssh \
    sqlite \
    gnupg \
    tzdata \
    && ln -sf /usr/share/zoneinfo/${TIMEZONE} /etc/localtime \
    && echo "${TIMEZONE}" > /etc/timezone

COPY --from=golang-builder /usr/bin/new_answer /usr/bin/answer
COPY --from=answer-builder /data /data
COPY --from=answer-builder /entrypoint.sh /entrypoint.sh
RUN chmod 755 /entrypoint.sh

VOLUME /data
EXPOSE 80
ENTRYPOINT ["/entrypoint.sh"]