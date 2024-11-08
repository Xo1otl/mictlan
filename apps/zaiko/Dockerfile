FROM golang:1.23.3 AS build

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o initkafka ./cmd/initkafka
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build -o echoserver ./cmd/echoserver

FROM docker.redpanda.com/redpandadata/connect:latest

COPY --from=build /app/initkafka .
COPY --from=build /app/echoserver .
