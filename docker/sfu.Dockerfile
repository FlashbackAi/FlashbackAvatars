FROM golang:1.22

WORKDIR /app
COPY services/sfu-gateway/ /app/

# TODO: Install ion-sfu dependencies and produce a minimal binary.
RUN go mod init flashback/sfu-gateway || true

CMD ["go", "run", "main.go"]