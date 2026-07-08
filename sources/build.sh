#!/usr/bin/env sh
set -eu

IMAGE="${IMAGE:-searchculturebot-pipelines:latest}"

docker build -t "$IMAGE" -f Dockerfile.pipelines .
