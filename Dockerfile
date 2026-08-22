# syntax=docker/dockerfile:1.6

# --- Stage 1: UI builder ---
FROM node:20-slim AS ui-builder
# git is required because some npm deps resolve via git (e.g. Baileys optional deps).
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app/ui
COPY ui/package*.json ./
RUN npm install --no-audit --no-fund --loglevel=error
# ui/src/i18n/store.ts imports '../../../roboclaw/i18n/{common,setup}.json'.
# Keep this COPY narrow (not the whole i18n dir) so Python-side i18n edits
# don't invalidate the ui-builder cache.
COPY roboclaw/i18n/common.json roboclaw/i18n/setup.json /app/roboclaw/i18n/
COPY ui/ ./
RUN npm run build

# --- Stage 2: WhatsApp bridge builder ---
FROM node:20-slim AS bridge-builder
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app/bridge
COPY bridge/package*.json ./
RUN npm install --no-audit --no-fund --loglevel=error
COPY bridge/ ./
RUN npm run build

# --- Stage 3: Python runtime ---
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      curl ca-certificates git nodejs \
      build-essential linux-libc-dev \
      ffmpeg libavcodec59 libavformat59 libavutil57 libswresample4 libswscale6 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps (layer-cached on pyproject/uv.lock)
# pyproject.toml force-includes `bridge/` into the wheel, so its source must
# be present when hatchling runs during `uv pip install -e .`.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY roboclaw/ roboclaw/
COPY bridge/   bridge/

# Fail loudly if the LeRobot submodule wasn't populated on the host before
# `docker build` — without it, `uv pip install` dies mid-resolution with a
# cryptic hatchling error because pyproject.toml points `lerobot` at this path.
RUN test -f roboclaw/embodied/engine/pyproject.toml \
 || (echo "ERROR: LeRobot submodule not populated at roboclaw/embodied/engine/. \
Clone the repo with --recurse-submodules (or run 'git submodule update --init --recursive') \
before 'docker build'." >&2 && exit 1)

# CPU-only torch: the WSL2 host has no GPU passthrough and the v1 goal is
# teleop + record, not inference/training. This saves ~4 GB of wheel downloads
# and ~6 GB of image size vs. the default CUDA wheels.
RUN uv pip install --system --no-cache -e . \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      --index-strategy unsafe-best-match

# Built artifacts from earlier stages (merged into the already-copied bridge/ dir)
COPY --from=ui-builder     /app/ui/dist              /app/ui/dist
COPY --from=bridge-builder /app/bridge/dist          /app/bridge/dist
COPY --from=bridge-builder /app/bridge/node_modules  /app/bridge/node_modules

RUN mkdir -p /root/.roboclaw /root/.cache/huggingface

# Keep /app/roboclaw/**/*.py clean of .pyc under the editable install.
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8765 18790 1455

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -fsS http://localhost:8765/api/health || exit 1

ENTRYPOINT ["roboclaw"]
CMD ["status"]
