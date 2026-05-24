# Dockerfile for RoboCasa.

# Build the container:
# docker build . -t robocasa -f examples/robocasa/Dockerfile

# Run the container:
# docker run --rm -it --network=host -v .:/app -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=$DISPLAY --gpus all robocasa /bin/bash

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

RUN apt-get update && \
    apt-get install -y \
    make \
    g++ \
    clang \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libglfw3-dev \
    libgles2-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    git

WORKDIR /app

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Write the virtual environment outside of the project directory so it doesn't
# leak out of the container when we mount the application code.
ENV UV_PROJECT_ENVIRONMENT=/.venv

# Copy the requirements files so we can install dependencies.
COPY ./examples/robocasa/requirements.txt /tmp/requirements.txt
COPY ./openpi-client/pyproject.toml /tmp/openpi-client/pyproject.toml

# Install python dependencies.
RUN uv venv --python 3.10 $UV_PROJECT_ENVIRONMENT
RUN uv pip sync /tmp/requirements.txt /tmp/openpi-client/pyproject.toml --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match

# Set PYTHONPATH to include the app, openpi-client, and robocasa
ENV PYTHONPATH=/app:/app/openpi-client/src:/app/third_party/robocasa
