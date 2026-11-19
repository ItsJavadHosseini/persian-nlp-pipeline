FROM ghcr.io/astral-sh/uv:0.13.0-python3.14-trixie-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_HTTP_TIMEOUT=300 \
    UV_HTTP_RETRIES=5 \
    UV_CACHE_DIR=/tmp/uv-cache \
    PYTHONUNBUFFERED=1

# Copy dependency files.
COPY pyproject.toml uv.lock ./

# Install dependencies without installing the project.
RUN uv sync --frozen --no-dev --no-install-project

# Copy application source code.
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY artifacts/ ./artifacts/
COPY README.md ./

# Install the project itself.
RUN uv sync --frozen --no-dev

# Create evaluation directories.
RUN mkdir -p /app/artifacts/evaluation/uploads

EXPOSE 8000 7860