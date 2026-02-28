# ============================================================================
# AgriSense — Multi-stage Dockerfile
# Stage 1: Julia precompilation (cached layer)
# Stage 2: Python dependencies (cached layer)
# Stage 3: Runtime — Python 3.12 + Julia 1.11 + CUDA-ready
# ============================================================================

# ── Stage 1: Julia dependency precompilation ────────────────────────────────
FROM julia:1.11-bookworm AS julia-deps

WORKDIR /julia-build
COPY core/AgriSenseCore/Project.toml core/AgriSenseCore/Project.toml

# Instantiate deps (downloads packages, creates Manifest.toml)
RUN julia --project=core/AgriSenseCore -e ' \
    using Pkg; \
    Pkg.instantiate(); \
    Pkg.precompile(); \
'

# ── Stage 2: Python dependencies ────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS python-deps

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir --prefix=/install .

# ── Stage 3: Runtime ────────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm AS runtime

# Metadata
LABEL maintainer="Mehdi Skouri"
LABEL description="AgriSense — GPU-accelerated agricultural hypergraph API"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl wget \
    && rm -rf /var/lib/apt/lists/*

# Install Julia 1.11
ENV JULIA_VERSION=1.11.2
RUN wget -q "https://julialang-s3.julialang.org/bin/linux/x64/1.11/julia-${JULIA_VERSION}-linux-x86_64.tar.gz" \
    && tar -xzf "julia-${JULIA_VERSION}-linux-x86_64.tar.gz" -C /opt/ \
    && ln -s "/opt/julia-${JULIA_VERSION}/bin/julia" /usr/local/bin/julia \
    && rm "julia-${JULIA_VERSION}-linux-x86_64.tar.gz"

# Copy Python deps from build stage
COPY --from=python-deps /install /usr/local

# Copy precompiled Julia packages
COPY --from=julia-deps /root/.julia /root/.julia

# Application code
WORKDIR /app
COPY . .

# Julia project setup: point to our package + precompile in final image
ENV JULIA_PROJECT=/app/core/AgriSenseCore
ENV JULIA_NUM_THREADS=auto
RUN julia --project=core/AgriSenseCore -e ' \
    using Pkg; \
    Pkg.instantiate(); \
    Pkg.precompile(); \
'

# Runtime config
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
