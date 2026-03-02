# ============================================================================
# AgriSense — Multi-stage Dockerfile
# Targets:
#   - runtime-cpu (default): CPU-first runtime
#   - runtime-gpu: GPU-enabled runtime (host must provide GPU runtime)
# ============================================================================

# ── Stage 1: Julia dependency precompilation ────────────────────────────────
FROM julia:1.12-bookworm AS julia-deps

WORKDIR /julia-build
COPY core/AgriSenseCore/Project.toml core/AgriSenseCore/Project.toml

# Instantiate deps (downloads packages, creates Manifest.toml)
RUN julia --project=core/AgriSenseCore -e ' \
    using Pkg; \
    Pkg.instantiate(); \
    Pkg.precompile(); \
'

# ── Stage 2: Python dependencies ────────────────────────────────────────────
FROM python:3.13-slim-bookworm AS python-deps

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --prefix=/install .

# ── Stage 3: Runtime base ───────────────────────────────────────────────────
FROM python:3.13-slim-bookworm AS runtime-base

# Metadata
LABEL maintainer="Mehdi Skouri"
LABEL description="AgriSense — GPU-accelerated agricultural hypergraph API"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl wget \
    && rm -rf /var/lib/apt/lists/*

# Install Julia 1.12
ENV JULIA_VERSION=1.12.5
RUN wget -q "https://julialang-s3.julialang.org/bin/linux/x64/1.12/julia-${JULIA_VERSION}-linux-x86_64.tar.gz" \
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
# Precompile is performed in the julia-deps stage; do a lightweight runtime sanity check only.
RUN julia --version

# Runtime config
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# ── Stage 4a: CPU runtime target (default target) ──────────────────────────
FROM runtime-base AS runtime-cpu
LABEL org.opencontainers.image.title="agrisense-cpu"
LABEL org.opencontainers.image.description="AgriSense API CPU runtime"
ENV AGRISENSE_RUNTIME_VARIANT=cpu

# ── Stage 4b: GPU runtime target ────────────────────────────────────────────
FROM runtime-base AS runtime-gpu
LABEL org.opencontainers.image.title="agrisense-gpu"
LABEL org.opencontainers.image.description="AgriSense API GPU runtime"
ENV AGRISENSE_RUNTIME_VARIANT=gpu
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
