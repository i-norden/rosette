# ---- Builder stage ----
FROM python:3.14-slim@sha256:486b8092bfb12997e10d4920897213a06563449c951c5506c2a2cfaf591c599f AS builder

LABEL org.opencontainers.image.source="https://github.com/i-norden/sniffer"
LABEL org.opencontainers.image.description="LLM-powered academic integrity analyzer"

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libffi-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install dependencies from lock file for reproducible builds
COPY requirements.lock pyproject.toml ./
RUN pip install --no-cache-dir --prefix=/install -r requirements.lock

# Copy source and install the package itself
COPY snoopy/ snoopy/
COPY config/ config/
RUN pip install --no-cache-dir --no-deps --prefix=/install .

# ---- Runtime stage ----
FROM python:3.14-slim@sha256:486b8092bfb12997e10d4920897213a06563449c951c5506c2a2cfaf591c599f

# Runtime system deps for PyMuPDF, opencv-python-headless, Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy config and any templates/static assets
COPY config/ config/

# Create data directories and non-root user
RUN useradd --create-home --shell /bin/bash snoopy \
    && mkdir -p /data/pdfs /data/figures /data/reports \
    && chown -R snoopy:snoopy /data /app

USER snoopy

# Declare volumes for persistent data
VOLUME ["/data/pdfs", "/data/figures", "/data/reports"]

# Default environment variables (DATABASE_URL must be provided at runtime)
ENV SNOOPY__STORAGE__PDF_DIR="/data/pdfs" \
    SNOOPY__STORAGE__FIGURES_DIR="/data/figures" \
    SNOOPY__STORAGE__REPORTS_DIR="/data/reports" \
    SNOOPY__LLM__PROVIDER="claude" \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=15s \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["snoopy"]
CMD ["serve"]
