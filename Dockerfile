# syntax=docker/dockerfile:1.7
FROM python:3.11-slim-bookworm

ARG TORCH_VERSION=2.9.1

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache \
    PINN_WEBAPP_HOST=0.0.0.0 \
    PINN_WEBAPP_MODELS=hybrid-cascade-292d-pinn,grid-unet-cascade-292d-pinn \
    PINN_WEBAPP_GITHUB_URL=https://github.com/jimmygasser/pinnfluid \
    PINN_WEBAPP_RESULTS_DIR=/tmp/pinnfluid-results \
    PINN_WEBAPP_MAX_RUNS=5 \
    PINN_WEBAPP_MAX_GB=1 \
    PORT=8080

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements-webapp.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch==${TORCH_VERSION}" \
    && python -m pip install -r requirements-webapp.txt

ENV PINN_WEBAPP_TORCH_THREADS=4 \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4 \
    OPENBLAS_NUM_THREADS=4 \
    NUMEXPR_NUM_THREADS=4

RUN groupadd --gid 10001 pinnfluid \
    && useradd --uid 10001 --gid pinnfluid --create-home pinnfluid

COPY --chown=pinnfluid:pinnfluid . .

# A public checkout has no weights until fetch_checkpoints.py is run. Fail the
# image build explicitly rather than shipping an incomplete model registry.
RUN test -s pinnfluid/webapp/checkpoints/hybrid-stage1-292d.pth \
    && test -s pinnfluid/webapp/checkpoints/hybrid-stage2-292d.pth \
    && test -s pinnfluid/webapp/checkpoints/grid-unet-stage1-292d.pth \
    && test -s pinnfluid/webapp/checkpoints/grid-unet-stage2-292d.pth \
    && mkdir -p dem pinnfluid/webapp/workspace \
    && chown -R pinnfluid:pinnfluid dem pinnfluid/webapp/workspace

USER pinnfluid

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8080') + '/healthz', timeout=3)" || exit 1

CMD ["python", "-u", "pinnfluid/webapp/app.py", "--no-browser"]
