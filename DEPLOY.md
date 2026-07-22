# Deploying the web app

The recommended public deployment uses one NVIDIA L4 GPU on Cloud Run. It
serves both final cascades, keeps the Hybrid selected by default, and scales to
zero when idle. All four checkpoints are baked into the image, so a cold
instance does not download model weights before serving its first request.

`Dockerfile.gpu` is the production image. The original `Dockerfile` remains a
CPU-only fallback for local use or a lower-cost service.

## Prerequisites

1. Put all four release assets in `pinnfluid/webapp/checkpoints/`. Run
   `python pinnfluid/webapp/fetch_checkpoints.py` after its release URLs have
   been configured.
2. Install Docker with NVIDIA Container Toolkit support for the local GPU test.
3. Install the Google Cloud CLI and create a Google Cloud project with billing
   enabled.

## Local GPU test

From the repository root:

```bash
docker build --progress=plain -f Dockerfile.gpu -t pinnfluid-web:gpu-local .

docker run --rm --gpus all pinnfluid-web:gpu-local \
  python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'

docker run --rm --name pinnfluid-web-gpu \
  --gpus all \
  -p 8080:8080 \
  --memory=12g \
  --cpus=4 \
  pinnfluid-web:gpu-local
```

Open <http://localhost:8080>. Verify the runtime in another terminal:

```bash
curl -fsS http://localhost:8080/healthz
```

The JSON must contain `"resolved": "cuda"`, `"cuda_available": true`, both
model ids, and the NVIDIA GPU name. `PINN_DEVICE=cuda` makes a missing GPU fail
loudly instead of silently running the public service on CPU.

## Google Cloud setup

Choose the identifiers once. Cloud Run L4 is available in Belgium; keep the
Artifact Registry repository and service in the same region.

```bash
PROJECT_ID=your-gcp-project-id
REGION=europe-west1
SERVICE=pinnfluid
REPOSITORY=pinnfluid

gcloud auth login
gcloud config set project "$PROJECT_ID"
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com

gcloud artifacts repositories create "$REPOSITORY" \
  --repository-format=docker \
  --location="$REGION" \
  --description="pinnfluid web images"
```

If the repository already exists, the final command reports `ALREADY_EXISTS`;
continue with the build.

## Build and deploy

Cloud Build uses `cloudbuild.gpu.yaml`, which explicitly selects
`Dockerfile.gpu`:

```bash
TAG=0.1.0
IMAGE="$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/web:$TAG"

gcloud builds submit \
  --config cloudbuild.gpu.yaml \
  --substitutions "_IMAGE=$IMAGE" \
  .
```

Deploy privately first:

```bash
gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --region "$REGION" \
  --platform managed \
  --execution-environment gen2 \
  --no-allow-unauthenticated \
  --gpu 1 \
  --gpu-type nvidia-l4 \
  --no-gpu-zonal-redundancy \
  --cpu 4 \
  --memory 16Gi \
  --concurrency 4 \
  --timeout 3600 \
  --max-instances 1 \
  --min-instances 0 \
  --no-cpu-throttling \
  --set-env-vars "^@^PINN_DEVICE=cuda@PINN_WEBAPP_HOST=0.0.0.0@PINN_WEBAPP_MODELS=hybrid-cascade-292d-pinn,grid-unet-cascade-292d-pinn@PINN_WEBAPP_GITHUB_URL=https://github.com/jimmygasser/pinnfluid@PINN_WEBAPP_RESULTS_DIR=/tmp/pinnfluid-results@PINN_WEBAPP_MAX_RUNS=5@PINN_WEBAPP_MAX_GB=1@PINN_WEBAPP_MAX_ACTIVE_JOBS=1@PINN_WEBAPP_RATE_LIMIT_JOBS=4@PINN_WEBAPP_RATE_LIMIT_WINDOW=3600@PINN_WEBAPP_TORCH_THREADS=4@OMP_NUM_THREADS=4@MKL_NUM_THREADS=4@OPENBLAS_NUM_THREADS=4@NUMEXPR_NUM_THREADS=4"
```

`--no-cpu-throttling` is required by the current background-thread job model.
`--max-instances 1` keeps jobs and status polls on the same in-memory instance
and caps simultaneous GPU allocation. `--min-instances 0` allows scale-to-zero.
The non-zonal L4 option is appropriate for a restartable research demo and is
cheaper than zonal redundancy.

## Private smoke test

Proxy the private service to a local port:

```bash
gcloud run services proxy "$SERVICE" --region "$REGION" --port 8085
```

Open <http://localhost:8085>, check `/healthz`, and complete one Hybrid and one
U-Net prediction. Inspect Cloud Run logs if either fails:

```bash
gcloud run services logs read "$SERVICE" --region "$REGION" --limit 100
```

## Make it public

After the private smoke test, disable the Invoker IAM check (Google's
recommended public-access method):

```bash
gcloud run services update "$SERVICE" \
  --region "$REGION" \
  --no-invoker-iam-check

gcloud run services describe "$SERVICE" \
  --region "$REGION" \
  --format='value(status.url)'
```

## Cost controls

Three different controls are involved:

- **Cloud Run:** `min-instances=0` and `max-instances=1` control allocated
  instances. They do not impose a monthly spending ceiling.
- **Application:** one heavy job at a time and four submissions per hour are
  enforced in-process by `PINN_WEBAPP_MAX_ACTIVE_JOBS` and
  `PINN_WEBAPP_RATE_LIMIT_*`. The counter resets on scale-down; this is a
  practical beta guard, not DDoS protection.
- **Cloud Billing:** create a project budget and email alerts in
  **Billing > Budgets & alerts**. Start with a small monthly budget such as
  USD/CHF 10 and alerts at 50%, 90%, and 100%. A budget alert notifies you; it
  does not automatically stop the service.

To adjust the public limit later:

```bash
gcloud run services update "$SERVICE" \
  --region "$REGION" \
  --update-env-vars PINN_WEBAPP_RATE_LIMIT_JOBS=8
```

For stronger per-user enforcement, put the service behind an external HTTPS
load balancer with Cloud Armor rate limiting or add authentication. That adds
cost and complexity and is not required for the initial low-traffic beta.

## Storage and cold starts

The `/tmp` result store is ephemeral and consumes instance memory. Completed
runs disappear whenever Cloud Run replaces or scales down the instance. This
is acceptable for immediate downloads; persistent run history would require
Cloud Storage and a job-state redesign.

The first request after scale-to-zero includes container startup, Python imports,
checkpoint loading, and model inference. Later requests on the same warm
instance reuse both cached model stacks.

## CPU fallback

The CPU image remains available:

```bash
docker build --progress=plain -t pinnfluid-web:cpu-local .
docker run --rm -p 8080:8080 --memory=4g --cpus=4 pinnfluid-web:cpu-local
```

For a CPU Cloud Run deployment, remove the GPU flags, use 4 GiB memory, and set
`PINN_WEBAPP_MODELS=hybrid-cascade-292d-pinn` unless long U-Net waits are
acceptable.
