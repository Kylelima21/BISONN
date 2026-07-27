# BISONN — Biotic Interactions with Sage Observations using Neural Networks

## Overview

BISONN is a Sage/Waggle edge plugin that detects **bird mobbing behavior** from
camera images and identifies the **species** in each image. It uses BioCLIP 2.5
embeddings (frozen, ViT-H/14, 1024-dim) fed into a trained Linear SVM classifier
to classify each camera snapshot as either `mobbing` (birds harassing or
distracting a predator/threat) or `none` (no mobbing interaction). It also uses
BioCLIP's text encoder for zero-shot species identification via cosine similarity
to a curated list of 30 North American bird species.

## How It Works

1. **Capture** — acquires a camera snapshot (RTSP stream, HTTP snapshot URL, or
   test image directory)
2. **Embed** — encodes the image with BioCLIP 2.5 Huge (1024-dim, L2-normalized)
3. **Classify behavior** — the frozen Linear SVM predicts `mobbing` (1) or `none` (0)
4. **Identify species** — cosine similarity between the image embedding and 30
   curated species text prompts (zero-shot, no training needed — BioCLIP was
   trained on TreeOfLife-200M species captions)
5. **Publish** — publishes behavior, species, and heartbeat topics to the Sage
   data pipeline via pywaggle
6. **Upload** — optionally uploads an annotated image (behavior + species overlay)
   for human review

## Model

- **Backbone**: BioCLIP 2.5 Huge (`imageomics/bioclip-2.5-vith14`, ViT-H/14,
  ~1B params, 1024-dim embeddings, TreeOfLife-200M pretraining)
- **Behavior classifier**: Linear SVM (`sklearn.svm.SVC`, `kernel='linear'`,
  `class_weight='balanced'`)
- **Species ID**: Zero-shot cosine similarity to 30 species text prompts
  (corvids, raptors, songbirds — curated for North American mobbing contexts)
- **Model file**: `models/svm.joblib` (1.7 MB)
- **Training data**: 1690 labeled bird images (101 mobbing + 1589 none)
- **Performance**: 98.5% accuracy, 0.935 macro-F1, mobbing F1=0.878
  (80/20 stratified split, seed=42)

## Topics Published

| Topic | Value | Meta |
|-------|-------|------|
| `biotic.interaction.bird_mobbing` | 1 (mobbing) or 0 (none) | camera, label, confidence, species, species_confidence, model |
| `biotic.species.bird` | 1 (presence indicator) | camera, species, species_confidence, species_top3, model |
| `biotic.interaction.summary` | same (heartbeat) | camera, label, confidence, species, model |

All meta values are strings (pywaggle requirement).

## Input Modes

- `--stream <source>` — camera name (e.g. `bottom_camera`), RTSP URL, or image path
- `--snapshot-url <url>` — HTTP URL returning a JPEG (Reolink CGI API; creds in query string)
- `--image-dir <path>` — batch process all images in a directory

## Deployment

Built on the NVIDIA PyTorch 25.08 base (`nvcr.io/nvidia/pytorch:25.08-py3`)
for Blackwell GPU support (sm_110). The BioCLIP weights (~3.5 GB) are
pre-downloaded at build time and baked into the image — no internet needed
at runtime.

Build and deploy via `pluginctl` (ECR portal builds are broken fleet-wide due
to a runc /proc/acpi bug; use local build + side-load instead):

```bash
cd ~/BISONN/plugin/
sudo pluginctl build .
sudo pluginctl run --name bisonn-test \
  --selector resource.gpu=true \
  <image-ref> -- --stream bottom_camera --continuous N
sudo pluginctl logs bisonn-test
```

## Camera Credentials

For Reolink cameras, credentials go in the URL query string
(`&user=USER&password=PASS`), NOT HTTP basic auth. For scheduled jobs, use
`--env-from <credsfile>` with `CAMERA_USER` / `CAMERA_PASSWORD` environment
variables. Never hardcode credentials in job YAML or command-line arguments.

## Using This Plugin

BISONN runs on any Sage/Waggle Thor node with a connected camera. It does not
require internet access at runtime — all model weights are baked into the
container image at build time.

**Quick test (batch mode, no camera needed):**

```bash
# Build the image (from the plugin/ directory)
cd ~/BISONN/plugin/
sudo pluginctl build .

# Run a one-shot test against the bundled sample images
sudo pluginctl run --name bisonn-test \
  --resource limit.memory=16Gi,request.memory=4Gi \
  <image-ref> -- --image-dir /app/test-images --continuous N

# View the predictions in the logs
sudo pluginctl logs bisonn-test
```

**Live camera (continuous):**

```bash
sudo pluginctl run --name bisonn-live \
  --resource limit.memory=16Gi,request.memory=4Gi \
  --selector resource.gpu=true \
  <image-ref> -- --stream bottom_camera --continuous Y --interval 30
```

**HTTP snapshot (Reolink cameras):**

```bash
sudo pluginctl run --name bisonn-cam \
  --resource limit.memory=16Gi,request.memory=4Gi \
  <image-ref> -- \
  --snapshot-url "http://CAMERA_IP/cgi-bin/api.cgi?cmd=Snap&user=USER&password=PASS" \
  --continuous Y --interval 30
```

**Reading the output:**
- `biotic.interaction.bird_mobbing` — 1 (mobbing detected) or 0 (none)
- `biotic.species.bird` — top-3 species with cosine similarity scores
- Annotated images are uploaded with behavior + species overlays for visual review

**Extending the species list:** Edit the `SPECIES_PROMPTS` list in `app.py` to
add or remove species. Prompts use the format `"a photo of a <species>"` to
match BioCLIP's training captions. Rebuild the image after changing the list.

**Retraining the behavior classifier:** Use `scripts/train_and_evaluate.py`
with your own labeled images. Place images in `data/labeled/mobbing/` and
`data/labeled/none/`, regenerate embeddings with `scripts/extract_embeddings.py`,
then train. Replace `models/svm.joblib` in the plugin directory and rebuild.

## License

MIT
