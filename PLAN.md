# BISONN Implementation Plan
## Biotic Interactions with Sage Observations using Neural Networks

---

## 1. Project Goal

Build a proof-of-concept edge application on a Sage/Waggle Thor node that quantifies
**biotic interactions** (bird mobbing behavior) from camera images using BioCLIP embeddings + lightweight classification heads.

### Scope
- **Images only** (no video or audio in phase 1)
- **Bird images only** — all training and inference images are of birds
- **Single interaction type**: **mobbing** (a group of birds harassing or distracting a predator/threat)
- **Binary classification**: `mobbing` / `none`
  - `none` = bird images with no mobbing interaction (background class for rejection)
- **Compare**: raw BioCLIP zero-shot retrieval vs. trained classification heads
- **Evaluate**: custom labeled set (iNaturalist + Wikimedia Commons + personal photos)

### Hardware (this Thor: sgt-thor-1423125006073-H021)
- JetPack R38.2.1, aarch64, 128GB unified memory
- Podman + Docker available, k3s/WES stack
- No torch/transformers installed in system Python yet (will use venv or container)

---

## 2. Architecture Overview

```
                         BISONN Plugin Pipeline
 ┌──────────────────────────────────────────────────────────────┐
 │                                                              │
 │  Training Phase (offline, on Thor host or dev machine)       │
 │  ┌──────────┐    ┌───────────┐    ┌────────────┐             │
 │  │ Labeled  │───►│ BioCLIP   │───►│ Embeddings │             │
 │  │ Images   │    │ Encoder   │    │  (.npy)    │             │
 │  └──────────┘    └───────────┘    └─────┬──────┘             │
 │                                         │                    │
 │                              ┌──────────▼──────────┐         │
 │                              │  Classification     │         │
 │                              │  Head Training      │         │
 │                              │  (sklearn / torch)  │         │
 │                              └──────────┬──────────┘         │
 │                                         │                    │
 │                              ┌──────────▼──────────┐         │
 │                              │  Trained Head       │         │
 │                              │  (model weights)    │         │
 │                              └─────────────────────┘         │
 │                                                              │
 │  Inference Phase (Sage plugin, runs in WES pod)              │
 │  ┌──────────┐    ┌───────────┐    ┌────────────┐             │
 │  │ Camera   │───►│ BioCLIP   │───►│ Embedding  │             │
 │  │ Snapshot │    │ Encode    │    │  (512-dim) │             │
 │  └──────────┘    └───────────┘    └─────┬──────┘             │
 │                                    │                         │
 │                    ┌───────────────▼──────────────┐          │
 │                    │ Classification Head          │          │
 │                    │ (linear / kNN / small MLP)   │          │
 │                    └───────────────┬──────────────┘          │
 │                                    │                         │
 │                    ┌───────────────▼──────────────┐          │
 │                    │ plugin.publish()             │          │
 │                    │ plugin.upload_file()         │          │
 │                    │ → Beehive → Sage data API    │          │
 │                    └──────────────────────────────┘          │
 │                                                              │
 └──────────────────────────────────────────────────────────────┘
```

---

## 3. Phased Plan

### Phase 0: Environment Setup

**Goal**: Working Python environment on the Thor with BioCLIP runnable.

- [ ] Create a Python venv at `~/BISONN/venv`
  ```bash
  python3 -m venv ~/BISONN/venv
  source ~/BISONN/venv/bin/activate
  pip install --upgrade pip
  ```
- [ ] Install core dependencies
  ```bash
  pip install torch torchvision torchaudio  # aarch64 CUDA wheels
  pip install open_clip_torch timm pillow scikit-learn numpy pandas
  pip install pybioclip  # convenience wrapper for BioCLIP
  pip install datasets huggingface_hub  # for INQUIRE + model downloads
  pip install matplotlib seaborn  # for evaluation plots
  ```
- [ ] Install pywaggle (for Sage plugin development)
  ```bash
  pip install "pywaggle[all]==0.56.0"
  ```
- [ ] Verify BioCLIP 2.5 loads on CPU (host venv, CUDA disabled)
  ```python
  # CUDA_VISIBLE_DEVICES='' forces CPU — avoids the Blackwell kernel hang
  # GPU is only needed later for the deployed plugin (Phase 5)
  import os
  os.environ['CUDA_VISIBLE_DEVICES'] = ''
  import open_clip
  model, _, preprocess = open_clip.create_model_and_transforms(
      "hf-hub:imageomics/bioclip-2.5-vith14"
  )
  # Verified: model loads in ~42s, 1024-dim embeddings, ~1.7s/image on CPU
  ```
- [x] Phase 0 status (as of 2026-07-23):
  - venv at ~/BISONN/venv with all packages installed
  - BioCLIP 2.5 Huge loads on CPU (CUDA disabled), ~1.7s/image inference
  - app.py verification script passes
  - requirements.txt written
- [ ] ~~Verify GPU is accessible~~ (deferred to Phase 5 — deployment)
  **Note**: Dev work (Phases 0-3) runs CPU-only on the host venv with
  `CUDA_VISIBLE_DEVICES=''`. GPU is only needed for the deployed plugin.
  Generic PyPI torch (2.13.0+cu130) hangs on any CUDA call (no Blackwell
  sm_110 kernels). The NVIDIA container `nvcr.io/nvidia/pytorch:25.08-py3`
  (torch 2.8, sm_110/sm_121) is pulled and ready for Phase 5 plugin building.
  GPU access at deploy time goes through `pluginctl --selector resource.gpu=true`
  (k3s pod path), not Docker `--runtime=nvidia`.

---

### Phase 1: Data Acquisition & Labeling

**Goal**: Assemble a labeled image dataset of bird images for the binary scheme
(`mobbing` / `none`).

**Note**: All images must be of birds. The `none` class serves as the background
rejection class (birds with no mobbing interaction).

#### 1A. Data Sources

- [x] **iNaturalist** — CC-licensed bird photos via API (text search + taxon filter)
  - `none` class: searched "perched bird", "bird flying" → ~150 images
  - `mobbing` class: searched "mobbing", "attacking hawk", "birds harassing", etc.
    filtered to Aves → ~36 images
- [x] **Wikimedia Commons** — CC-licensed bird photos via API
  - `mobbing` class: searched "bird mobbing", "crow mobbing hawk", etc. → ~200 images
- [x] **Personal photos** (Kyle Lima) — ~1500 personal bird photos
  - `none` class: bulk of personal photos (birds without mobbing)
  - `mobbing` class: a few personal mobbing photos
- [ ] ~~INQUIRE Benchmark~~ — not used (limited mobbing coverage)

#### 1B. Data Cleanup

- [x] Downloaded 404 images from iNaturalist (150 feeding_young, 104 mobbing, 150 none)
- [x] Downloaded 236 mobbing images (36 iNaturalist + 200 Wikimedia Commons)
- [x] Focused on mobbing only — removed feeding_young class
- [x] User manually cleaned mobbing folder (98 images after review)
- [x] BioCLIP zero-shot classification cleaned non-bird images from `none` folder
  (removed 50 images flagged as bones, droppings, mammals, plants, insects)
- [x] 36 images moved from mobbing to none (reclassified by user)
- [x] Manifest synced: /home/kylelima21/BISONN/data/manifest_unified.csv
- [x] Add personal photos (~1485 images) to dataset — done
- [x] Re-sync manifest after personal photos are added — done

#### 1C. Final Dataset (with personal photos)

| Class    | Count | Sources                                        |
|----------|-------|------------------------------------------------|
| mobbing  | 101   | Wikimedia Commons (98) + personal (3)          |
| none     | 1589  | iNaturalist (78) + WMC (29) + personal (1482)  |
| TOTAL    | 1690  |                                                |

Personal photos use Flickr-style descriptive filenames (e.g.,
`none_accipiter-cooperii---coopers-hawk_27633331108_o.jpg`).
One `.mov` video in `none/` was excluded from the image manifest.
Class imbalance is significant: 101 mobbing vs 1589 none — will
need class-weighted training or balanced sampling in Phase 3.

#### 1D. Directory Structure

```
~/BISONN/data/
  labeled/
    mobbing/          # birds mobbing a predator/threat
    none/             # birds with no mobbing interaction
  manifest_unified.csv  # unified manifest with attribution + license
  manifest_mobbing.csv # mobbing-only manifest
  manifest.csv         # original manifest (iNaturalist only)
```

- [ ] Train/test split: 80/20, stratified by class

---

### Phase 2: BioCLIP Embedding Extraction

**Goal**: Generate BioCLIP 2.5 embeddings for all labeled images and zero-shot
text prompts, saved as portable artifacts for downstream training and evaluation.

- [x] Copy `embedding_bundles.py` from the Sage 2026 BioCLIP workshop
  (`sage-summer-2026-bioclip-main/notebooks/embedding_bundles.py`) into
  `scripts/`. Provides `EmbeddingBundle` (ordered IDs, L2-normalized features,
  producer manifest) with save/load + sha256 integrity — same evaluation
  boundary pattern the workshop uses.
- [x] Write `scripts/extract_embeddings.py`
  - Reads `data/manifest_unified.csv` (1690 images)
  - Loads BioCLIP 2.5 Huge (ViT-H/14, 1024-dim) on CPU
    (`CUDA_VISIBLE_DEVICES=''` — PyPI torch lacks Blackwell sm_110 kernels)
  - Encodes all images in batches of 4, ~1.3s/image, ~37 min total
  - Saves an `EmbeddingBundle` (.npz) with L2-normalized features + producer
    manifest, plus labels (.npy) and class name mapping (.json)
  - Round-trip verification on load (`np.allclose`, atol=1e-6 — re-normalization
    of unit vectors in float32 introduces ~7e-9 drift, so `array_equal` is too
    strict)
- [x] Write `scripts/extract_text_embeddings.py`
  - Encodes 16 hand-authored behavior prompts (8 mobbing, 8 none) through
    BioCLIP 2.5's text encoder with `normalize=True`
  - Saves a text `EmbeddingBundle` (.npz) + prompt metadata (.json)
  - "a photo of..." framing (BioCLIP was trained on photo captions)
- [x] Extract image embeddings for all 1690 labeled images
- [x] Extract text embeddings for 16 zero-shot behavior prompts
- [x] Verify all artifacts (shape, L2 normalization, no NaNs, round-trip load)

#### Phase 2 Artifacts

| File | Shape | Contents |
|------|-------|----------|
| `data/embeddings_bisonn.npz` | (1690, 1024) | Image embeddings (L2-normalized) + producer manifest |
| `data/labels_bisonn.npy` | (1690,) | Integer labels (0=mobbing, 1=none) |
| `data/label_names.json` | — | Class name mapping |
| `data/text_embeddings_bisonn.npz` | (16, 1024) | Text prototype embeddings (L2-normalized) + manifest |
| `data/behavior_prompts.json` | — | Prompt text + class mapping |

- [x] Phase 2 status (as of 2026-07-24):
  - All 1690 images encoded on CPU in ~37 min (~1.3s/image, batch_size=4)
  - 16 text prototypes encoded (8 mobbing, 8 none)
  - All bundles verified: correct shape, L2-normalized, no NaNs/inf, round-trip load
  - Workshop reference: `sage-summer-2026-bioclip-main/` (peromyscus.ipynb lessons
    1-5 — embedding bundles, zero-shot prototypes, few-shot SVM — adapted for
    BISONN's binary behavior task)

---

### Phase 3: Train Classification Heads (BioCLIP 2.5)

**Goal**: Train simple classifiers on frozen BioCLIP 2.5 embeddings and compare.

#### 3A. Baseline: Zero-Shot Retrieval (no training)

- [x] Compute zero-shot predictions using cosine similarity between image
  embeddings and text prompt embeddings
- [x] Two voting schemes: averaged class prototypes and best-of-prompts (max
  individual prompt score)
- [x] Evaluate: accuracy, precision, recall, F1 per class
- [x] Result: best-of-prompts 44.9% accuracy, 0.375 macro-F1 (poor — BioCLIP
  was trained on taxonomic captions, not behavior descriptions; over-predicts
  mobbing heavily)

#### 3B. Classification Head 1: Logistic Regression (linear probe)

- [x] Train with `class_weight='balanced'`, `max_iter=2000`
- [x] Result: 97.0% accuracy, 0.883 macro-F1 (mobbing F1=0.783, none F1=0.984)
- [x] Save model: `data/models/logistic.joblib`

#### 3C. Classification Head 2: Linear SVM

- [x] Train with `kernel='linear'`, `class_weight='balanced'`
- [x] Result: 98.5% accuracy, 0.935 macro-F1 (mobbing F1=0.878, none F1=0.992)
- [x] **Best model** — saved as `data/models/svm.joblib`

#### 3D. Classification Head 3: k-Nearest Neighbors

- [x] k=5, cosine metric, distance-weighted voting
- [x] Result: 97.0% accuracy, 0.845 macro-F1 (mobbing recall only 60% — misses
  8/20 mobbing test images)
- [x] Model saved alongside others for comparison

#### 3E. Classification Head 4: Small MLP (optional)

- [ ] ~~MLP on CPU~~ — skipped: torch 2.13+cu130 hangs on even trivial CPU
  matmul due to Blackwell CUDA init. Can be run inside the NVIDIA container
  (`nvcr.io/nvidia/pytorch:25.08-py3`) by setting `BISONN_ENABLE_MLP=1`.
  The MLP code is in `run_mlp()` — ready to run in the right container.

#### 3F. Evaluation Dashboard

- [x] Comparison table:
  ```
  Method           Accuracy  MacroF1  Mob P  Mob R  Mob F1  None P None R None F1
  Zero-shot           0.449    0.375  0.088  0.881  0.160  0.982  0.421  0.589
  Logistic Reg        0.970    0.883  0.692  0.900  0.783  0.994  0.975  0.984
  Linear SVM          0.985    0.935  0.857  0.900  0.878  0.994  0.991  0.992
  kNN (k=5)           0.970    0.845  0.857  0.600  0.706  0.975  0.994  0.984
  ```
- [x] Confusion matrix saved: `data/results/best_model_confusion.png` (Linear SVM)
- [x] Comparison bar chart saved: `data/results/comparison_bar.png`
- [x] Full evaluation report: `data/results/evaluation_report.txt`

#### Phase 3 Artifacts

| File | Contents |
|------|----------|
| `scripts/train_and_evaluate.py` | Training + evaluation script (all methods) |
| `data/models/svm.joblib` | Best model (Linear SVM, macro-F1=0.935) |
| `data/models/logistic.joblib` | Logistic regression (deployment candidate — simplest) |
| `data/results/evaluation_report.txt` | Full metrics + confusion matrices + classification reports |
| `data/results/best_model_confusion.png` | Confusion matrix for Linear SVM |
| `data/results/comparison_bar.png` | Bar chart comparing all 4 methods |

#### Key Findings

- **Zero-shot is poor** (44.9% acc): BioCLIP's text encoder was trained on
  taxonomic captions, not behavior descriptions. Behavior prompts do not
  produce useful zero-shot prototypes.
- **Supervised heads are strong**: Even a linear probe (logistic regression)
  achieves 90% mobbing recall. The linear SVM is the clear winner with 0.935
  macro-F1 and 85.7% mobbing precision.
- **Class weighting is essential**: Without `class_weight='balanced'`, the
  1:16 imbalance causes the classifier to trivially predict "none" every time.
- **kNN has low mobbing recall** (60%): The mobbing cluster is sparse in
  embedding space; 8/20 mobbing test images have their 5 nearest neighbors
  dominated by "none" images. Distance weighting helps but doesn't fix this.
- **MLP deferred**: torch CPU matmul hangs on this host (Blackwell CUDA init
  bug). Can be run in an NVIDIA container. The linear SVM's 0.935 macro-F1
  is likely hard to beat with a 2-layer MLP on 1024-dim embeddings anyway.

---

### Phase 3-DINOv3: DINOv3 Embedding Extraction + Head Training

**Goal**: Repeat the Phase 2-3 pipeline with DINOv3 replacing BioCLIP, then
compare all models head-to-head. DINOv3 is a self-supervised vision transformer
(no text encoder, no CLIP-style training) — it cannot do zero-shot text-image
retrieval, but its visual features may be stronger for supervised classification.

#### DINOv3 Model Selection

Two sizes, using timm (non-gated HF repos):

| Model | timm name | HF repo | Embedding dim | Params | Input size |
|-------|-----------|---------|---------------|--------|------------|
| DINOv3 Large | `vit_large_patch16_dinov3_qkvb` | `timm/vit_large_patch16_dinov3_qkvb.lvd1689m` | 1024 | ~300M | 256x256 |
| DINOv3 Small | `vit_small_patch16_dinov3_qkvb` | `timm/vit_small_patch16_dinov3_qkvb.lvd1689m` | 384 | ~22M | 256x256 |

- Large matches BioCLIP 2.5's 1024-dim for a direct head-to-head comparison
- Small tests whether a lightweight edge model suffices for this task
- Both use the `qkvb` variant (qkv-bias fused, the recommended DINOv3 form)
- DINOv3 uses ImageNet normalization (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
- DINOv3 has NO text encoder — zero-shot retrieval is not applicable

#### Phase 2b: DINOv3 Embedding Extraction

- [ ] Write `scripts/extract_embeddings_dinov3.py`
  - Accept model name as argument (large or small)
  - Use `timm.create_model(name, pretrained=True, num_classes=0)`
  - Use `timm.data.resolve_model_data_config()` + `timm.data.create_transform()` for preprocessing
  - Extract pooled features via `model.forward_head(model.forward_features(x), pre_logits=True)`
  - L2-normalize and save as `EmbeddingBundle` (.npz) — same format as BioCLIP
  - Labels are reused from Phase 2 (same 1690 images, same manifest)
  - Outputs: `data/embeddings_dinov3_large.npz`, `data/embeddings_dinov3_small.npz`
- [ ] Extract embeddings for DINOv3 Large (1024-dim) — ~300M params on CPU
- [ ] Extract embeddings for DINOv3 Small (384-dim) — ~22M params on CPU, faster
- [ ] Verify both bundles (shape, L2-normalized, no NaNs, round-trip load)

#### Phase 3b: DINOv3 Classification Head Training

- [ ] Run `scripts/train_and_evaluate.py` adapted for DINOv3 embeddings
  - Same methods: logistic regression, linear SVM, kNN (class-weighted)
  - No zero-shot (DINOv3 has no text encoder)
  - Same 80/20 stratified split, same seed (42) for comparability
  - MLP skipped (same torch CPU issue)
- [ ] Save models: `data/models/dinov3_large_svm.joblib`, etc.
- [ ] Save results: `data/results/evaluation_report_dinov3_large.txt`, etc.

#### Phase 3c: Cross-Model Comparison

- [ ] Write `scripts/compare_models.py`
  - Load all evaluation results (BioCLIP 2.5, DINOv3 Large, DINOv3 Small)
  - Produce a unified comparison table:
    ```
    Backbone      Head         Accuracy  MacroF1  Mob F1  None F1
    BioCLIP 2.5   Linear SVM     0.985    0.935   0.878   0.992
    DINOv3 Large  Linear SVM     ?        ?       ?       ?
    DINOv3 Small  Linear SVM     ?        ?       ?       ?
    ...           LogReg         ...      ...     ...     ...
    ```
  - Bar chart comparing macro-F1 across backbones and heads
  - Confusion matrices side-by-side for best head per backbone
  - Discussion: which backbone wins? Does DINOv3's self-supervised training
    beat BioCLIP's bio-taxonomic pretraining for behavior classification?
- [ ] Save: `data/results/cross_model_comparison.txt`, `data/results/cross_model_comparison.png`

#### Key Questions to Answer

1. Does DINOv3 (self-supervised, general) outperform BioCLIP 2.5 (biology-taxed)
   on a BEHAVIOR task? BioCLIP's pretraining is taxonomic — it may not help
   for mobbing vs. none.
2. Does the larger DINOv3 (1024-dim, 300M params) justify the extra cost over
   the small variant (384-dim, 22M params) for edge deployment?
3. Is BioCLIP's zero-shot text retrieval handicap (44.9% accuracy) relevant
   when supervised heads achieve >97% regardless of backbone?
4. Which backbone+head combination is the best deploy choice for a Sage/Waggle
   edge plugin considering accuracy, model size, and inference latency?

---

### Phase 4: Sage Plugin Packaging

**Goal**: Package the inference pipeline as a deployable Sage/Waggle plugin.

- [ ] Create plugin structure in `~/BISONN/plugin/`
  ```
  plugin/
    app.py              # main inference loop
    Dockerfile          # container definition
    requirements.txt    # Python dependencies
    sage.yaml           # Sage metadata
    models/             # baked-in model weights (BioCLIP + classifier head)
      bioclip_weights/  # BioCLIP safetensors
      classifier.joblib # trained classification head
    overview.md        # plugin documentation
    ecr-meta/
      ecr-science-description.md
  ```

- [ ] Write `app.py` — the Sage plugin
  ```python
  # Pseudocode for the plugin entry point
  import json, time, os
  import numpy as np
  import open_clip, torch, joblib
  from PIL import Image
  from waggle.plugin import Plugin
  from waggle.data.vision import Camera

  # Load BioCLIP 2.5 Huge (ViT-H/14, 1024-dim embeddings)
  # Weights baked into image at build time (HF download at build, offline at runtime)
  model, _, preprocess = open_clip.create_model_and_transforms(
      "hf-hub:imageomics/bioclip-2.5-vith14"
  )
  classifier = joblib.load("/app/models/classifier.joblib")

  def predict_interaction(image_data):
      img = preprocess(Image.fromarray(image_data).convert("RGB"))
      with torch.no_grad():
          emb = model.encode_image(img.unsqueeze(0))
      label = classifier.predict(emb.cpu().numpy())[0]
      return label

  def main():
      with Plugin() as plugin:
          camera = Camera(os.environ.get("CAMERA_URL", "file:///app/example.jpg"))
          image = camera.snapshot()
          prediction = predict_interaction(image.data)
          plugin.publish(
              "biotic.interaction.type",
              str(prediction),  # one of: mobbing, none
              timestamp=image.timestamp,
              meta={"model": "bioclip+linear",
                    "confidence": str(confidence)}
          )
          # Optionally upload the annotated image
          plugin.upload_file(image.path,
              timestamp=image.timestamp,
              meta={"interaction": str(prediction)})

  if __name__ == "__main__":
      main()
  ```

- [ ] Write `Dockerfile` (NVIDIA base for Blackwell GPU support)
  ```dockerfile
  # NVIDIA PyTorch 25.08 — CUDA 13.0, PyTorch 2.8, Python 3.12
  # Supports Thor (Blackwell sm_110). Generic PyPI torch lacks these kernels.
  FROM nvcr.io/nvidia/pytorch:25.08-py3

  WORKDIR /app
  COPY requirements.txt .

  # CRITICAL: Freeze base image packages so pip cannot replace them.
  # The NVIDIA base ships torch/torchvision/numpy compiled for Blackwell GPUs.
  # pip install of open_clip etc. will try to pull generic PyPI versions
  # that LACK GPU kernels or break ABI. Freeze them as constraints.
  RUN pip install --no-cache-dir --upgrade pip && \
      TORCH_VER=$(python3 -c "import torch; print(torch.__version__)") && \
      TV_VER=$(python3 -c "import torchvision; print(torchvision.__version__)") && \
      NP_VER=$(python3 -c "import numpy; print(numpy.__version__)") && \
      echo "Freezing: torch==${TORCH_VER} torchvision==${TV_VER} numpy==${NP_VER}" && \
      printf "torch==${TORCH_VER}\ntorchvision==${TV_VER}\nnumpy==${NP_VER}\n" > /tmp/constraints.txt && \
      pip install --no-cache-dir -c /tmp/constraints.txt -r requirements.txt

  # Pre-download BioCLIP 2.5 weights at build time (edge nodes may lack internet)
  RUN python3 -c "import open_clip; open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2.5-vith14')"

  # Bake classifier weights
  COPY models/ /app/models/

  # Copy app code LAST (small, changes often)
  COPY app.py .
  ENTRYPOINT ["python3", "app.py"]
  ```

- [ ] Write `requirements.txt` (torch/torchvision/numpy come from the NVIDIA base — frozen, not pip-installed)
  ```txt
  # torch, torchvision, numpy are FROZEN from the NVIDIA base image (do NOT pin here)
  pywaggle[all]==0.56.0
  open_clip_torch>=2.20.0
  timm>=1.0.15
  scikit-learn
  joblib
  pillow
  ```

- [ ] Write `sage.yaml`
  ```yaml
  name: "bisonn"
  version: "0.1.0"
  description: "Biotic Interactions with Sage Observations using Neural Networks"
  keywords: "biotic,interaction,mobbing,bioclip,bird"
  authors: "Kyle Lima"
  collaborators: ""
  funding: ""
  license: "MIT"
  homepage: "https://github.com/<your-org>/BISONN"
  source:
    architectures:
      - "linux/arm64"
  ```

---

### Phase 5: Build, Test, Deploy on Thor

**Goal**: Running plugin on the Thor node via `pluginctl`.

- [ ] Build the plugin image
  ```bash
  cd ~/BISONN/plugin/
  sudo pluginctl build .
  ```

- [ ] Run the plugin (one-shot test with sample image)
  ```bash
  sudo pluginctl run --name bisonn-test \
    --selector resource.gpu=true \
    <image-ref> -- --image /app/example.jpg
  ```

- [ ] Check logs and verify output
  ```bash
  sudo pluginctl logs bisonn-test
  ```

- [ ] Test with a live camera (if available on the node)
  ```bash
  # Set camera env vars (get credentials from instructor)
  sudo pluginctl run --name bisonn-live \
    --selector resource.gpu=true \
    -e CAMERA_URL="rtmp://<user>:<pass>@<camera-ip>/...?&user=&password=" \
    <image-ref>
  ```

- [ ] Clean up
  ```bash
  sudo pluginctl rm bisonn-test
  ```

**Pitfalls to watch for**:
- Camera URLs: Reolink cameras need query-param auth (`&user=&password=`), not
  HTTP basic auth (basic auth makes ffmpeg fail with exit 187)
- `plugin.publish()` meta values must ALL be strings — never pass floats/ints
- `timestamp` must be int nanoseconds: `timestamp=image.timestamp` (from Camera)
- BioCLIP model download: bake weights into the Docker image at build time; don't
  rely on runtime download (edge nodes may not have internet)
- ECR portal builds are broken (runc /proc/acpi bug fleet-wide). Use `pluginctl`
  for local builds on the Thor

---

## 4. Evaluation Summary

### Metrics to Compare

| Method               | Trained?  | What it tests                           |
|----------------------|-----------|----------------------------------------|
| Zero-shot retrieval  | No        | Raw BioCLIP text-image cosine sim      |
| Logistic regression  | Yes       | Linear separability of embeddings     |
| kNN (cosine)         | Yes       | Local structure of embedding space     |
| MLP (256 hidden)     | Yes       | Non-linear separability                |

### Key Questions to Answer

1. How well do raw BioCLIP embeddings separate the two classes
   (mobbing, none) without any training? (zero-shot baseline)
2. How much does a simple linear probe improve over zero-shot?
3. Does a non-linear head (MLP) add value over linear, or are the embeddings
   already linearly separable?
4. Is the approach generalizable to additional interaction types beyond mobbing?
5. How well does the classifier handle the class imbalance (few mobbing
   examples vs many none examples)?

---

## 5. Timeline (Camp- pace)

| Phase | Task                                   | Est. Time  | Dependency    |
|-------|----------------------------------------|------------|---------------|
| 0     | Environment setup (venv, BioCLIP)      | 30-60 min  | None          |
| 1     | Data acquisition & labeling            | 2-4 hrs    | Phase 0       |
| 2     | Embedding extraction                   | 30 min     | Phase 0, 1    |
| 3     | Train & evaluate classification heads  | 1-2 hrs    | Phase 2       |
| 4     | Sage plugin packaging                  | 1-2 hrs    | Phase 3       |
| 5     | Build, test, deploy on Thor            | 30-60 min  | Phase 4       |

**Total estimate**: ~6-10 hours of focused work

---

## 6. Key References

- **BioCLIP**: `imageomics/bioclip` (ViT-B/16, TreeOfLife-10M, 512-dim) on HuggingFace
- **BioCLIP-2**: `imageomics/bioclip-2` (SigLIP2, 430M params, 512-dim)
- **BioCLIP 2.5 Huge** (selected for BISONN): `imageomics/bioclip-2.5-vith14`
  (ViT-H/14, ~1B params, 1024-dim embeddings, 61.3% species accuracy, TreeOfLife-200M)
- **pybioclip**: Python wrapper for BioCLIP taxonomy classification
- **INQUIRE Benchmark**: `sagecontinuum/INQUIRE-Benchmark-small` (20K images, 250 expert queries)
- **NVIDIA PyTorch base**: `nvcr.io/nvidia/pytorch:25.08-py3` (CUDA 13.0, PyTorch 2.8, sm_110/sm_121)
- **Sage pluginctl**: build and test plugins on Thor nodes
- **pywaggle**: `Plugin`, `Camera`, `upload_file` — the Sage edge SDK
- **waggle/sage-thor-base**: Docker base image for Thor plugins

### Sage platform reference docs
- Plugin structure: app-tutorial template (in `~/app-tutorial/`)
- Deploy: `pluginctl` (camp guide in sage-waggle skill)
- ECR workaround: podman build + side-load (runc /proc/acpi bug fleet-wide)
- Data API: `sage-data-client` or Sage MCP server
