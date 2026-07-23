# BISONN Implementation Plan
## Biotic Interactions with Sage Observations using Neural Networks

---

## 1. Project Goal

Build a proof-of-concept edge application on a Sage/Waggle Thor node that quantifies
**biotic interactions** (e.g., bird feeding young) from camera images using BioCLIP embeddings + lightweight classification heads.

### Scope
- **Images only** (no video or audio in phase 1)
- **Bird images only** — all training and inference images are of birds
- **Two biotic interaction types**:
  - **Feeding young**: adult bird bringing food to nestlings or fledglings
  - **Mobbing**: a group of birds harassing or distracting a predator/threat
- **Single 3-class model**: `feeding_young` / `mobbing` / `none`
  - `none` = bird images with no interaction (background class for rejection)
- **Compare**: raw BioCLIP zero-shot retrieval vs. trained classification heads
- **Evaluate**: INQUIRE Benchmark (if usable for interaction queries) + custom labeled set

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
 │  │ Snapshot │    │ Encode   │    │  (512-dim)  │             │
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

**Goal**: Assemble a labeled image dataset of bird images for the 3-class scheme
(`feeding_young` / `mobbing` / `none`).

**Note**: All images must be of birds. The `none` class serves as the background
rejection class (birds with no visible interaction).

#### 1A. Evaluate the INQUIRE Benchmark

The INQUIRE benchmark (`sagecontinuum/INQUIRE-Benchmark-small` on HuggingFace)
is a text-to-image retrieval dataset with 20K images, 250 expert queries, and
relevance labels. Each image has iNat24 species metadata + GPS coordinates.

INQUIRE is unlikely to have strong coverage for "feeding young" or "mobbing"
since those are specific behaviors, but may contain some bird images we can
use as the `none` class or for mining.

- [ ] Load and inspect INQUIRE
  ```python
  from datasets import load_dataset
  ds = load_dataset("sagecontinuum/INQUIRE-Benchmark-small", split="validation")
  print(ds.features)
  print(ds[0])  # look at fields: image, query, relevant, category, species_name...
  ```
- [ ] Filter INQUIRE queries for bird-related content (any bird images we can use)
- [ ] **Limitation**: INQUIRE is a *retrieval* benchmark, not a *classification* one.
  It's great for evaluating raw BioCLIP zero-shot retrieval but doesn't directly
  provide "interaction type" labels for supervised training. We'll use it for:
  1. Zero-shot retrieval baseline (raw BioCLIP text→image matching)
  2. Mining candidate bird images for our `none` class

#### 1B. Assemble Custom Labeled Interaction Set

For the PoC, we need bird images labeled with **interaction type** (feeding_young,
mobbing, none). Likely data sources:
1. **HuggingFace datasets**: search for bird behavior / interaction datasets
2. **iNaturalist**: mine for images showing feeding or mobbing behavior
3. **Manual web search**: find labeled images of "bird feeding young" and "bird mobbing"
4. **Sage node camera data**: existing Sage camera images of outdoor scenes with birds
5. **INQUIRE images**: pooled bird images for the `none` class

**Recommended for PoC**: ~100-200 images per class:
  - `feeding_young`: adult bird with food, at nest with nestlings, etc.
  - `mobbing`: multiple birds gathering around / harassing a predator
  - `none`: solitary birds, perched birds, flying birds — no interaction

- [ ] Set up labeling directory structure
  ```
  ~/BISONN/data/
    raw/               # source images (all birds)
    labeled/
      feeding_young/    # bird feeding young (adult + nestlings/fledglings)
      mobbing/           # birds mobbing a predator
      none/              # birds with no interaction
    inquire_subset/    # bird images pulled from INQUIRE for `none` class
  ```
- [ ] Download / curate ~100-200 images per class (bird images only)
- [ ] Create a CSV manifest: `image_path, label, interaction_type, source`
- [ ] Train/test split: 80/20, stratified by class

---

### Phase 2: BioCLIP Embedding Extraction

**Goal**: Generate raw BioCLIP embeddings for all labeled images.

- [ ] Write `scripts/extract_embeddings.py`
  ```python
  # Pseudocode:
  import open_clip, torch, numpy as np
  from PIL import Image
  from pathlib import Path
  import json

  # Load BioCLIP 2.5 Huge (ViT-H/14, 1024-dim embeddings, ~1B params)
  # Run inside nvcr.io/nvidia/pytorch:25.08-py3 container for GPU access
  model, _, preprocess = open_clip.create_model_and_transforms(
      "hf-hub:imageomics/bioclip-2.5-vith14"
  )
  model.eval()
  if torch.cuda.is_available():
      model = model.cuda()

  def embed_image(image_path):
      img = preprocess(Image.open(image_path).convert("RGB"))
      if torch.cuda.is_available():
          img = img.cuda()
      with torch.no_grad():
          emb = model.encode_image(img.unsqueeze(0))
      return emb.cpu().numpy().flatten()

  # Embed all labeled images, save to .npy
  manifest = json.load(open("data/manifest.json"))
  embeddings = []
  labels = []
  for entry in manifest:
      emb = embed_image(entry["image_path"])
      embeddings.append(emb)
      labels.append(entry["label"])
  np.save("data/embeddings.npy", np.array(embeddings))
  np.save("data/labels.npy", np.array(labels))
  ```
- [ ] Extract embeddings for all labeled images
- [ ] Save embeddings + labels as `.npy` files
- [ ] Also extract text embeddings for interaction-type prompts (zero-shot baseline)
  ```python
  tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip-2.5-vith14")
  prompts = [
      "a photo of a bird feeding its young",
      "a photo of a bird bringing food to nestlings",
      "a photo of a bird at a nest with chicks",
      "a photo of birds mobbing a predator",
      "a photo of a flock of birds harassing a threat",
      "a photo of a solitary bird perched",
      "a photo of a bird flying with no interaction",
      "a photo of a bird with no visible interaction",
  ]
  with torch.no_grad():
      text_tokens = tokenizer(prompts)
      if torch.cuda.is_available():
          text_tokens = text_tokens.cuda()
      text_embs = model.encode_text(text_tokens)
  np.save("data/text_embeddings.npy", text_embs.cpu().numpy())
  ```

---

### Phase 3: Train Classification Heads

**Goal**: Train simple classifiers on the frozen BioCLIP embeddings and compare them.

#### 3A. Baseline: Zero-Shot Retrieval (no training)

- [ ] Compute zero-shot predictions using cosine similarity between image embeddings
  and text prompt embeddings
- [ ] Evaluate: accuracy, precision, recall, F1, mAP
- [ ] This is the "raw embeddings" baseline — no supervised training needed

#### 3B. Classification Head 1: Logistic Regression (linear probe)

- [ ] Train on the train split of labeled embeddings
  ```python
  from sklearn.linear_model import LogisticRegression
  clf = LogisticRegression(max_iter=1000)
  clf.fit(X_train, y_train)
  ```
- [ ] Evaluate on test split
- [ ] Save model weights: `scikit-learn` `joblib.dump`

#### 3C. Classification Head 2: k-Nearest Neighbors

- [ ] Train kNN on embeddings (trivial — just store reference embeddings)
  ```python
  from sklearn.neighbors import KNeighborsClassifier
  knn = KNeighborsClassifier(n_neighbors=5, metric="cosine")
  knn.fit(X_train, y_train)
  ```
- [ ] Evaluate on test split

#### 3D. Classification Head 3: Small MLP (optional, if time allows)

- [ ] Define and train a 2-layer MLP on embeddings using PyTorch
  ```python
  import torch.nn as nn
  mlp = nn.Sequential(
      nn.Linear(1024, 256),  # 1024 = BioCLIP 2.5 Huge embedding dim
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(256, 3)  # 3 classes: feeding_young, mobbing, none
  )
  # train with cross-entropy loss, Adam, ~20-50 epochs
  ```
- [ ] Evaluate on test split

#### 3E. Evaluation Dashboard

- [ ] Write `scripts/evaluate.py` to produce a comparison table:
  ```
  ┌─────────────────────┬──────────┬──────────┬──────────┬──────────┐
  │ Model               │ Accuracy │ F1       │ mAP      │ Notes    │
  ├─────────────────────┼──────────┼──────────┼──────────┼──────────┤
  │ Zero-shot retrieval │   XX%    │   XX%    │   XX%    │ baseline │
  │ Logistic regression │   XX%    │   XX%    │   XX%    │ linear   │
  │ kNN (k=5, cosine)   │   XX%    │   XX%    │   XX%    │ nonparam │
  │ MLP (256 hidden)    │   XX%    │   XX%    │   XX%    │ nonlinear│
  └─────────────────────┴──────────┴──────────┴──────────┴──────────┘
  ```
- [ ] Save a confusion matrix for the best model
- [ ] Generate a comparison plot (bar chart or radar)

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
              str(prediction),  # one of: feeding_young, mobbing, none
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
  keywords: "biotic,interaction,pollinator,bioclip"
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

1. How well do raw BioCLIP embeddings separate the three classes
   (feeding_young, mobbing, none) without any training? (zero-shot baseline)
2. How much does a simple linear probe improve over zero-shot?
3. Does a non-linear head (MLP) add value over linear, or are the embeddings
   already linearly separable?
4. Is the approach generalizable to additional interaction types beyond
   feeding young and mobbing?
5. Which interaction type is easier to detect — feeding young or mobbing?

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
