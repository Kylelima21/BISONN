# BISONN Implementation Plan
## Biotic Interactions with Sage Observations using Neural Networks

---

## 1. Project Goal

Build a proof-of-concept edge application on a Sage/Waggle Thor node that quantifies
**biotic interactions** (e.g., bird feeding young) from camera images using BioCLIP embeddings + lightweight classification heads.

### Scope
- **Images only** (no video or audio in phase 1)
- **Proof of concept**: 1-2 interaction types with labeled training data
- **Compare**: raw BioCLIP zero-shot retrieval vs. trained classification heads
- **Evaluate**: Inquire Benchmark (if usable for interaction queries) + custom labeled set

### Hardware (this Thor: sgt-thor-1423125006073-H021)
- JetPack R38.2.1, aarch64, 128GB unified memory
- Podman + Docker available, k3s/WES stack
- No torch/transformers installed in system Python yet (will use venv or container)

---

## 2. Architecture Overview

```
                         BISONN Plugin Pipeline
 ┌──────────────────────────────────────────────────────────────┐
 │                                                                │
 │  Training Phase (offline, on Thor host or dev machine)         │
 │  ┌──────────┐    ┌───────────┐    ┌────────────┐              │
 │  │ Labeled  │───►│ BioCLIP   │───►│ Embeddings │              │
 │  │ Images   │    │ Encoder  │    │  (.npy)    │              │
 │  └──────────┘    └───────────┘    └─────┬──────┘              │
 │                                         │                      │
 │                              ┌──────────▼──────────┐           │
 │                              │  Classification     │           │
 │                              │  Head Training      │           │
 │                              │  (sklearn / torch)  │           │
 │                              └──────────┬──────────┘           │
 │                                         │                      │
 │                              ┌──────────▼──────────┐           │
 │                              │  Trained Head       │           │
 │                              │  (model weights)    │           │
 │                              └─────────────────────┘           │
 │                                                                │
 │  Inference Phase (Sage plugin, runs in WES pod)                 │
 │  ┌──────────┐    ┌───────────┐    ┌────────────┐               │
 │  │ Camera   │───►│ BioCLIP   │───►│ Embedding  │               │
 │  │ Snapshot │    │ Encode   │    │  (512-dim)  │               │
 │  └──────────┘    └───────────┘    └─────┬──────┘               │
 │                                    │                            │
 │                    ┌───────────────▼──────────────┐             │
 │                    │ Classification Head          │             │
 │                    │ (linear / kNN / small MLP)  │             │
 │                    └───────────────┬──────────────┘             │
 │                                    │                            │
 │                    ┌───────────────▼──────────────┐             │
 │                    │ plugin.publish()             │             │
 │                    │ plugin.upload_file()         │             │
 │                    │ → Beehive → Sage data API    │             │
 │                    └──────────────────────────────┘             │
 │                                                                │
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
- [ ] Verify BioCLIP loads
  ```python
  import open_clip
  model, _, preprocess = open_clip.create_model_and_transforms(
      "hf-hub:imageomics/bioclip"
  )
  # or for BioCLIP-2:
  # model, _, preprocess = open_clip.create_model_and_transforms(
  #     "hf-hub:imageomics/bioclip-2"
  # )
  ```
- [ ] Verify GPU is accessible from the venv
  ```python
  import torch
  print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
  ```
  **Pitfall**: On Tegra/Thor, `torch.cuda.is_available()` may return False
  when run from the host Python without proper device permissions. If so,
  testing must happen inside a `pluginctl` pod (the WES stack provides
  `/dev/nvmap` and GPU device access). The training phase can still run
  on CPU if needed — BioCLIP inference is ~500ms/image on GPU, slower
  on CPU but acceptable for offline training on a small PoC dataset.

---

### Phase 1: Data Acquisition & Labeling

**Goal**: Assemble a labeled image dataset for 1-2 biotic interaction types.

#### 1A. Evaluate the INQUIRE Benchmark

The INQUIRE benchmark (`sagecontinuum/INQUIRE-Benchmark-small` on HuggingFace)
is a text-to-image retrieval dataset with 20K images, 250 expert queries, and
relevance labels. Each image has iNat24 species metadata + GPS coordinates.

- [ ] Load and inspect INQUIRE
  ```python
  from datasets import load_dataset
  ds = load_dataset("sagecontinuum/INQUIRE-Benchmark-small", split="validation")
  print(ds.features)
  print(ds[0])  # look at fields: image, query, relevant, category, species_name...
  ```
- [ ] Filter INQUIRE queries for biotic interaction relevance
  - INQUIRE queries are expert naturalists' text queries ("find images of X doing Y")
  - Some queries may describe interactions (e.g., "bird on flower", "insect pollinating")
  - These can be used as **text-based zero-shot retrieval baselines**
  - The `relevant` field (0/1) provides ground truth for retrieval evaluation
- [ ] **Limitation**: INQUIRE is a *retrieval* benchmark, not a *classification* one.
  It's great for evaluating raw BioCLIP zero-shot retrieval but doesn't directly
  provide "interaction type" labels for supervised training. We'll use it for:
  1. Zero-shot retrieval baseline (raw BioCLIP text→image matching)
  2. Mining candidate images for our custom labeled set (filter by query + relevant)

#### 1B. Assemble Custom Labeled Interaction Set

For the PoC, we need images labeled with **interaction type** (not just species).
INQUIRE images have species labels but not interaction labels. Options:

1. **Manual labeling from INQUIRE images**: Browse relevant images from
   interaction-themed queries, label each as "interaction present" / "no interaction"
   / specific interaction type.

2. **External datasets**: Search HuggingFace / iNaturalist for datasets with
   co-occurrence or interaction annotations.
   - `leonelgv/pollinator-insects-dataset` (419 downloads) — pollinator insects
   - iNat24 (the base for INQUIRE) has species + location metadata
   - Could mine for images where a pollinator species + flowering plant co-occur
     in the same frame

3. **Sage node camera data**: Use existing Sage camera images from nodes that
   capture outdoor scenes. Label a small set manually.

**Recommended for PoC**: Start with ~100-200 manually labeled images for 1-2
interaction types:
  - **Type A**: "Pollinator on flower" (bee/butterfly/fly on a flower)
  - **Type B**: "No interaction" (empty flower, solitary bird, etc.)
  - Binary classifier: interaction / no-interaction

- [ ] Set up labeling directory structure
  ```
  ~/BISONN/data/
    raw/               # source images
    labeled/
      pollinator_on_flower/
        positive/      # images showing pollinator on flower
        negative/      # images of flowers without pollinators
      [optional: second interaction type]
    inquire_subset/    # images pulled from INQUIRE for our subset
  ```
- [ ] Download / curate ~100-200 images per class
- [ ] Create a CSV manifest: `image_path, label, interaction_type, source`
- [ ] Train/test split: 80/20, stratified by interaction type

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

  # Load BioCLIP (start with original; upgrade to BioCLIP-2 if VRAM allows)
  model, _, preprocess = open_clip.create_model_and_transforms(
      "hf-hub:imageomics/bioclip"
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
  tokenizer = open_clip.get_tokenizer("hf-hub:imageomics/bioclip")
  prompts = [
      "a photo of an insect pollinating a flower",
      "a photo of a flower with no insect on it",
      "a bee on a flower",
      "a butterfly on a flower",
      "an empty flower",
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
      nn.Linear(512, 256),  # 512 = BioCLIP embedding dim
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(256, num_classes)
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

  # Load models (from baked-in paths, no network)
  model, _, preprocess = open_clip.create_model_and_transforms(
      "hf-hub:imageomics/bioclip"
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
              str(prediction),
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

- [ ] Write `Dockerfile`
  ```dockerfile
  FROM docker.io/waggle/sage-thor-base:0.1.0

  WORKDIR /app

  COPY requirements.txt .
  RUN pip3 install --no-cache-dir -r requirements.txt

  # Bake model weights at build time
  COPY models/ /app/models/

  COPY app.py .
  ENTRYPOINT ["python3", "app.py"]
  ```

- [ ] Write `requirements.txt`
  ```
  pywaggle[all]==0.56.0
  open_clip_torch>=2.20.0
  timm==1.0.15
  scikit-learn
  joblib
  pillow
  numpy
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

1. How well do raw BioCLIP embeddings separate "interaction" from "no interaction"
   without any training? (zero-shot baseline)
2. How much does a simple linear probe improve over zero-shot?
3. Does a non-linear head (MLP) add value over linear, or are the embeddings
   already linearly separable?
4. Is the approach generalizable? (test on a second interaction type if data allows)
5. Can we use INQUIRE queries as a proxy for interaction detection?

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

- **BioCLIP**: `imageomics/bioclip` (ViT-B/16, TreeOfLife-10M) on HuggingFace
- **BioCLIP-2**: `imageomics/bioclip-2` (SigLIP2, 430M params, better accuracy)
- **BioCLIP 2.5 Huge**: `imageomics/bioclip-2.5-vith14` (ViT-H/14, ~1B params, 61.3% species accuracy)
- **pybioclip**: Python wrapper for BioCLIP taxonomy classification
- **INQUIRE Benchmark**: `sagecontinuum/INQUIRE-Benchmark-small` (20K images, 250 expert queries)
- **Sage pluginctl**: build and test plugins on Thor nodes
- **pywaggle**: `Plugin`, `Camera`, `upload_file` — the Sage edge SDK
- **waggle/sage-thor-base**: Docker base image for Thor plugins

### Sage platform reference docs
- Plugin structure: app-tutorial template (in `~/app-tutorial/`)
- Deploy: `pluginctl` (camp guide in sage-waggle skill)
- ECR workaround: podman build + side-load (runc /proc/acpi bug fleet-wide)
- Data API: `sage-data-client` or Sage MCP server
