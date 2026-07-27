# BISONN — Development Log

Biotic Interactions with Sage Observations using Neural Networks

---

## Labeled Data Breakdown (as of 2026-07-24)

Manifest: `data/manifest_unified.csv` (1690 rows)

| Class    | Count | iNaturalist | Wikimedia | Personal | Total |
|----------|-------|-------------|-----------|----------|-------|
| mobbing  | —     | 0           | 98        | 3        | 101   |
| none     | —     | 78          | 29        | 1482     | 1589  |
| **TOTAL**|       | **78**      | **127**   | **1485** | **1690** |

Class imbalance: 101 mobbing vs 1589 none (~1:16). Phase 3 training
will need class-weighted loss or balanced sampling to avoid a trivial
"always predict none" classifier.

Notes:
- Personal photos use Flickr-style descriptive filenames ending `_o.jpg` /
  `_o.jpeg` (e.g. `none_accipiter-cooperii---coopers-hawk_27633331108_o.jpg`)
- One `.mov` video (eastern-whip-poor-will) was in `none/` — removed by user,
  not in manifest
- `scripts/sync_manifests.py` updated to handle the new filename pattern and
  skip non-image files

---

## Phase Status

| Phase | Description                        | Status |
|-------|------------------------------------|--------|
| 0     | Environment setup (venv, BioCLIP)  | Done   |
| 1     | Data acquisition & labeling        | Done   |
| 2     | BioCLIP + DINOv3 embedding extraction | Done   |
| 3     | Train & evaluate classification heads | Done   |
| 4     | Sage plugin packaging              | Done   |
| 5     | Build, test, deploy on Thor        | Done   |

### Phase 0 — Environment Setup (done)
- venv at `~/BISONN/venv` with torch, open_clip, BioCLIP, pywaggle, etc.
- BioCLIP 2.5 Huge (`imageomics/bioclip-2.5-vith14`, ViT-H/14, 1024-dim)
  loads on CPU with `CUDA_VISIBLE_DEVICES=''` — ~1.7s/image
- PyPI torch (2.13+cu130) hangs on CUDA calls (no Blackwell sm_110 kernels),
  so dev work is CPU-only. GPU only needed for deployed plugin (Phase 5) via
  `pluginctl --selector resource.gpu=true`
- NVIDIA container `nvcr.io/nvidia/pytorch:25.08-py3` (torch 2.8, sm_110)
  pulled and ready for Phase 5 plugin builds

### Phase 1 — Data Acquisition & Labeling (done)
- Sources: iNaturalist (CC-licensed API), Wikimedia Commons, personal photos
- `feeding_young` class dropped — binary mobbing/none only
- User manually reviewed all images, reclassified 36 from mobbing to none
- BioCLIP zero-shot cleaned non-bird images from none folder
- Manifest synced via `python3 scripts/sync_manifests.py`

### Phase 2 — BioCLIP + DINOv3 Embedding Extraction (done)
- `scripts/extract_embeddings.py` — BioCLIP 2.5 Huge (1024-dim), CPU
- `scripts/extract_text_embeddings.py` — behavior prompts for zero-shot
- `scripts/extract_embeddings_dinov3.py` — DINOv3 small (384-dim) + large (1024-dim)
- `scripts/embedding_bundles.py` — EmbeddingBundle pattern (from Peromyscus notebook)
- Embeddings saved as `.npz` (L2-normalized features + IDs + manifest)
- Labels: `data/labels_bisonn.npy` (0=mobbing, 1=none), `data/label_names.json`
- Behavior prompts: `data/behavior_prompts.json`
- Text embeddings: `data/text_embeddings_bisonn.npz`

### Phase 3 — Train & Evaluate Classification Heads (done)
- `scripts/train_and_evaluate.py` — BioCLIP 2.5 heads
- `scripts/train_dinov3.py` — DINOv3 small + large heads
- `scripts/compare_models.py` — cross-model comparison
- 80/20 stratified split (seed=42), class_weight='balanced'
- Methods per backbone: zero-shot (BioCLIP only), logistic regression,
  linear SVM, kNN (k=5, cosine), MLP (planned but see results)

**Cross-model comparison (macroF1):**

| Backbone      | Logistic | Linear SVM | kNN (k=5) |
|---------------|----------|------------|-----------|
| BioCLIP 2.5   | 0.883    | **0.935**  | 0.845     |
| DINOv3 Large  | 0.792    | 0.821      | 0.845     |
| DINOv3 Small  | 0.711    | 0.835      | 0.843     |

**Best overall: BioCLIP 2.5 + Linear SVM (macroF1=0.935)**

Confusion matrix (best model, 20 mobbing / 318 none test):
```
              mobbing    none
     mobbing       18       2
        none         3     315
```

Key findings:
- BioCLIP 2.5 dominates DINOv3 on mobbing detection — biology-trained
  features beat general vision features
- Linear SVM is the best head — embeddings are largely linearly separable
- Zero-shot is poor (macroF1=0.375) — text prompts don't capture mobbing
  behavior well enough for cosine retrieval
- kNN is competitive across backbones (0.84-0.85) — local neighborhood
  structure is consistent
- Mobbing recall 0.90 (18/20 caught) with the best model — only 2 missed
- Mobbing precision 0.86 (18/21 predicted correct) — 3 false positives

Model weights saved: `data/models/{svm,logistic}.joblib` (BioCLIP 2.5),
`data/models/dinov3_{small,large}_{svm,logistic,best}.joblib`
Results: `data/results/evaluation_report*.txt`, `comparison_bar*.png`,
`best_model_confusion*.png`, `cross_model_comparison.{txt,png}`,
`all_confusion_matrices.png`

### Phase 4 — Sage Plugin Packaging (done)
- Plugin scaffold at `~/BISONN/plugin/`:
  - `app.py` — Sage/Waggle plugin: camera snapshot → BioCLIP 2.5 encode
    (1024-dim, L2-normalized) → Linear SVM classify → publish + upload
  - `Dockerfile` — NVIDIA PyTorch 25.08 base (sm_110 Blackwell), frozen
    torch/torchvision/numpy, pre-downloads BioCLIP weights at build time,
    offline mode at runtime
  - `requirements.txt` — pywaggle, open_clip, sklearn, joblib, PIL, opencv
  - `sage.yaml` — plugin metadata (name=bisonn, arch=linux/arm64)
  - `overview.md` + `ecr-meta/ecr-science-description.md` — documentation
  - `models/svm.joblib` (1.7MB) + `models/label_names.json` baked in
  - `Makefile` + `.dockerignore` for build hygiene
- Input modes: camera stream, HTTP snapshot URL, image directory (batch test)
- Publishes `biotic.interaction.bird_mobbing` (1/0) + heartbeat summary
- All pywaggle meta values are strings (pywaggle requirement)
- Verified: app.py syntax OK, sage.yaml valid YAML, SVM classes [0,1]
  match label_names.json (0=mobbing, 1=none)

### Phase 5 — Build, Test, Deploy on Thor (done)
- Built plugin image with `sudo pluginctl build` (NVIDIA PyTorch 25.08 base,
  12 Dockerfile steps, BioCLIP weights pre-downloaded at build time)
- K3s registry TLS issue: `10.31.81.1:5000` uses a self-signed cert that
  containerd could not verify (ImagePullBackOff). Workaround: podman save
  to tar, then used a privileged k3s debug pod with `nsenter --target 1`
  to import the 25GB tar into k3s containerd via `k3s ctr images import`
- One-shot test with 2 images (1 mobbing + 1 none) via `pluginctl run`:
  - Memory limit required: `--resource limit.memory=16Gi,request.memory=4Gi`
    (pluginctl default is too small for BioCLIP 2.5 — OOMKilled without it)
  - BioCLIP loaded in 7.5s, SVM classifier loaded (sklearn version warning
    is benign — 1.9 vs 1.6, works fine)
  - mobbing_sample.jpg → **mobbing** (confidence=0.731) ✓ correct
  - none_sample.jpg → **none** (confidence=0.919) ✓ correct
  - Both published to `biotic.interaction.bird_mobbing` (0=mobbing, 1=none)
- Plugin runs on CPU (~2s/image) — GPU not auto-injected by pluginctl.
  For production: need `--selector resource.gpu=true` or k8s resource limit
  for GPU. CPU is adequate for a 30s capture interval.
- Node: sgt-thor-1423125006073-H021 (00003c6d66fd3ac0.agx-thor), zone=core, resource.gpu=true
- Image: `localhost/bisonn:0.1.0` (15.6GB in k3s containerd)

### Confusion matrix figure
`scripts/plot_all_confusion_matrices.py` produces a single 3x4 grid
(`data/results/all_confusion_matrices.png`) with every backbone x head
combination:
- Row 0 (BioCLIP 2.5): Zero-shot | Logistic Reg | Linear SVM | kNN
- Row 1 (DINOv3 Large): (empty) | Logistic Reg | Linear SVM | kNN
- Row 2 (DINOv3 Small): (empty) | Logistic Reg | Linear SVM | kNN

All 40 confusion matrix cells verified to match the per-model
evaluation reports.

### Why DINOv3 has no zero-shot row

Zero-shot classification in the BioCLIP sense relies on a **text encoder**:
the model encodes behavioral prompts ("birds mobbing a predator") and
camera-trap images into a shared embedding space, then classifies by
cosine similarity between text and image vectors. This is a capability
unique to CLIP-family models (BioCLIP, BioCLIP 2, BioCLIP 2.5), which
are trained with contrastive language-image pretraining.

DINOv3 (available in timm as `vit_{small,large,huge}_patch16_dinov3`) is
a **self-supervised vision-only model**. It trains a vision transformer
on image augmentations alone — no text, no paired captions, no language
encoder. It produces rich image embeddings but has no text counterpart
and no shared text-image space.

Therefore DINOv3 cannot do text-prompt zero-shot classification. The
zero-shot row in the confusion matrix figure is BioCLIP 2.5 only. This
is not a gap in the analysis — it is a structural limitation of the
model family. The comparison remains fair because DINOv3 is included in
all supervised methods (logistic, SVM, kNN) where its vision embeddings
are directly usable.

A label-efficient "nearest class mean" (NCM) baseline could be computed
for DINOv3 (mean image embedding per class from a few labels, classify
by cosine to nearest class mean), but this uses labels and is more
analogous to 1-shot learning than true zero-shot. We chose not to add it
to avoid mixing label-free and label-dependent baselines in the same
figure.

### Phase 4 — Sage Plugin Packaging (pending)
- Plugin structure in `~/BISONN/plugin/`
- Dockerfile using NVIDIA base for Blackwell
- Bake BioCLIP weights + classifier into image at build time
- pywaggle Plugin, Camera, upload_file for inference loop

### Phase 5 — Build, Test, Deploy (pending)
- Build with podman locally, side-load with pluginctl (ECR broken)
- Test one-shot with sample image, then live camera
- `pluginctl --selector resource.gpu=true` for GPU access

---

## Reference: Peromyscus Notebook

Source: https://github.com/Imageomics/sage-summer-2026-bioclip/blob/main/notebooks/peromyscus.ipynb

This is the official BioCLIP tutorial/evaluation notebook from the Imageomics
team. It provides a complete workflow from zero-shot through quantization that
maps directly onto several BISONN phases.

### Three BioCLIP model generations compared

| Model       | Repo                           | Dim  | Params | Weight file                 |
|-------------|--------------------------------|------|--------|-----------------------------|
| BioCLIP     | imageomics/bioclip             | 512  | ~110M  | open_clip_pytorch_model.bin |
| BioCLIP 2   | imageomics/bioclip-2           | 768  | 430M   | open_clip_model.safetensors |
| BioCLIP 2.5 | imageomics/bioclip-2.5-vith14  | 1024 | ~1B    | open_clip_model.safetensors |

BISONN uses BioCLIP 2.5 (the largest, highest accuracy).

### Key patterns for BISONN

**Loading model with pinned revision:**
```python
from huggingface_hub import snapshot_download
import open_clip

spec = {
    "repo_id": "imageomics/bioclip-2.5-vith14",
    "revision": "191d741545e4c741cdef4b22c6eb69c945c1e592",
    "weight_file": "open_clip_model.safetensors",
}
snapshot = snapshot_download(
    repo_id=spec["repo_id"],
    revision=spec["revision"],
    allow_patterns=["open_clip_config.json", spec["weight_file"]],
)
model, _, _ = open_clip.create_model_and_transforms(
    f"local-dir:{snapshot}", device=DEVICE, precision="fp32"
)
```
Note: The `_` are preprocess transforms. The `precision="fp32"` matters for
reproducible embeddings. `local-dir:{snapshot}` is how open_clip loads a
remote checkpoint from a local cache path.

**Zero-shot classification (two methods):**
1. Plain taxonomic lineage — raw species name as text prompt
2. Training-template prototype — ensemble of 10 training-format prompts per
   species (e.g. "{common_name}", "a photo of {taxonomic_name}", etc.)
   The prompt ensemble is built via `build_training_prompt_ensemble()`.
   Text prototypes are the mean of the ensemble prompt embeddings.

For BISONN: instead of species names, our prompts are behavior descriptions
("birds mobbing a predator", "a solitary bird perched", etc.). The prototype
ensemble pattern still applies — multiple phrasings per behavior class,
averaged into a single text prototype.

**Stratified train/test split:**
```python
from sklearn.model_selection import train_test_split
train_indices, test_indices = train_test_split(
    indices, test_size=0.20, random_state=42, stratify=labels
)
```
One split for all models — prevents favorable splits. BISONN should do the same.

**Linear SVM on frozen embeddings:**
```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SVM_C = 0.01  # fixed before test evaluation
svm = make_pipeline(StandardScaler(), SVC(kernel="linear", C=SVM_C))
svm.fit(features[train_indices], labels[train_indices])
predictions = svm.predict(features[test_indices])
```
StandardScaler is fit on training data only. C=0.01 is deliberately small
(strong regularization) — fixed before looking at test results.

For BISONN: our class imbalance (1:16) requires `class_weight='balanced'`
in the SVC, which the Peromyscus notebook doesn't need (balanced 50/species).

**Few-shot experiment:**
```python
shot_counts = [1, 2, 5, 10, 20, 30, 40]
repeat_seeds = range(20)  # 20 draws per shot count, nested support sets
```
Larger sets retain smaller sets' examples. Reports mean accuracy ± std.
Useful BISONN question: how few mobbing images are enough?

**Adaptation (fine-tune last visual block only):**
- Trains only the final visual transformer block, final norm, and image
  projection of BioCLIP 2.5 — text encoder and earlier blocks frozen
- Targets: fixed text prototypes (no new classification head added)
- Temperature 0.10 (multiply cosine similarities by 10) to keep close
  alternatives in the gradient
- 30 train / 10 validation / 10 test per species
- Validation loss for early stopping, validation macro-F1 for checkpoint
- lr=1e-5, weight_decay=0.05, batch_size=8 (GPU) or 2 (CPU), max 30 epochs,
  patience=6

For BISONN: adaptation is optional Phase 3+ stretch goal. Could help if
frozen embeddings aren't linearly separable for mobbing detection. But our
small mobbing count (101) makes this risky — may overfit.

**Quantization (W8A8 dynamic PTQ):**
- `torchao` library for post-training quantization (weights 8-bit, activations
  8-bit dynamic)
- Compares FP32 vs W8A8: storage size, inference time, embedding cosine
  agreement, classification accuracy
- Key edge-relevant finding: W8A8 reduces storage and latency with minimal
  accuracy loss for BioCLIP 2.5

For BISONN: relevant for Phase 5 edge deployment — quantized model is smaller
and faster on the Thor, important for real-time camera trap inference.

**BioBench (164-task NeWT evaluation):**
- Broader benchmark beyond the Peromyscus task
- Evaluates frozen representations across appearance, counting, detection,
  and fine-grained tasks
- Compares FP32 vs W8A8 across all tasks
- Not directly applicable to BISONN (behavior classification, not in NeWT)
  but establishes that quantization doesn't broadly degrade representations

### Helper modules referenced
The notebook imports from sidecar files (auto-downloaded from the repo):
- `taxonomic_prompts.py` — builds class definitions and training-template
  prompt ensembles
- `embedding_bundles.py` — EmbeddingBundle (features + metadata + manifest)
- `fine_tuning_helpers.py` — `configure_last_visual_block()`,
  `train_last_visual_block()`, checkpoint save/load
- `quantization_helpers.py` — `benchmark_image_encoder()`,
  `encode_image_collection()`, quantization pipeline
- `interactive_camera_trap.py` — ImageBrowser widget
- `biobench_helpers.py` — NeWT task evaluation

These are at:
https://github.com/Imageomics/sage-summer-2026-bioclip/tree/main/notebooks

---

## Hardware Environment

- Node: sgt-thor-1423125006073-H021
- JetPack R38.2.1, aarch64, 128GB unified memory
- Podman + Docker available, k3s/WES stack
- Dev: CPU-only on host venv (`CUDA_VISIBLE_DEVICES=''`)
- Deploy: GPU via `pluginctl --selector resource.gpu=true` (k3s pod)
- NVIDIA container `nvcr.io/nvidia/pytorch:25.08-py3` for plugin builds
  (CUDA 13.0, PyTorch 2.8, sm_110/sm_121 for Blackwell)

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/download_inat.py` | Download CC-licensed bird photos from iNaturalist |
| `scripts/download_mobbing.py` | Download mobbing images from iNaturalist + WMC |
| `scripts/download_wmc_retry.py` | Retry WMC downloads that failed |
| `scripts/sync_manifests.py` | Rebuild unified manifest from disk folders |
| `scripts/clean_none_folder.py` | BioCLIP zero-shot cleaning of non-bird images |

---

## Key Decisions

1. Binary classification only (mobbing / none) — feeding_young dropped
2. BioCLIP 2.5 Huge (1024-dim) — frozen embeddings, no fine-tuning initially
3. Dev on CPU, GPU only for deployed plugin
4. Build with podman + pluginctl (ECR registration broken fleet-wide)
5. Personal photos included as `source=personal` in manifest
6. Class imbalance (1:16) needs addresssing in Phase 3
