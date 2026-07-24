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

| Phase | Description                           | Status  |
|-------|---------------------------------------|---------|
| 0     | Environment setup (venv, BioCLIP)     | Done    |
| 1     | Data acquisition & labeling           | Done    |
| 2     | BioCLIP embedding extraction          | Next    |
| 3     | Train & evaluate classification heads | Pending |
| 4     | Sage plugin packaging                 | Pending |
| 5     | Build, test, deploy on Thor           | Pending |

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

### Phase 2 — BioCLIP Embedding Extraction (next)
- Write `scripts/extract_embeddings.py` (not done yet)
- Extract 1024-dim embeddings for all 1690 labeled images on CPU
- Save embeddings + labels as `.npy` files
- Also extract text embeddings for zero-shot baseline prompts
- Estimated time: ~48 min for 1690 images at 1.7s/image

### Phase 3 — Train Classification Heads (pending)
- Zero-shot retrieval (cosine sim with text prompts)
- Logistic regression (linear probe)
- kNN (cosine, k=5)
- Optional MLP (1024 -> 256 -> 2)
- Evaluate: accuracy, macro-F1, precision, recall
- Handle class imbalance with class_weight='balanced' or balanced sampling

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
