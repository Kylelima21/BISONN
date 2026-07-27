# BISONN
## Biotic Interactions with Sage Observations using Neural Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 25.08](https://img.shields.io/badge/PyTorch-25.08(NVIDIA)-orange.svg)](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)
[![Sage/Waggle](https://img.shields.io/badge/Sage-Waggle-green.svg)](https://sagecontinuum.org/)

**Author:** Kyle Lima — University of Maine, Schoodic Institute at Acadia National Park (`klima@schoodicinstitute.org`)

**Repository:** [https://github.com/Kylelima21/BISONN](https://github.com/Kylelima21/BISONN)

<br/>

A Sage/Waggle edge plugin that detects **bird mobbing behavior** and identifies
**species** from camera images using BioCLIP 2.5 embeddings. Behavior is
classified by a frozen Linear SVM (98.8% accuracy, 0.948 macro-F1 on held-out
test set). Species is identified zero-shot via cosine similarity to 30 curated
North American bird text prompts — a task BioCLIP was explicitly trained for on
TreeOfLife-200M.

For the full project write-up with figures, tables, and methods detail, see
[`project.md`](project.md). For detailed plugin usage instructions, see
[`overview.md`](overview.md).

<br/>

## How It Works

1. **Capture** — acquires a camera snapshot (RTSP stream, HTTP snapshot URL, or test image directory)
2. **Embed** — encodes the image with BioCLIP 2.5 Huge (ViT-H/14, 1024-dim, L2-normalized)
3. **Classify behavior** — frozen Linear SVM predicts `mobbing` (1) or `none` (0)
4. **Identify species** — cosine similarity between the image embedding and 30 species text prompts (zero-shot, no training needed)
5. **Publish** — publishes behavior, species, and heartbeat topics to the Sage data pipeline via pywaggle
6. **Upload** — optionally uploads an annotated image (behavior + species overlay) for human review

## Model

- **Backbone**: BioCLIP 2.5 Huge (`imageomics/bioclip-2.5-vith14`, ViT-H/14, ~1B params, 1024-dim, TreeOfLife-200M pretraining)
- **Behavior classifier**: Linear SVM (`sklearn.svm.SVC`, `kernel='linear'`, `class_weight='balanced'`)
- **Species ID**: Zero-shot cosine similarity to 30 North American bird text prompts (corvids, raptors, songbirds)
- **Model file**: `models/svm.joblib` (1.7 MB)
- **Training data**: 1690 labeled bird images (101 mobbing + 1589 none)
- **Split**: 75/15/15 stratified (train/val/test, seed=42)
- **Performance**: 98.8% accuracy, 0.948 macro-F1 (test set), mobbing F1=0.903 (val macro-F1=0.965)

## Cross-Model Comparison

Three foundation model backbones (BioCLIP 2.5, DINOv3 Large, DINOv3 Small) were compared with three classification heads (logistic regression, linear SVM, kNN). BioCLIP 2.5 + Linear SVM was the clear winner:

| Backbone      | Head         | Test Macro-F1 | Mobbing F1 |
|---------------|-------------|---------------|------------|
| **BioCLIP 2.5**   | **Linear SVM**  | **0.948**         | **0.903**      |
| BioCLIP 2.5   | Logistic Reg | 0.905         | 0.824      |
| DINOv3 Large  | Linear SVM   | 0.842         | 0.706      |
| DINOv3 Small  | Linear SVM   | 0.829         | 0.684      |

## Data Sources

- iNaturalist (CC-licensed, via API)
- Wikimedia Commons (CC-licensed)
- Personal field photography (Kyle Lima)

## Topics Published

| Topic | Value | Meta |
|-------|-------|------|
| `biotic.interaction.bird_mobbing` | 1 (mobbing) or 0 (none) | camera, label, confidence, species, species_confidence, model |
| `biotic.species.bird` | 1 (presence indicator) | camera, species, species_confidence, species_top3, model |
| `biotic.interaction.summary` | same (heartbeat) | camera, label, confidence, species, model |

## Repository Structure

```
BISONN/
  app.py                  # plugin entry point (also in plugin/)
  sage.yaml               # Sage plugin metadata
  Dockerfile              # NVIDIA PyTorch 25.08 base (Blackwell sm_110)
  requirements.txt        # Python dependencies
  Makefile                # build / test / run targets
  overview.md             # detailed plugin documentation
  README.md               # this file
  project.md              # full project write-up with figures
  PLAN.md                 # implementation plan (Phases 0-5)
  DEVLOG.md               # development log
  data/
    labeled/              # mobbing/ and none/ image folders
    embeddings_bisonn.npz # BioCLIP 2.5 embeddings (1690, 1024)
    labels_bisonn.npy     # integer labels (0=mobbing, 1=none)
    models/               # trained classifiers (svm.joblib, logistic.joblib, etc.)
    results/              # evaluation reports, confusion matrices, comparison charts
  scripts/
    extract_embeddings.py       # BioCLIP embedding extraction
    extract_embeddings_dinov3.py # DINOv3 embedding extraction
    train_and_evaluate.py       # BioCLIP head training + evaluation
    train_dinov3.py             # DINOv3 head training + evaluation
    compare_models.py           # cross-model comparison
    plot_all_confusion_matrices.py
  plugin/                 # deployable plugin (mirrors root files + models/)
  ecr-meta/               # Sage ECR science description + images
```

## Requirements

- Python 3.12+ (aarch64 on Thor, or x86_64 for dev)
- See [`requirements.txt`](requirements.txt) for Python dependencies
- Core stack: `open_clip_torch`, `scikit-learn`, `joblib`, `opencv-python`, `pywaggle`
- For training/evaluation (CPU-only dev): `numpy`, `pandas`, `matplotlib`, `timm` (DINOv3)
- For deployment: NVIDIA PyTorch 25.08 container (`nvcr.io/nvidia/pytorch:25.08-py3`) with Blackwell sm_110 GPU support

## Quickstart

```bash
# Clone
git clone https://github.com/Kylelima21/BISONN.git
cd BISONN

# Set up dev environment (CPU-only on Thor host)
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run the test suite (syntax + YAML + model files)
make test

# Train and evaluate BioCLIP classification heads (75/15/15 split)
CUDA_VISIBLE_DEVICES='' python3 scripts/train_and_evaluate.py

# Cross-model comparison (BioCLIP vs DINOv3)
CUDA_VISIBLE_DEVICES='' python3 scripts/compare_models.py

# Deploy as a Sage plugin (on a Thor node)
cd plugin/
sudo pluginctl build .
sudo pluginctl run --name bisonn-test \
  --resource limit.memory=16Gi,request.memory=4Gi \
  <image-ref> -- --image-dir /app/test-images --continuous N
```

## Deployment

Built on the NVIDIA PyTorch 25.08 base (`nvcr.io/nvidia/pytorch:25.08-py3`)
for Blackwell GPU support (sm_110). BioCLIP weights are pre-downloaded at build
time and baked into the image — no internet needed at runtime.

```bash
cd ~/BISONN/plugin/
sudo pluginctl build .
sudo pluginctl run --name bisonn-test \
  --resource limit.memory=16Gi,request.memory=4Gi \
  <image-ref> -- --image-dir /app/test-images --continuous N
sudo pluginctl logs bisonn-test
```

## Initial Sage Grande Testbed (SGT) Workshop Goals
- Develop a proof of concept example for monitoring biotic interactions at the edge using AI
- Build and test simple classification heads for BioCLIP to quantify biotic interactions
- Compare performance of these classification heads and of the base BioCLIP embeddings for quantifying biotic interactions
- Do the same for DINOv3 and compare performance among all models
- Publish the best model to the SGT at the edge

<br/>

## Long Term Goals
- Integrate multiple sensors (ARU, video, image) to monitor for interactions more effectively
- Develop a more powerful, well trained version of this model for many more types of biotic interactions
- Scale data collection across the SGT network

## Acknowledgments

- [Sage Grande Testbed (SGT)](https://sagecontinuum.org/) — edge computing infrastructure and Waggle plugin framework
- [BioCLIP 2.5](https://huggingface.co/imageomics/bioclip-2.5-vith14) — Imageomics Institute, trained on TreeOfLife-200M
- [DINOv3](https://huggingface.co/timm/vit_large_patch16_dinov3_qkvb.lvd1689m) — Meta AI self-supervised vision transformer
- [Imageomics BioCLIP Workshop](https://github.com/Imageomics/sage-summer-2026-bioclip) — embedding bundle pattern and evaluation methodology
- [NVIDIA PyTorch Container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch) — Blackwell GPU support for Thor nodes

## License

MIT
