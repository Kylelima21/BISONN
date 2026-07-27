# BISONN — Science Description

## Purpose

BISONN (Biotic Interactions with Sage Observations using Neural Networks)
demonstrates automated detection of bird mobbing behavior — a widespread but
understudied anti-predator interaction in which multiple birds harass a predator — using edge-deployed computer vision on Sage/Waggle
environmental sensor nodes. In addition to behavior classification, the plugin
performs zero-shot species identification on each image, leveraging BioCLIP's
text encoder to identify which North American bird species are present.

## Scientific Motivation

Mobbing is a biotic interaction that mediates predator-prey dynamics across
avian communities. While well-documented in behavioral ecology, mobbing events
are ephemeral, context-dependent, and difficult to observe systematically.
Automated camera-based detection on distributed environmental sensor nodes
enables continuous, unbiased observation at temporal and spatial scales not
achievable with manual fieldwork.

## Methods

BISONN combines:

1. **BioCLIP 2.5** (ViT-H/14, TreeOfLife-200M) — a foundation model pretrained
   on taxonomically-rich biological image-text pairs, producing 1024-dimensional
   visual embeddings that capture morphological and contextual features relevant
   to species identification and behavioral cue recognition.
2. **Linear SVM classifier** trained on 1690 labeled bird images (101 mobbing,
   1589 none) with class-weighted training to address the 1:16 class imbalance.
3. **Zero-shot species identification** using BioCLIP's text encoder with 30
   curated North American bird species prompts (corvids, raptors, songbirds),
   selected to cover species most relevant to mobbing interactions. No species-
   labeled training data is required — BioCLIP was explicitly trained on species
   captions via TreeOfLife-200M.

The classifier was selected through cross-model comparison across three
backbones (BioCLIP 2.5, DINOv3 Large, DINOv3 Small) and three classification
heads (logistic regression, linear SVM, kNN). A 75/15/15 stratified
train/validation/test split (seed=42) was used: models were trained on 75%
of the data, the best head per backbone was selected on the 15% validation set,
and final performance was reported on the held-out 15% test set. BioCLIP 2.5 +
Linear SVM achieved the best performance: 98.8% accuracy, 0.948 macro-F1,
0.903 mobbing F1 on the test set (validation macro-F1=0.965).

## Data

- 1690 labeled bird images sourced from iNaturalist (CC-licensed), Wikimedia
  Commons, and personal field photography
- Binary labels: `mobbing` (birds harassing/distracting a predator) and `none`
  (birds with no mobbing interaction)
- 75/15/15 stratified train/validation/test split (seed=42)
  - Train: 1182 images (71 mobbing, 1111 none)
  - Validation: 254 images (15 mobbing, 239 none)
  - Test: 254 images (15 mobbing, 239 none)

## Deployment

The plugin runs on Sage/Waggle Thor edge nodes (NVIDIA Jetson AGX Thor,
Blackwell architecture). It is built on the NVIDIA PyTorch 25.08 base image
(CUDA 13.0, PyTorch 2.8) for Blackwell GPU support (sm_110). BioCLIP 2.5
weights and the SVM classifier are baked into the container image at build
time, enabling offline inference at runtime. Each cycle captures a camera
snapshot, extracts a 1024-dim BioCLIP embedding, classifies behavior with the
frozen Linear SVM, identifies species via zero-shot cosine similarity to the
30 species text prompts, and publishes results to the Sage data pipeline.

## Outputs

- `biotic.interaction.bird_mobbing`: 1 (mobbing detected) or 0 (none), with
  confidence score, species, and species confidence in metadata
- `biotic.species.bird`: presence indicator (1) with top-3 species predictions
  and cosine similarity scores in metadata
- `biotic.interaction.summary`: heartbeat with label, confidence, and species
- Annotated image upload (optional, for human review) with behavior label,
  species prediction, and confidence overlays
