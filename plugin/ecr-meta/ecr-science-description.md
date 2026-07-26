# BISONN — Science Description

## Purpose

BISONN (Biotic Interactions with Sage Observations using Neural Networks)
demonstrates automated detection of bird mobbing behavior — a widespread but
understudied anti-predator interaction in which multiple birds harass or
distract a predator — using edge-deployed computer vision on Sage/Waggle
environmental sensor nodes.

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

The classifier was selected through cross-model comparison across three
backbones (BioCLIP 2.5, DINOv3 Large, DINOv3 Small) and three classification
heads (logistic regression, linear SVM, kNN). BioCLIP 2.5 + Linear SVM achieved
the best performance: 98.5% accuracy, 0.935 macro-F1, 0.878 mobbing F1.

## Data

- 1690 labeled bird images sourced from iNaturalist (CC-licensed), Wikimedia
  Commons, and personal field photography
- Binary labels: `mobbing` (birds harassing/distracting a predator) and `none`
  (birds with no mobbing interaction)
- 80/20 stratified train/test split (seed=42)

## Deployment

The plugin runs on Sage/Waggle Thor edge nodes (NVIDIA Jetson AGX Thor,
Blackwell architecture). Inference uses the GPU via the NVIDIA container
runtime. Each cycle captures a camera snapshot, extracts a 1024-dim BioCLIP
embedding, classifies it, and publishes the result to the Sage data pipeline.

## Outputs

- `biotic.interaction.bird_mobbing`: 1 (mobbing detected) or 0 (none)
- `biotic.interaction.summary`: heartbeat with label and confidence
- Annotated image upload (optional, for human review)
