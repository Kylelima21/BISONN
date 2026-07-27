"""
BISONN — Biotic Interactions with Sage Observations using Neural Networks.

Sage/Waggle plugin that detects bird mobbing behavior from camera images
using BioCLIP 2.5 embeddings + a frozen Linear SVM classifier, and also
provides zero-shot species identification using BioCLIP's text encoder.

Pipeline per capture:
  camera snapshot → BioCLIP encode (1024-dim, L2-normalized) → SVM classify
   → publish biotic.interaction.bird_mobbing (1=mobbing, 0=none)
   → cosine similarity vs species text prompts → publish biotic.species (top-3)

Model: BioCLIP 2.5 Huge (ViT-H/14, ~1B params, 1024-dim embeddings)
       + Linear SVM (class_weight='balanced', trained on 1690 bird images)
       Best: 98.5% accuracy, 0.935 macro-F1, mobbing F1=0.878
       Species ID: zero-shot cosine similarity to curated species prompts
       (BioCLIP was trained on TreeOfLife-200M, so species is its strength)

Usage modes (mutually exclusive, priority: image-dir > snapshot-url > stream):
  --stream <source>        Camera name, RTSP URL, or single image path
  --snapshot-url <url>     HTTP URL returning a JPEG snapshot (Reolink CGI API)
  --image-dir <path>       Process all images in a directory (batch test)
"""
import argparse
import logging
import os
import time

import cv2
import joblib
import numpy as np
import torch
from PIL import Image

from waggle.plugin import Plugin
from waggle.data.vision import Camera

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# BioCLIP 2.5 Huge — ViT-H/14, 1024-dim embeddings
BIOCLIP_MODEL = "hf-hub:imageomics/bioclip-2.5-vith14"

# Label mapping (must match training: data/label_names.json)
LABEL_NAMES = ["mobbing", "none"]  # 0=mobbing, 1=none

# Species prompt list for zero-shot species ID.
# BioCLIP was trained on TreeOfLife-200M species captions — species is its
# strength. We use "a photo of a <species>" framing (matches training format).
# Curated for North American birds, emphasizing mobbing-relevant species
# (corvids, raptors, and common songbirds that participate in mobbing).
SPECIES_PROMPTS = [
    # ── Corvids (frequent mobbers) ──
    "a photo of an American Crow",
    "a photo of a Common Raven",
    "a photo of a Blue Jay",
    "a photo of a Steller's Jay",
    "a photo of a Black-billed Magpie",
    "a photo of a Fish Crow",
    # ── Raptors (frequent mobbing targets) ──
    "a photo of a Red-tailed Hawk",
    "a photo of a Cooper's Hawk",
    "a photo of a Sharp-shinned Hawk",
    "a photo of a Broad-winged Hawk",
    "a photo of a Great Horned Owl",
    "a photo of a Barred Owl",
    "a photo of an Eastern Screech-Owl",
    "a photo of a Northern Saw-whet Owl",
    "a photo of a Peregrine Falcon",
    "a photo of an American Kestrel",
    "a photo of a Bald Eagle",
    "a photo of a Turkey Vulture",
    "a photo of a Zone-tailed Hawk",
    # ── Songbirds (common mobbing participants) ──
    "a photo of a Black-capped Chickadee",
    "a photo of a Tufted Titmouse",
    "a photo of a Red-breasted Nuthatch",
    "a photo of a White-breasted Nuthatch",
    "a photo of a Downy Woodpecker",
    "a photo of a Northern Cardinal",
    "a photo of an American Robin",
    "a photo of a Gray Catbird",
    "a photo of a Common Grackle",
    "a photo of a Red-winged Blackbird",
]


class BisonnClassifier:
    """BioCLIP 2.5 embedding extractor + frozen Linear SVM classifier."""

    def __init__(self, classifier_path, device="cuda"):
        self.device = device
        self._load_bioclip()
        self._load_classifier(classifier_path)
        self._load_species_prompts()

    def _load_bioclip(self):
        """Load BioCLIP 2.5 Huge model + preprocessing transform."""
        import open_clip

        logger.info("Loading BioCLIP 2.5 Huge (%s)...", BIOCLIP_MODEL)
        t0 = time.time()
        model, _, preprocess = open_clip.create_model_and_transforms(BIOCLIP_MODEL)
        model = model.to(self.device).eval()
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(BIOCLIP_MODEL)
        logger.info("BioCLIP loaded in %.1fs (device=%s)", time.time() - t0, self.device)

    def _load_classifier(self, path):
        """Load the trained Linear SVM classifier."""
        logger.info("Loading SVM classifier: %s", path)
        self.classifier = joblib.load(path)
        logger.info("Classifier loaded: %s", type(self.classifier).__name__)

    def _load_species_prompts(self):
        """Encode the curated species prompts with BioCLIP's text encoder."""
        logger.info("Encoding %d species prompts...", len(SPECIES_PROMPTS))
        t0 = time.time()
        with torch.inference_mode():
            tokens = self.tokenizer(SPECIES_PROMPTS)
            text_features = self.model.encode_text(tokens, normalize=True)
        self.species_features = text_features.float()  # (N_species, 1024)
        # Extract just the species name (strip "a photo of a/an ")
        self.species_names = [
            p.replace("a photo of a ", "").replace("a photo of an ", "")
            for p in SPECIES_PROMPTS
        ]
        logger.info("Species prompts encoded in %.1fs (%d species)",
                     time.time() - t0, len(self.species_names))

    def predict(self, frame):
        """
        Run inference on a BGR numpy array (H, W, 3).

        Returns:
            dict with keys: label (str), label_idx (int), confidence (float),
            species (str), species_confidence (float), species_top3 (list)
        """
        # Convert BGR (OpenCV) → RGB (PIL)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = self.preprocess(Image.fromarray(rgb))

        with torch.inference_mode():
            emb = self.model.encode_image(img.unsqueeze(0).to(self.device), normalize=True)
        emb_np = emb.float().cpu().numpy()  # (1, 1024)

        # ── Behavior classification (SVM) ──
        label_idx = int(self.classifier.predict(emb_np)[0])
        label = LABEL_NAMES[label_idx]

        # Confidence: SVM decision_function distance from the margin
        decision = 0.0
        confidence = 0.5
        try:
            decision = float(self.classifier.decision_function(emb_np)[0])
            confidence = 1.0 / (1.0 + np.exp(-abs(decision)))
        except Exception:
            pass

        # ── Species identification (zero-shot cosine similarity) ──
        # Both image embedding and species text embeddings are L2-normalized,
        # so dot product = cosine similarity. Shape: (1, N_species)
        species_scores = (emb_np @ self.species_features.cpu().numpy().T)[0]
        top3_idx = np.argsort(species_scores)[-3:][::-1]
        top3 = [
            (self.species_names[i], float(species_scores[i]))
            for i in top3_idx
        ]
        species = top3[0][0]
        species_confidence = top3[0][1]

        return {
            "label": label,
            "label_idx": label_idx,
            "confidence": confidence,
            "decision": decision,
            "species": species,
            "species_confidence": species_confidence,
            "species_top3": top3,
        }


def iter_image_dir(directory):
    """Yield (path, frame, timestamp) for each image in a directory."""
    from pathlib import Path as P

    files = sorted(
        f for f in P(directory).iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
        and not f.name.startswith(".")
    )
    if not files:
        raise FileNotFoundError(
            f"No image files found in {directory}. "
            f"Supported: {', '.join(sorted(IMAGE_EXTENSIONS))}"
        )
    logger.info("Found %d test images in %s", len(files), directory)
    for img_path in files:
        frame = cv2.imread(str(img_path))
        if frame is None:
            logger.warning("Skipping unreadable file: %s", img_path.name)
            continue
        yield str(img_path), frame, time.time_ns()


def fetch_snapshot(url):
    """Fetch a JPEG snapshot from an HTTP URL, return as BGR numpy array."""
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=15) as resp:
            img_bytes = resp.read()
    except Exception as e:
        raise ConnectionError(f"Failed to fetch snapshot from {url}: {e}") from e

    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f"Could not decode image from {url} ({len(img_bytes)} bytes)")
    return frame


def main():
    parser = argparse.ArgumentParser(
        description="BISONN — bird mobbing behavior classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Input modes (mutually exclusive, priority: image-dir > snapshot-url > stream):
  --stream <source>       Camera name, RTSP URL, or single image path
  --snapshot-url <url>    HTTP URL returning a JPEG (e.g. Reolink CGI API)
  --image-dir <path>      Process all images in a directory (batch)

Examples:
  # Live camera (continuous):
  python app.py --stream bottom_camera --continuous Y

  # HTTP snapshot (one-shot test):
  python app.py --snapshot-url "http://IP/cgi-bin/api.cgi?cmd=Snap&user=U&password=P" --continuous N

  # Directory of test images:
  python app.py --image-dir ./test-images/ --continuous N
""",
    )
    parser.add_argument("--stream", default="bottom_camera",
                        help="Camera name, RTSP URL, or image path")
    parser.add_argument("--image-dir", default=None,
                        help="Process all images in a directory (overrides --stream)")
    parser.add_argument("--snapshot-url", default=None,
                        help="HTTP URL that returns a JPEG snapshot. "
                             "Credentials go in the URL query string.")
    parser.add_argument("--continuous", default="Y",
                        help="Y=loop, N=single-shot")
    parser.add_argument("--interval", type=int, default=30,
                        help="Seconds between captures in continuous mode")
    parser.add_argument("--classifier", default="/app/models/svm.joblib",
                        help="Path to the trained SVM classifier (.joblib)")
    parser.add_argument("--upload-image", default="Y",
                        help="Y = upload the analyzed image each cycle")
    args = parser.parse_args()

    # Device: use CUDA if available, else CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    if device == "cpu":
        logger.warning("Running on CPU — inference will be slow (~1.7s/image)")

    using_image_dir = args.image_dir is not None
    using_snapshot_url = args.snapshot_url is not None

    # Initialize image source
    if using_image_dir:
        image_source = iter_image_dir(args.image_dir)
        source_label = f"image-dir:{args.image_dir}"
    elif using_snapshot_url:
        source_label = args.snapshot_url.split("?")[0]  # strip credentials
    else:
        camera = Camera(args.stream)
        source_label = args.stream

    # Load model
    classifier = BisonnClassifier(args.classifier, device=device)

    with Plugin() as plugin:
        logger.info(
            "BISONN plugin started — source=%s, interval=%ds, continuous=%s",
            source_label, args.interval, args.continuous,
        )

        while True:
            try:
                # Acquire frame
                if using_image_dir:
                    try:
                        img_path, frame, timestamp = next(image_source)
                    except StopIteration:
                        logger.info("All test images processed")
                        break
                    source_name = os.path.basename(img_path)
                    logger.info("Processing: %s (%dx%d)",
                                source_name, frame.shape[1], frame.shape[0])
                elif using_snapshot_url:
                    frame = fetch_snapshot(args.snapshot_url)
                    timestamp = time.time_ns()
                    source_name = "http-snapshot"
                    logger.info("Snapshot: %dx%d from %s",
                                frame.shape[1], frame.shape[0], source_label)
                else:
                    sample = camera.snapshot()
                    frame = sample.data  # numpy BGR
                    timestamp = sample.timestamp
                    source_name = args.stream

                # Run inference
                result = classifier.predict(frame)
                label = result["label"]
                label_idx = result["label_idx"]
                confidence = result["confidence"]
                species = result["species"]
                species_confidence = result["species_confidence"]

                top3_str = ", ".join(
                    f"{s} ({c:.2f})" for s, c in result["species_top3"]
                )
                logger.info(
                    "Prediction: %s (%.3f) | species: %s (%.3f) | top3: %s — %s",
                    label, confidence, species, species_confidence,
                    top3_str, source_name,
                )

                # Publish behavior classification (1=mobbing detected, 0=none)
                # PYWAGGLE GOTCHA: every meta value MUST be a string.
                # Floats/ints cause "Meta must be a dictionary of strings"
                # at publish, silently dropping the record.
                plugin.publish(
                    "biotic.interaction.bird_mobbing",
                    label_idx,  # value can be numeric (only META must be str)
                    timestamp=timestamp,
                    meta={
                        "camera": source_name,
                        "label": label,
                        "confidence": f"{confidence:.4f}",
                        "species": species,
                        "species_confidence": f"{species_confidence:.4f}",
                        "model": "bioclip2.5-vith14+linear-svm",
                    },
                )

                # Publish species identification (zero-shot BioCLIP)
                plugin.publish(
                    "biotic.species.bird",
                    1,  # presence indicator — always 1 (bird detected)
                    timestamp=timestamp,
                    meta={
                        "camera": source_name,
                        "species": species,
                        "species_confidence": f"{species_confidence:.4f}",
                        "species_top3": top3_str,
                        "model": "bioclip2.5-vith14-zeroshot",
                    },
                )

                # Heartbeat summary — always publish so data plane knows job is alive
                plugin.publish(
                    "biotic.interaction.summary",
                    label_idx,
                    timestamp=timestamp,
                    meta={
                        "camera": source_name,
                        "label": label,
                        "confidence": f"{confidence:.4f}",
                        "species": species,
                        "model": "bioclip2.5-vith14",
                    },
                )
                logger.info("Published mobbing=%d (%s) species=%s (%.3f)",
                            label_idx, label, species, species_confidence)

                # Optionally upload the analyzed image
                if args.upload_image == "Y":
                    import tempfile
                    tmpdir = tempfile.mkdtemp()
                    out_path = os.path.join(tmpdir, f"bird_{label}_{timestamp}.jpg")

                    # Annotate the image with behavior + species predictions
                    annotated = frame.copy()
                    # Line 1: behavior label + confidence
                    text1 = f"{label} ({confidence:.2f})"
                    cv2.putText(annotated, text1, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    # Line 2: top species + confidence
                    text2 = f"{species} ({species_confidence:.2f})"
                    cv2.putText(annotated, text2, (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    # Line 3: runner-up species
                    if len(result["species_top3"]) > 1:
                        s2, c2 = result["species_top3"][1]
                        text3 = f"2nd: {s2} ({c2:.2f})"
                        cv2.putText(annotated, text3, (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)
                    cv2.imwrite(out_path, annotated)

                    plugin.upload_file(
                        out_path,
                        timestamp=timestamp,
                        meta={
                            "label": label,
                            "confidence": f"{confidence:.4f}",
                            "species": species,
                            "species_confidence": f"{species_confidence:.4f}",
                            "model": "bioclip2.5-vith14",
                            "camera": source_name,
                        },
                    )
                    logger.info("Uploaded annotated image: %s", out_path)
                    os.remove(out_path)
                    os.rmdir(tmpdir)

            except Exception:
                logger.exception("Inference error")

            if args.continuous != "Y" and not using_image_dir:
                break
            if not using_image_dir:
                time.sleep(args.interval)


if __name__ == "__main__":
    main()
