#!/usr/bin/env python3
"""
Rebuild a unified manifest from the actual image folders on disk.

Scans all three labeled folders, matches each image to its metadata
from the existing manifests by photo_id, and writes a clean unified CSV.

Usage:
  python3 scripts/sync_manifests.py
"""

import csv
import os
from pathlib import Path
import re

DATA_DIR = Path("/home/kylelima21/BISONN/data")
LABELED_DIR = DATA_DIR / "labeled"

CLASSES = ["feeding_young", "mobbing", "none"]

# Only these extensions are images we care about
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}

# Read both existing manifests for metadata lookup
MANIFESTS = [
    DATA_DIR / "manifest.csv",
    DATA_DIR / "manifest_mobbing.csv",
    DATA_DIR / "manifest_unified.csv",  # pick up metadata from previous run
]


def load_all_metadata():
    """Load all manifest entries into a dict keyed by photo_id."""
    metadata = {}
    for mpath in MANIFESTS:
        if not mpath.exists():
            continue
        with open(mpath) as f:
            for row in csv.DictReader(f):
                pid = row.get("photo_id", "")
                if pid:
                    metadata[pid] = row
    return metadata


def parse_filename(filename):
    """Extract (class, source, photo_id) from a filename.

    Patterns:
      feeding_young_287547084.jpg
      mobbing_inat_517764085.jpg
      mobbing_wiki_10201920.jpg
      none_402766047.jpg
      none_wiki_134071783.jpg
      # Flickr-style descriptive names (personal photos / WMC batch 2):
      none_accipiter-cooperii---coopers-hawk_27633331108_o.jpg
      mobbing_buteo-albonotatus---zone-tailed-hawk-and-corvus-cryptoleucus---chihuahuan-raven_55190237223_o.jpg
      none_american-crow_15054929199_o.jpg
    """
    stem = Path(filename).stem
    ext = Path(filename).suffix

    # Try patterns
    # mobbing_inat_ID or mobbing_wiki_ID
    m = re.match(r"mobbing_(inat|wiki)_(\d+)", stem)
    if m:
        source = "inaturalist" if m.group(1) == "inat" else "wikimedia"
        return "mobbing", source, m.group(2)

    # none_wiki_ID
    m = re.match(r"none_wiki_(\d+)", stem)
    if m:
        return "none", "wikimedia", m.group(1)

    # feeding_young_ID or none_ID (numeric only, no descriptive part)
    m = re.match(r"(feeding_young|none)_(\d+)$", stem)
    if m:
        return m.group(1), "inaturalist", m.group(2)

    # Flickr-style descriptive names:
    # <class>_<descriptive-slug>_<flickr_id>_o
    # The slug can contain digits, hyphens, and --- separators.
    # The Flickr photo_id is the last numeric group before _o suffix.
    m = re.match(r"(mobbing|none|feeding_young)_.+?(\d+)_o$", stem)
    if m:
        return m.group(1), "personal", m.group(2)

    return None, None, None


def infer_class_from_folder(filepath):
    """Determine class from the parent folder name."""
    for cls in CLASSES:
        if f"/labeled/{cls}/" in str(filepath):
            return cls
    return None


def main():
    metadata = load_all_metadata()
    print(f"Loaded {len(metadata)} metadata entries from existing manifests")
    print()

    # Scan all folders
    unified_rows = []

    for cls in CLASSES:
        folder = LABELED_DIR / cls
        if not folder.exists():
            continue
        images = sorted(
            f for f in folder.glob("*") if f.suffix.lower() in IMAGE_EXTS
        )
        found = 0
        missing_meta = 0

        for img_path in images:
            filename = img_path.name
            parsed_cls, parsed_source, photo_id = parse_filename(filename)

            # Use folder as source of truth for class
            cls_actual = infer_class_from_folder(img_path) or cls

            # Look up metadata
            meta = metadata.get(photo_id, {})

            row = {
                "filename": filename,
                "local_path": str(img_path),
                "class": cls_actual,
                "source": meta.get("source", parsed_source or "inaturalist"),
                "observation_id": meta.get("observation_id", ""),
                "observation_uri": meta.get("observation_uri", ""),
                "species": meta.get("species", ""),
                "taxon_name": meta.get("taxon_name", ""),
                "photo_id": photo_id or "",
                "photo_url": meta.get("photo_url", ""),
                "photo_license": meta.get("photo_license", ""),
                "photo_attribution": meta.get("photo_attribution", ""),
                "search_query": meta.get("search_query", ""),
            }
            unified_rows.append(row)

            if meta:
                found += 1
            else:
                missing_meta += 1

        print(f"{cls}: {len(images)} images ({found} with metadata, {missing_meta} without)")

    # Write unified manifest
    fields = [
        "filename", "local_path", "class", "source",
        "observation_id", "observation_uri",
        "species", "taxon_name",
        "photo_id", "photo_url",
        "photo_license", "photo_attribution", "search_query",
    ]

    unified_path = DATA_DIR / "manifest_unified.csv"
    with open(unified_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in unified_rows:
            writer.writerow(row)

    print()
    print(f"Written: {unified_path} ({len(unified_rows)} rows)")
    print()
    print("=" * 50)
    print("Final dataset summary:")
    for cls in CLASSES:
        count = sum(1 for r in unified_rows if r["class"] == cls)
        print(f"  {cls}: {count}")
    print(f"  TOTAL: {len(unified_rows)}")
    print("=" * 50)


if __name__ == "__main__":
    main()
