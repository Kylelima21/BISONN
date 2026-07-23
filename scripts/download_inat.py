#!/usr/bin/env python3
"""
Download bird behavior images from iNaturalist for BISONN.

Pulls CC-licensed bird photos for three classes:
  - feeding_young: adult bird feeding nestlings/fledglings, nests with chicks
  - mobbing: birds mobbing/harassing a predator
  - none: perched/flying birds with no interaction (background class)

All images come from iNaturalist with individual CC licenses (CC-BY, CC-BY-NC,
CC-BY-SA, CC0). Attribution and license are recorded in the manifest.

Usage:
  source ~/BISONN/venv/bin/activate
  python3 scripts/download_inat.py [--max-per-class 150] [--dry-run]
"""

import argparse
import csv
import json
import os
import time
import urllib.request
import urllib.parse
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────

INAT_API = "https://api.inaturalist.org/v1/observations"

# CC licenses we accept (CC0 = public domain, CC-BY = attribution,
# CC-BY-NC = non-commercial, CC-BY-SA = share-alike)
CC_LICENSES = "CC-BY,CC-BY-NC,CC-BY-SA,CC0"

# Search terms per class — combined and deduplicated
CLASS_SEARCHES = {
    "feeding_young": [
        "feeding young",
        "feeding chick",
        "feeding nestling",
        "feeding fledgling",
        "nest with chicks",
        "nestling",
        "bird nest feeding",
        "adult feeding",
    ],
    "mobbing": [
        "mobbing",
        "birds attacking",
        "harassing predator",
    ],
    "none": [
        "perched bird",
        "bird perched",
        "bird flying",
    ],
}

# How many to request per API page (iNat max is 200)
PER_PAGE = 100

# Default target per class
DEFAULT_MAX = 150


def search_inat(query, taxon_name="Aves", per_page=PER_PAGE, page=1):
    """Search iNaturalist observations with photos and CC license filter."""
    params = {
        "q": query,
        "taxon_name": taxon_name,
        "photos": "true",
        "photo_license": CC_LICENSES,
        "per_page": str(per_page),
        "page": str(page),
        "order_by": "votes",
        "order": "desc",
    }
    url = f"{INAT_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "BISONN-Research/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    results = []
    for obs in data.get("results", []):
        for photo in obs.get("photos", []):
            license_code = photo.get("license_code")
            if not license_code:
                continue  # skip unlicensed photos
            # Get the original-size URL (replace "square" with "original")
            photo_url = photo.get("url", "")
            if not photo_url:
                continue
            # iNat URLs: .../photos/XXXX/square.jpg → .../photos/XXXX/original.jpg
            original_url = photo_url.replace("/square.", "/original.")
            if original_url == photo_url:
                # fallback: try "medium" if "square" isn't in the URL
                original_url = photo_url.replace("/medium.", "/original.")

            results.append({
                "observation_id": obs.get("id"),
                "observation_uri": obs.get("uri", ""),
                "species": obs.get("species_guess", ""),
                "taxon_name": obs.get("taxon", {}).get("name", ""),
                "photo_id": photo.get("id"),
                "photo_url": original_url,
                "photo_license": license_code,
                "photo_attribution": photo.get("attribution", ""),
                "search_query": query,
                "class": None,  # filled by caller
            })

    return results, data.get("total_results", 0)


def deduplicate(results):
    """Remove duplicate photos by photo_id."""
    seen = set()
    unique = []
    for r in results:
        pid = r["photo_id"]
        if pid not in seen:
            seen.add(pid)
            unique.append(r)
    return unique


def download_image(url, filepath, timeout=30):
    """Download an image to filepath. Returns True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "BISONN-Research/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        with open(filepath, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"    download failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download bird behavior images from iNaturalist")
    parser.add_argument("--max-per-class", type=int, default=DEFAULT_MAX,
                        help=f"Max images per class (default: {DEFAULT_MAX})")
    parser.add_argument("--dry-run", action="store_true",
                        help="Search and show counts without downloading")
    parser.add_argument("--data-dir", type=str, default="/home/kylelima21/BISONN/data",
                        help="Base data directory (default: /home/kylelima21/BISONN/data)")
    args = parser.parse_args()

    data_dir = Path(os.path.expanduser(args.data_dir))
    raw_dir = data_dir / "raw"
    labeled_dir = data_dir / "labeled"

    print("=" * 60)
    print("BISONN — iNaturalist Image Downloader")
    print("=" * 60)
    print(f"Max per class: {args.max_per_class}")
    print(f"Data dir: {data_dir}")
    print()

    # ── Phase 1: Search and collect metadata ──────────────────────────
    all_records = []

    for class_name, search_terms in CLASS_SEARCHES.items():
        print(f"--- {class_name} ---")
        class_results = []
        for term in search_terms:
            # Calculate how many pages we need (but cap at a reasonable number)
            results, total = search_inat(term, per_page=PER_PAGE, page=1)
            print(f"  \"{term}\": {total} total (got {len(results)} from page 1)")

            class_results.extend(results)

            # If there are more pages and we still need more, fetch next pages
            remaining = args.max_per_class - len(class_results)
            pages_needed = min(
                (remaining // PER_PAGE) + 1,
                (total // PER_PAGE) + 1,
                4,  # max 4 pages per search term to be rate-friendly
            )
            for page in range(2, pages_needed + 1):
                if remaining <= 0:
                    break
                time.sleep(1)  # be nice to the API
                results, _ = search_inat(term, per_page=PER_PAGE, page=page)
                class_results.extend(results)
                remaining = args.max_per_class - len(class_results)

            time.sleep(1)  # rate limit between searches

        # Deduplicate within the class
        class_results = deduplicate(class_results)
        # Trim to max
        class_results = class_results[:args.max_per_class]

        # Tag with class name
        for r in class_results:
            r["class"] = class_name

        print(f"  → {len(class_results)} unique images for '{class_name}'")
        all_records.extend(class_results)
        print()

    print(f"Total unique records: {len(all_records)}")
    print()

    if args.dry_run:
        print("Dry run — not downloading. Classes and counts:")
        for class_name in CLASS_SEARCHES:
            count = sum(1 for r in all_records if r["class"] == class_name)
            print(f"  {class_name}: {count}")
        return

    # ── Phase 2: Download images ──────────────────────────────────────
    if not all_records:
        print("No records to download.")
        return

    # Create directory structure
    for class_name in CLASS_SEARCHES:
        (labeled_dir / class_name).mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Download
    downloaded = 0
    failed = 0
    manifest_path = data_dir / "manifest.csv"

    print(f"Downloading {len(all_records)} images...")
    for i, record in enumerate(all_records, 1):
        class_name = record["class"]
        ext = ".jpg"
        if ".jpeg" in record["photo_url"]:
            ext = ".jpeg"
        if ".png" in record["photo_url"]:
            ext = ".png"
        filename = f"{class_name}_{record['photo_id']}{ext}"
        filepath = labeled_dir / class_name / filename

        print(f"  [{i}/{len(all_records)}] {class_name}/{filename} ", end="", flush=True)

        if download_image(record["photo_url"], filepath):
            record["local_path"] = str(filepath)
            record["filename"] = filename
            downloaded += 1
            print("OK")
        else:
            record["local_path"] = ""
            record["filename"] = ""
            failed += 1
            print("FAILED")

        time.sleep(0.5)  # be nice to S3

    # ── Phase 3: Write manifest CSV ───────────────────────────────────
    manifest_fields = [
        "filename", "local_path", "class", "observation_id", "observation_uri",
        "species", "taxon_name", "photo_id", "photo_url",
        "photo_license", "photo_attribution", "search_query",
    ]

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=manifest_fields, extrasaction="ignore")
        writer.writeheader()
        for record in all_records:
            writer.writerow(record)

    print()
    print("=" * 60)
    print(f"Download complete: {downloaded} ok, {failed} failed")
    print(f"Manifest: {manifest_path}")
    print(f"Labeled images: {labeled_dir}")
    print()
    for class_name in CLASS_SEARCHES:
        count = len(list((labeled_dir / class_name).glob("*")))
        print(f"  {class_name}: {count} images")
    print("=" * 60)


if __name__ == "__main__":
    main()
