#!/usr/bin/env python3
"""
Retry-download the failed Wikimedia Commons mobbing images from the manifest.

Wikimedia's upload.wikimedia.org aggressively rate-limits bulk downloads.
This script reads manifest_mobbing.csv, finds entries with empty local_path
(Wikimedia source only), and retries with long backoff + thumbnail URLs
(Wikimedia recommends thumbnails over full-res for bulk access).

Usage:
  python3 scripts/download_wmc_retry.py
"""

import csv
import os
import time
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path

MANIFEST = Path("/home/kylelima21/BISONN/data/manifest_mobbing.csv")
MOBBING_DIR = Path("/home/kylelima21/BISONN/data/labeled/mobbing")


def make_thumbnail_url(url, width=1280):
    """Convert a Wikimedia upload URL to a thumbnail URL.

    Example:
      https://upload.wikimedia.org/wikipedia/commons/4/41/File.jpg
      → https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/File.jpg/1280px-File.jpg
    """
    if "/thumb/" in url:
        return url  # already a thumbnail
    if "/commons/" not in url:
        return url

    # Extract the path after /commons/
    parts = url.split("/commons/", 1)
    if len(parts) != 2:
        return url

    suffix = parts[1]  # e.g., "4/41/File.jpg"
    segments = suffix.split("/", 2)  # ["4", "41", "File.jpg"]
    if len(segments) != 3:
        return url

    hash_dir, hash_subdir, filename = segments
    # For SVG/PDF/TIFF, Wikimedia generates PNG thumbnails
    if filename.lower().endswith(".svg"):
        thumb_name = f"{width}px-{filename}.png"
    elif filename.lower().endswith((".tif", ".tiff")):
        thumb_name = f"{width}px-{filename}.png"
    elif filename.lower().endswith(".gif"):
        thumb_name = f"{width}px-{filename}.png"
    else:
        thumb_name = f"{width}px-{filename}"

    thumb_url = (
        f"https://upload.wikimedia.org/wikipedia/commons/thumb/"
        f"{hash_dir}/{hash_subdir}/{filename}/{thumb_name}"
    )
    return thumb_url


def download_with_retry(url, filepath, max_retries=5, base_wait=10):
    """Download with exponential backoff on 429."""
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "BISONN-Research/0.1 (educational; contact: kylelima21@uchicago.edu)",
                },
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = resp.read()
            if len(data) < 2000:
                return False, "file too small"
            with open(filepath, "wb") as f:
                f.write(data)
            return True, "OK"
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = base_wait * (2 ** attempt)
                print(f"\n      429 retry {attempt+1}/{max_retries}, waiting {wait}s...", end="", flush=True)
                time.sleep(wait)
                continue
            return False, f"HTTP {e.code}"
        except Exception as e:
            return False, str(e)[:80]
    return False, "max retries exceeded"


def main():
    # Read manifest
    with open(MANIFEST) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Find failed WMC entries
    failed_wmc = [r for r in rows if r["source"] == "wikimedia" and not r["local_path"]]
    print(f"Failed WMC images to retry: {len(failed_wmc)}")
    print(f"Already downloaded: {sum(1 for r in rows if r['local_path'])}")
    print()

    if not failed_wmc:
        print("Nothing to retry.")
        return

    # Determine the filename from photo_url for each failed entry
    # The original filename pattern was mobbing_wiki_<photo_id>.<ext>
    # We need to reconstruct it since the manifest has empty filename
    downloaded = 0
    still_failed = 0

    for i, record in enumerate(failed_wmc, 1):
        url = record["photo_url"]
        photo_id = record["photo_id"]

        # Determine extension from URL
        ext = ".jpg"
        lower_url = url.lower()
        if ".jpeg" in lower_url:
            ext = ".jpeg"
        elif ".png" in lower_url:
            ext = ".png"
        elif ".gif" in lower_url:
            ext = ".gif"
        elif ".tif" in lower_url:
            ext = ".tif"
        elif ".svg" in lower_url:
            ext = ".png"  # SVG thumbnails are PNG

        filename = f"mobbing_wiki_{photo_id}{ext}"
        filepath = MOBBING_DIR / filename

        # Use thumbnail URL (1280px is plenty for BioCLIP 224px input)
        thumb_url = make_thumbnail_url(url, width=1280)

        print(f"  [{i}/{len(failed_wmc)}] {filename} ", end="", flush=True)

        ok, msg = download_with_retry(thumb_url, filepath)
        if ok:
            record["local_path"] = str(filepath)
            record["filename"] = filename
            record["photo_url"] = thumb_url  # update to actual URL used
            downloaded += 1
            print("OK")
        else:
            print(f"FAILED ({msg})")
            still_failed += 1

            # If we're getting hammered, take a longer break
            if "429" in msg or "max retries" in msg:
                print("  taking a 30s break...")
                time.sleep(30)

        # Be gentle — 2s between downloads
        time.sleep(2)

    # Rewrite manifest with updated paths
    with open(MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=" * 60)
    print(f"WMC retry complete: {downloaded} ok, {still_failed} still failed")
    total_ok = sum(1 for r in rows if r["local_path"])
    print(f"Total mobbing images now: {total_ok}")
    print(f"Manifest updated: {MANIFEST}")
    print("=" * 60)


if __name__ == "__main__":
    main()
