#!/usr/bin/env python3
"""
Download bird mobbing images for BISONN from iNaturalist + Wikimedia Commons.

Mobbing = smaller birds (passerines, crows, etc.) harassing/attacking a
larger bird (raptor, owl, crow, heron, etc.). We target photos showing
at least two individual birds of two species in this dynamic.

Sources:
  1. iNaturalist API — CC-licensed observations matching mobbing terms
     (searches both passerine side and raptor side of the interaction)
  2. Wikimedia Commons API — files matching mobbing search terms
     (typically CC-BY-SA or CC-BY licensed)

Usage:
  source ~/BISONN/venv/bin/activate
  python3 scripts/download_mobbing.py [--max-per-source 150] [--dry-run]
"""

import argparse
import csv
import json
import os
import re
import time
import urllib.request
import urllib.parse
import urllib.error
from pathlib import Path

# ── iNaturalist ────────────────────────────────────────────────────────────

INAT_API = "https://api.inaturalist.org/v1/observations"
CC_LICENSES = "CC-BY,CC-BY-NC,CC-BY-SA,CC0"

# iNaturalist search terms for mobbing.
# Group 1: terms on the passerine/crow side (the mobbers)
# Group 2: terms on the raptor/owl/crow side (the mobbed)
# We search both with Aves filter, and also search without taxon filter
# but keep only bird taxa in the results.
INAT_SEARCHES_AVES = [
    "mobbing",
    "mobbing hawk",
    "mobbing owl",
    "mobbing crow",
    "mobbing raven",
    "mobbing eagle",
    "mobbing heron",
    "crow mobbing",
    "jay mobbing",
    "blackbird mobbing",
    "crows attacking",
    "attacking hawk",
    "attacking owl",
    "attacking raptor",
    "attacking crow",
    "harassing hawk",
    "harassing owl",
    "harassing raptor",
    "birds harassing",
    "dive bombing",
    "scolding hawk",
    "scolding owl",
    "nest defense",
    "group mobbing",
    "flock mobbing",
    "alarm calling",
    "distraction display",
    "anti-predator",
]

# Bird taxon names to filter for when searching all taxa (no Aves restriction)
BIRD_TAXA = [
    "Accipitriformes",   # hawks, eagles, vultures
    "Strigiformes",      # owls
    "Falconiformes",     # falcons
    "Passeriformes",     # perching birds (mobbers)
    "Coraciiformes",     # kingfishers etc.
    "Cathartiformes",    # vultures
    "Anseriformes",      # ducks/geese (sometimes mobbed)
    "Charadriiformes",   # shorebirds (mobbing gulls)
    "Columbiformes",     # pigeons (sometimes mobbed)
    "Suliformes",        # cormorants
    "Pelecaniformes",    # herons, pelicans
    "Gruiformes",        # cranes, rails
]

INAT_SEARCHES_ALL_TAXA = [
    "mobbing",
    "being mobbed",
    "mobbed by birds",
    "mobbed by crows",
    "mobbed by",
    "group mobbing",
    "flock mobbing",
    "dive bombing",
]


def strip_html(text):
    """Remove HTML tags from a string."""
    return re.sub(r"<[^>]+>", "", text).strip()


def search_inat(query, taxon_name=None, per_page=100, page=1):
    """Search iNaturalist observations with photos and CC license filter."""
    params = {
        "q": query,
        "photos": "true",
        "photo_license": CC_LICENSES,
        "per_page": str(per_page),
        "page": str(page),
        "order_by": "votes",
        "order": "desc",
    }
    if taxon_name:
        params["taxon_name"] = taxon_name

    url = f"{INAT_API}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "BISONN-Research/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    results = []
    for obs in data.get("results", []):
        # Check if the taxon is a bird (ancestor check)
        taxon = obs.get("taxon", {})
        taxon_name = taxon.get("name", "")
        ancestors = [a.get("name", "") for a in taxon.get("ancestors", [])]

        # If searching all taxa, only keep bird observations
        if not taxon_name:
            continue
        is_bird = any(
            t in ancestors or t == taxon_name or taxon_name.endswith("formes")
            or taxon_name.endswith("forme")
            for t in BIRD_TAXA
        )
        # Check ancestor names more carefully
        all_taxa_names = [taxon_name] + ancestors
        is_bird = any(
            name in BIRD_TAXA for name in all_taxa_names
        )
        if not is_bird:
            continue

        for photo in obs.get("photos", []):
            license_code = photo.get("license_code")
            if not license_code:
                continue
            photo_url = photo.get("url", "")
            if not photo_url:
                continue
            original_url = photo_url.replace("/square.", "/original.")
            if original_url == photo_url:
                original_url = photo_url.replace("/medium.", "/original.")

            results.append({
                "source": "inaturalist",
                "observation_id": obs.get("id"),
                "observation_uri": obs.get("uri", ""),
                "species": obs.get("species_guess", ""),
                "taxon_name": taxon_name,
                "photo_id": photo.get("id"),
                "photo_url": original_url,
                "photo_license": license_code,
                "photo_attribution": photo.get("attribution", ""),
                "search_query": query,
            })

    return results, data.get("total_results", 0)


# ── Wikimedia Commons ─────────────────────────────────────────────────────

WMC_API = "https://commons.wikimedia.org/w/api.php"

WMC_SEARCHES = [
    "bird mobbing",
    "crow mobbing hawk",
    "crow mobbing owl",
    "crows mobbing",
    "jay mobbing",
    "blackbird mobbing",
    "birds mobbing raptor",
    "birds mobbing owl",
    "birds mobbing hawk",
    "passerine mobbing",
    "mobbing behavior bird",
    "dive bombing bird",
    "birds harassing raptor",
    "songbird mobbing",
    "tern mobbing gull",
    "gull mobbing",
    "magpie mobbing",
    "starling mobbing",
]


def search_wmc(query, limit=50):
    """Search Wikimedia Commons for files matching a query."""
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srnamespace": "6",  # File namespace
        "format": "json",
        "srlimit": str(limit),
    }
    url = f"{WMC_API}?{urllib.parse.urlencode(search_params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "BISONN-Research/0.1"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            titles = [hit["title"] for hit in data.get("query", {}).get("search", [])]
            return titles
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 5 * (attempt + 1)
                print(f"\n    rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            raise


def get_wmc_image_info(titles):
    """Get image URL, license, and metadata for a list of file titles."""
    if not titles:
        return []

    # Process in batches of 20 (API limit)
    all_infos = []
    for i in range(0, len(titles), 20):
        batch = titles[i : i + 20]
        params = {
            "action": "query",
            "titles": "|".join(batch),
            "prop": "imageinfo",
            "iiprop": "url|extmetadata|mime|size",
            "format": "json",
        }
        url = f"{WMC_API}?{urllib.parse.urlencode(params)}"

        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "BISONN-Research/0.1"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read())
                break
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    wait = 5 * (attempt + 1)
                    print(f"\n    rate limited (imageinfo), waiting {wait}s...")
                    time.sleep(wait)
                    continue
                raise

        for _pageid, page in data.get("query", {}).get("pages", {}).items():
            ii = page.get("imageinfo", [{}])[0]
            mime = ii.get("mime", "")
            # Only keep raster images (skip PDFs, videos, SVGs)
            if mime not in ("image/jpeg", "image/png", "image/tiff", "image/gif"):
                continue

            meta = ii.get("extmetadata", {})
            license_name = strip_html(meta.get("LicenseShortName", {}).get("value", ""))

            # Only keep CC-licensed or public domain images
            if not any(cc in license_name.lower() for cc in
                       ["cc-by", "cc0", "public domain", "cc by", "gfdl"]):
                continue

            artist = strip_html(meta.get("Artist", {}).get("value", ""))
            desc = strip_html(meta.get("ImageDescription", {}).get("value", ""))

            all_infos.append({
                "source": "wikimedia",
                "observation_id": "",
                "observation_uri": page.get("title", ""),
                "species": "",
                "taxon_name": "",
                "photo_id": page.get("pageid", ""),
                "photo_url": ii.get("url", ""),
                "photo_license": license_name,
                "photo_attribution": artist,
                "search_query": "",
                "description": desc[:200],
            })

        time.sleep(1)
    # WMC imageinfo also rate-limited — add a wait between batches
    # (the loop above already has the time.sleep(1) at the end)
    # End of batch processing
    return all_infos


# ── Shared utilities ──────────────────────────────────────────────────────

def deduplicate(records):
    """Remove duplicates by photo_url (works across both sources)."""
    seen = set()
    unique = []
    for r in records:
        key = r["photo_url"]
        if key not in seen:
            seen.add(key)
            unique.append(r)
    return unique


def download_image(url, filepath, timeout=60):
    """Download an image to filepath. Returns True on success."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "BISONN-Research/0.1"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if len(data) < 1000:  # skip tiny/broken images
            return False
        with open(filepath, "wb") as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"    FAILED: {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download bird mobbing images from iNaturalist + Wikimedia Commons"
    )
    parser.add_argument("--max-per-source", type=int, default=150,
                        help="Max images per source (default: 150)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Search and show counts without downloading")
    parser.add_argument("--data-dir", type=str, default="/home/kylelima21/BISONN/data",
                        help="Base data directory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    mobbing_dir = data_dir / "labeled" / "mobbing"
    manifest_path = data_dir / "manifest_mobbing.csv"

    print("=" * 60)
    print("BISONN — Mobbing Image Downloader")
    print("Sources: iNaturalist + Wikimedia Commons")
    print("=" * 60)
    print()

    all_records = []

    # ── iNaturalist ────────────────────────────────────────────
    print("--- iNaturalist (Aves-filtered) ---")
    inat_aves = []
    for term in INAT_SEARCHES_AVES:
        results, total = search_inat(term, taxon_name="Aves", per_page=100)
        print(f"  Aves + \"{term}\": {total} total (page 1: {len(results)})")
        inat_aves.extend(results)
        # Fetch page 2 if we got a full page and still need more
        if len(results) >= 100 and len(inat_aves) < args.max_per_source:
            time.sleep(1)
            results2, _ = search_inat(term, taxon_name="Aves", per_page=100, page=2)
            inat_aves.extend(results2)
        time.sleep(1)

    inat_aves = deduplicate(inat_aves)
    inat_aves = inat_aves[: args.max_per_source]
    print(f"  → {len(inat_aves)} unique Aves-filtered images")
    print()

    print("--- iNaturalist (all taxa, bird-only filter) ---")
    inat_all = []
    for term in INAT_SEARCHES_ALL_TAXA:
        results, total = search_inat(term, taxon_name=None, per_page=100)
        print(f"  all taxa + \"{term}\": {total} total (bird-filtered page 1: {len(results)})")
        inat_all.extend(results)
        if len(results) >= 100 and len(inat_all) < args.max_per_source:
            time.sleep(1)
            results2, _ = search_inat(term, taxon_name=None, per_page=100, page=2)
            inat_all.extend(results2)
        time.sleep(1)

    inat_all = deduplicate(inat_all)
    inat_all = inat_all[: args.max_per_source]
    print(f"  → {len(inat_all)} unique all-taxa (bird-filtered) images")
    print()

    # Merge iNaturalist results
    inat_all_records = deduplicate(inat_aves + inat_all)
    inat_all_records = inat_all_records[: args.max_per_source]
    print(f"  iNaturalist total: {len(inat_all_records)} unique images")
    print()

    # ── Wikimedia Commons ──────────────────────────────────────
    print("--- Wikimedia Commons ---")
    wmc_titles = []
    for term in WMC_SEARCHES:
        titles = search_wmc(term, limit=50)
        print(f"  \"{term}\": {len(titles)} files found")
        wmc_titles.extend(titles)
        time.sleep(3)  # WMC rate limiting

    # Deduplicate titles
    wmc_titles = list(set(wmc_titles))
    print(f"  → {len(wmc_titles)} unique file titles")

    # Get image info (URLs, licenses)
    wmc_records = get_wmc_image_info(wmc_titles)
    wmc_records = deduplicate(wmc_records)
    wmc_records = wmc_records[: args.max_per_source]
    print(f"  → {len(wmc_records)} CC-licensed image files")
    print()

    # ── Merge all sources ─────────────────────────────────────
    all_records = deduplicate(inat_all_records + wmc_records)
    print(f"=== Total unique mobbing images: {len(all_records)} ===")
    print()

    if args.dry_run:
        print("Dry run — by source:")
        for src in ("inaturalist", "wikimedia"):
            count = sum(1 for r in all_records if r["source"] == src)
            print(f"  {src}: {count}")
        return

    # ── Download ───────────────────────────────────────────────
    mobbing_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    failed = 0

    for i, record in enumerate(all_records, 1):
        ext = ".jpg"
        url = record["photo_url"].lower()
        if ".jpeg" in url:
            ext = ".jpeg"
        elif ".png" in url:
            ext = ".png"
        elif ".gif" in url:
            ext = ".gif"
        elif ".tif" in url:
            ext = ".tif"

        source_tag = record["source"][:4]  # "inat" or "wiki"
        filename = f"mobbing_{source_tag}_{record['photo_id']}{ext}"
        filepath = mobbing_dir / filename

        print(f"  [{i}/{len(all_records)}] {filename} ", end="", flush=True)

        if download_image(record["photo_url"], filepath):
            record["local_path"] = str(filepath)
            record["filename"] = filename
            downloaded += 1
            print("OK")
        else:
            record["local_path"] = ""
            record["filename"] = ""
            failed += 1

        time.sleep(0.5)

    # ── Write manifest ────────────────────────────────────────
    manifest_fields = [
        "filename", "local_path", "source", "observation_id", "observation_uri",
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
    print(f"Images: {mobbing_dir}")
    print(f"  Mobbing images: {len(list(mobbing_dir.glob('*')))}")
    print("=" * 60)


if __name__ == "__main__":
    main()
