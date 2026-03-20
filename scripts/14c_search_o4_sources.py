#!/usr/bin/env python3
"""Search for O4 Gravity Spy data sources on Zenodo."""

import requests
import json

# Search Zenodo for gravity spy datasets
searches = [
    "gravity spy O4",
    "gravity spy glitch O4a",
    "gravity spy spectrogram 2024",
    "LIGO glitch classification O4",
]

for query in searches:
    print(f"Search: '{query}'")
    r = requests.get("https://zenodo.org/api/records",
                     params={"q": query, "size": 5, "sort": "mostrecent"},
                     timeout=30)
    if r.status_code == 200:
        hits = r.json().get("hits", {}).get("hits", [])
        for h in hits:
            print(f"  ID: {h['id']}, Title: {h['metadata']['title'][:80]}")
            files = h.get("files", [])
            for f in files:
                print(f"    {f['key']} ({f['size']/1e6:.1f} MB)")
    print()

# Also check the specific Gravity Spy community
r = requests.get("https://zenodo.org/api/records",
                 params={"q": "gravity spy", "size": 20, "sort": "mostrecent",
                          "type": "dataset"},
                 timeout=30)
if r.status_code == 200:
    hits = r.json().get("hits", {}).get("hits", [])
    print(f"Recent Gravity Spy datasets ({len(hits)}):")
    for h in hits:
        print(f"  [{h['id']}] {h['metadata']['title'][:80]}")
        # Check if any file has O4 in the name
        files = h.get("files", [])
        for f in files:
            if "o4" in f["key"].lower() or "O4" in f["key"]:
                print(f"    ** O4 FILE: {f['key']} ({f['size']/1e6:.1f} MB)")
