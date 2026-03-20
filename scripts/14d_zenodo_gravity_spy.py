#!/usr/bin/env python3
"""Search Zenodo specifically for Gravity Spy records."""
import requests

# Search with exact phrase
for query in ['"gravity spy"', '"Gravity Spy"', 'title:"Gravity Spy"']:
    r = requests.get("https://zenodo.org/api/records",
                     params={"q": query, "size": 20, "sort": "mostrecent"},
                     timeout=30)
    if r.status_code == 200:
        hits = r.json().get("hits", {}).get("hits", [])
        print(f"Query '{query}': {len(hits)} results")
        for h in hits:
            rid = h['id']
            title = h['metadata']['title'][:100]
            pub_date = h['metadata'].get('publication_date', 'N/A')
            files = h.get("files", [])
            file_info = ", ".join([f"{f['key']} ({f['size']/1e6:.1f}MB)" for f in files[:3]])
            print(f"  [{rid}] {pub_date} | {title}")
            if files:
                print(f"    Files: {file_info}")
        print()

# Also check the known record 5649212 for updates
for rid in [5649212, 7638727, 13904421]:
    r = requests.get(f"https://zenodo.org/api/records/{rid}", timeout=30)
    if r.status_code == 200:
        data = r.json()
        print(f"Record {rid}: {data['metadata']['title'][:80]}")
        print(f"  Published: {data['metadata'].get('publication_date', 'N/A')}")
        print(f"  Description (first 300): {data['metadata'].get('description', '')[:300]}")
        files = data.get("files", [])
        for f in files:
            print(f"  File: {f['key']} ({f['size']/1e6:.1f} MB)")
        print()
