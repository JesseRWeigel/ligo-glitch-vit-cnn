#!/usr/bin/env python3
"""
Try Gravity Spy's own API to get O4 subject data.

The Gravity Spy project has APIs:
1. gravityspyplus API (newer)
2. Direct Zooniverse search by metadata
3. Panoptes Python client
"""

import json
import sys
import time

import requests

# Try 1: Search Zooniverse subjects by metadata within Gravity Spy project
# Project ID 1104 for Gravity Spy
# The subjects endpoint supports filtering by project

BASE_URL = "https://www.zooniverse.org/api/subjects"
HEADERS = {
    "Accept": "application/vnd.api+json; version=1",
    "Content-Type": "application/json",
}

# Try searching subjects filtered by project
# Zooniverse API: /subjects?project_id=1104&page=X
# But we need to search by gravityspy_id in metadata
# Unfortunately there's no metadata search in the subjects API

# Try 2: GravitySpy API at gravityspy.org
gspy_urls = [
    "https://gravityspyplus.com/api/",
    "https://gravityspy.org/api/",
    "https://gravityspyplus.com/api/v1/",
]

for url in gspy_urls:
    try:
        r = requests.get(url, timeout=10)
        print(f"{url}: HTTP {r.status_code}")
        if r.status_code == 200:
            print(f"  Response: {r.text[:300]}")
    except Exception as e:
        print(f"{url}: {e}")

print()

# Try 3: Panoptes client (if installed)
try:
    from panoptes_client import Panoptes, Project, Subject

    # Connect anonymously
    Panoptes.connect()

    # Get Gravity Spy project
    project = Project.find(1104)
    print(f"Project: {project.display_name}")

    # Try to get a subject by gravityspy_id metadata search
    # This is project-level subject search
    subjects = Subject.where(project_id=1104)
    for i, s in enumerate(subjects):
        if i >= 3:
            break
        print(f"Subject {s.id}: locations={len(s.locations)}")
        print(f"  Metadata: {json.dumps(dict(s.metadata))[:200]}")
        print(f"  Locations: {s.locations}")

except ImportError:
    print("panoptes_client not installed")
except Exception as e:
    print(f"Panoptes error: {e}")

print()

# Try 4: Direct Zooniverse subject set query
# Gravity Spy may organize O4 subjects in a specific subject set
# Get subject sets for project 1104
try:
    r = requests.get("https://www.zooniverse.org/api/subject_sets",
                     params={"project_id": 1104, "page_size": 100},
                     headers=HEADERS, timeout=30)
    if r.status_code == 200:
        data = r.json()
        sets = data.get("subject_sets", [])
        print(f"Subject sets for Gravity Spy ({len(sets)}):")
        for s in sets:
            print(f"  [{s['id']}] {s.get('display_name', 'N/A')} "
                  f"(subjects: {s.get('set_member_subjects_count', '?')})")
except Exception as e:
    print(f"Subject sets query failed: {e}")
