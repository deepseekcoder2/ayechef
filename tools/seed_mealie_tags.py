#!/usr/bin/env python3
"""
Seed canonical cuisine tags into Mealie (idempotent).

Why this exists:
- We intentionally do NOT create tags in the normal tagging pipeline because free-form
  names caused drift/duplicates ("American Chinese" vs "Chinese American").
- We now have a canonical cuisine taxonomy and deterministic formatting.
- This script pre-creates the canonical tags in Mealie once, safely and idempotently.

OpenAPI reference:
- POST /api/organizers/tags with body TagIn: {"name": "<tag name>"}

Usage:
  python tools/seed_mealie_tags.py --apply
  python tools/seed_mealie_tags.py --apply --only-missing
  python tools/seed_mealie_tags.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List

# Ensure project root is on sys.path when running as a script from tools/
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mealie_client import MealieClient
from cuisine_taxonomy import canonical_cuisine_tag_names


def _get_all_tags() -> List[dict]:
    """
    Fetch all tags from Mealie.
    """
    client = MealieClient()
    try:
        return client.get_all_tags()
    finally:
        client.close()


def _existing_tag_names_ci(tags: List[dict]) -> Dict[str, str]:
    """
    Return a case-insensitive name mapping: lower(name) -> original name.
    """
    out: Dict[str, str] = {}
    for t in tags:
        name = t.get("name")
        if isinstance(name, str) and name.strip():
            out[name.strip().lower()] = name.strip()
    return out


def _create_tag(name: str) -> None:
    """
    Create a tag in Mealie.
    Fail-fast on API errors.
    """
    client = MealieClient()
    try:
        client.create_tag(name)
    finally:
        client.close()


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description="Seed canonical cuisine tags into Mealie (idempotent).")
    parser.add_argument("--apply", action="store_true", help="Actually create missing tags (writes to Mealie).")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be created; no writes.")
    parser.add_argument("--only-missing", action="store_true", help="Only print/create missing tags.")
    args = parser.parse_args(argv)

    if args.dry_run and args.apply:
        raise SystemExit("Use either --dry-run or --apply, not both.")

    required = canonical_cuisine_tag_names()
    required_ci = [t.strip() for t in required if isinstance(t, str) and t.strip()]

    tags = _get_all_tags()
    existing_ci = _existing_tag_names_ci(tags)

    missing: List[str] = []
    existing: List[str] = []
    for t in required_ci:
        if t.lower() in existing_ci:
            existing.append(existing_ci[t.lower()])
        else:
            missing.append(t)

    # Get Mealie URL for display
    from config import MEALIE_URL
    print(f"Mealie: {MEALIE_URL}")
    print(f"Required canonical cuisine tags: {len(required_ci)}")
    print(f"Already exists: {len(existing)}")
    print(f"Missing: {len(missing)}")

    if not args.only_missing:
        if existing:
            print("\n--- Existing (sample) ---")
            for t in existing[:25]:
                print(t)

    if not missing:
        print("\n✅ Nothing to do; all required tags already exist.")
        return 0

    print("\n--- Missing ---")
    for t in missing:
        print(t)

    if args.dry_run or not args.apply:
        print("\nℹ️  Dry run mode (no writes). Re-run with --apply to create missing tags.")
        return 0

    # Apply creates
    created = 0
    for t in missing:
        _create_tag(t)
        created += 1

    print(f"\n✅ Created {created} tags.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

