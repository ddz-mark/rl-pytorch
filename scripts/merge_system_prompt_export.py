#!/usr/bin/env python3
"""
Append cloud-agent-only context to a locally exported Cursor system snapshot.

Typical use (from repo root):
  python scripts/merge_system_prompt_export.py \\
    system_prompt_session_snapshot.txt \\
    -o system_prompt_merged.txt

Use a custom cloud supplement:
  python scripts/merge_system_prompt_export.py snapshot.txt -s scripts/my_cloud.txt -o merged.txt

The local snapshot (communication, citing_code, user_rules, agent_skills, mcp_file_system, ...)
is unchanged; the supplement documents what cloud sessions often add so diffs are comparable.
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_SUPPLEMENT = Path(__file__).resolve().parent / "cloud_agent_supplement.default.txt"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "snapshot",
        type=Path,
        help="Path to system_prompt_session_snapshot.txt (or equivalent export).",
    )
    p.add_argument(
        "-s",
        "--supplement",
        type=Path,
        default=DEFAULT_SUPPLEMENT,
        help=f"Cloud supplement file (default: {DEFAULT_SUPPLEMENT.name})",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Merged output path.",
    )
    args = p.parse_args()

    snap = args.snapshot.read_text(encoding="utf-8")
    supp = args.supplement.read_text(encoding="utf-8")
    merged = snap.rstrip() + "\n\n" + supp.strip() + "\n"
    args.output.write_text(merged, encoding="utf-8")
    print(f"Wrote {args.output} ({len(merged)} chars)")


if __name__ == "__main__":
    main()
