"""Keep report/body.tex in sync with the tables/figures a generator produces."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Sequence

_INPUT_RE = re.compile(r"\\input\{([^}]*)\}")


def ensure_inputs(body_path: Path, targets: Sequence[str]) -> None:
    """Append \\input{target} for any target not already present in body_path."""
    text = body_path.read_text(encoding="utf-8") if body_path.exists() else ""
    existing = set(_INPUT_RE.findall(text))

    missing = [t for t in targets if t not in existing]
    if not missing:
        return

    block = "\n".join(f"\\input{{{t}}}" for t in missing)
    if text and not text.endswith("\n\n"):
        text = text.rstrip("\n") + "\n\n"
    body_path.write_text(text + block + "\n", encoding="utf-8")
