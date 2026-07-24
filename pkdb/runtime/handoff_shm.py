"""Pass handoff breakpoint payload (dict) to the attach helper via POSIX shared memory (no temp JSON)."""

from __future__ import annotations

import json
from multiprocessing import shared_memory
from multiprocessing import resource_tracker
from typing import Any, Dict, Optional, Tuple


def handoff_payload_write_shm(payload: Dict[str, Any], name: Optional[str] = None) -> Tuple[str, int]:
    """
    Serialize *payload* into a new shared memory segment.

    Layout: 8-byte little-endian JSON length, then UTF-8 ``json.dumps`` bytes.

    With ``name=None`` a fresh segment name is generated (initial handoff:
    the name is passed to the child via argv). A caller-chosen ``name`` is
    used to push a follow-up breakpoint update to an already-attached session:
    the receiver derives that same name itself (its own pid plus a sequence
    number), so a bare signal is enough to say "go look".

    :returns: ``(shm_name, nbytes)``; caller must not unlink before child reads.
    """
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    blob = len(raw).to_bytes(8, "little") + raw
    shm = shared_memory.SharedMemory(name=name, create=True, size=len(blob))
    shm.buf[: len(blob)] = blob
    shm_name = shm.name
    nbytes = len(blob)
    shm.close()
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass
    return (shm_name, nbytes)


def handoff_payload_read_shm(name: str, nbytes: int) -> Optional[Dict[str, Any]]:
    """Read and decode handoff dict; unlink the segment. Returns ``None`` on failure."""
    try:
        shm = shared_memory.SharedMemory(name=name, create=False)
    except FileNotFoundError:
        return None
    try:
        buf = bytes(shm.buf[:nbytes])
    finally:
        shm.close()
        try:
            shm.unlink()
        except FileNotFoundError:
            pass
    if len(buf) < 8:
        return None
    plen = int.from_bytes(buf[:8], "little")
    if plen < 0 or 8 + plen > len(buf):
        return None
    try:
        return json.loads(buf[8 : 8 + plen].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
