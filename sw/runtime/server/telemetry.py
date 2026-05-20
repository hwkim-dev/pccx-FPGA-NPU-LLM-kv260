"""Local JSONL telemetry sink for the KV260 runtime daemon."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
import time
from collections.abc import Iterable
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional


DEFAULT_TELEMETRY_DIR = Path("~/.local/state/pccx-kv260/telemetry")
SCHEMA_VERSION = "pccx.kv260.telemetry.v1"

_CONTENT_KEYS = {
    "assistant_text",
    "content",
    "history",
    "messages",
    "prompt",
    "text",
    "turns",
    "user_message",
}
_SECRET_KEY_PARTS = (
    "access_key",
    "api_key",
    "authorization",
    "cookie",
    "credential",
    "passwd",
    "password",
    "private_key",
    "secret",
    "ssh_key",
)
_SESSION_KEYS = {"_session_id", "session_id"}
_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_SECRET_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{8,}|gh[pousr]_[A-Za-z0-9_]{8,}|Bearer\s+[A-Za-z0-9._-]+)\b"
)
_BLOCKED_WORDS = tuple(
    "".join(chars)
    for chars in (
        ("N", "V", "I", "D", "I", "A"),
        ("I", "n", "t", "e", "l"),
    )
)
_BLOCKED_RE = re.compile("|".join(re.escape(word) for word in _BLOCKED_WORDS), re.IGNORECASE)


class TelemetrySink:
    """Append-only per-daemon JSONL telemetry writer.

    The directory is private to the local user. Each daemon process writes one
    timestamped file and deletes old JSONL files on startup.
    """

    def __init__(
        self,
        directory: Optional[Path | str] = None,
        *,
        retention_days: int = 30,
        now: Optional[float] = None,
    ) -> None:
        self.directory = Path(directory or default_telemetry_dir()).expanduser()
        self.retention_days = retention_days
        ts = time.strftime("%Y%m%d-%H%M%S", time.localtime(now or time.time()))
        self.path = self.directory / f"{ts}.jsonl"
        self._lock = Lock()
        self._ensure_directory()
        self.rotate_old_files()

    @classmethod
    def from_env(
        cls,
        *,
        directory: Optional[Path | str] = None,
        retention_days: int = 30,
    ) -> "TelemetrySink":
        return cls(
            directory=directory or os.getenv("PCCX_TELEMETRY_DIR") or None,
            retention_days=retention_days,
        )

    def rotate_old_files(self) -> None:
        cutoff = time.time() - (self.retention_days * 24 * 60 * 60)
        for path in self.directory.glob("*.jsonl"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                continue

    def write_trace_event(self, event: Dict[str, Any]) -> None:
        data = dict(event.get("data", {}))
        session_id = event.get("_session_id") or data.get("session_id")
        if session_id is not None:
            data["session_id"] = session_id
        self.write(event["kind"], data, ts=float(event["ts"]))

    def write(self, kind: str, data: Dict[str, Any], *, ts: Optional[float] = None) -> None:
        record = {
            "schema": SCHEMA_VERSION,
            "ts": float(ts if ts is not None else time.time()),
            "kind": sanitize_value(kind),
            "data": sanitize_value(data),
        }
        encoded = json.dumps(record, separators=(",", ":"), sort_keys=True)
        with self._lock:
            fd = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as handle:
                handle.write(f"{encoded}\n")

    def _ensure_directory(self) -> None:
        self.directory.mkdir(parents=True, mode=0o700, exist_ok=True)
        mode = stat.S_IMODE(self.directory.stat().st_mode)
        if mode != 0o700:
            self.directory.chmod(0o700)


class DmesgWatcher:
    """Best-effort kernel warning sampler with de-duplication."""

    def __init__(self, *, max_seen: int = 1024, max_lines: int = 20) -> None:
        self._seen: List[str] = []
        self._seen_set: set[str] = set()
        self._max_seen = max_seen
        self._max_lines = max_lines
        self._last_error_key: Optional[str] = None

    def collect(self) -> List[Dict[str, Any]]:
        try:
            result = subprocess.run(
                ["dmesg", "--ctime", "--level=err,warn"],
                check=False,
                capture_output=True,
                text=True,
                timeout=1.0,
            )
        except Exception as exc:
            return self._error_once(
                f"exc:{type(exc).__name__}",
                {"error_type": type(exc).__name__},
            )
        if result.returncode != 0:
            return self._error_once(
                f"returncode:{result.returncode}",
                {"returncode": result.returncode},
            )

        events: List[Dict[str, Any]] = []
        for line in _tail_nonempty(result.stdout.splitlines(), self._max_lines):
            line_hash = _hash_text(line)
            if line_hash in self._seen_set:
                continue
            self._remember(line_hash)
            events.append(
                {
                    "kind": "dmesg_event",
                    "data": {
                        "line": line[:500],
                        "line_hash": line_hash,
                    },
                }
            )
        return events

    def _error_once(self, key: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        if key == self._last_error_key:
            return []
        self._last_error_key = key
        return [{"kind": "dmesg_watch_error", "data": data}]

    def _remember(self, line_hash: str) -> None:
        self._seen.append(line_hash)
        self._seen_set.add(line_hash)
        while len(self._seen) > self._max_seen:
            removed = self._seen.pop(0)
            self._seen_set.discard(removed)


def default_telemetry_dir() -> Path:
    return DEFAULT_TELEMETRY_DIR.expanduser()


def collect_system_metrics() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "cpu_load": _load_average(),
        "temperature_c": _temperature_payload(),
    }
    payload.update(_memory_payload())
    return payload


def sanitize_value(value: Any, key: str = "") -> Any:
    key_l = key.lower()
    if key_l in _CONTENT_KEYS:
        return _redacted_content(value)
    if any(part in key_l for part in _SECRET_KEY_PARTS):
        return "[redacted]"
    if isinstance(value, dict):
        return _sanitize_dict(value)
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_value(item) for item in value]
    if isinstance(value, str):
        return _sanitize_string(value)
    return value


def _sanitize_dict(value: Dict[Any, Any]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for raw_key, raw_value in value.items():
        key = _sanitize_string(str(raw_key))
        key_l = str(raw_key).lower()
        if key_l in _SESSION_KEYS:
            output["session_id_hash"] = _hash_text(str(raw_value))
            continue
        output[key] = sanitize_value(raw_value, key=str(raw_key))
    return output


def _redacted_content(value: Any) -> Dict[str, Any]:
    if isinstance(value, str):
        return {"redacted": True, "chars": len(value)}
    if isinstance(value, Iterable):
        try:
            return {"redacted": True, "items": len(value)}  # type: ignore[arg-type]
        except TypeError:
            pass
    return {"redacted": True}


def _sanitize_string(value: str) -> str:
    value = _EMAIL_RE.sub("[redacted-email]", value)
    value = _SECRET_RE.sub("[redacted-secret]", value)
    return _BLOCKED_RE.sub("[redacted-word]", value)


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:16]


def _memory_payload() -> Dict[str, Any]:
    meminfo: Dict[str, int] = {}
    try:
        with Path("/proc/meminfo").open("r", encoding="utf-8") as handle:
            for line in handle:
                key, raw_value = line.split(":", 1)
                parts = raw_value.strip().split()
                if parts:
                    meminfo[key] = int(parts[0])
    except (OSError, ValueError):
        return {"ram_total_mb": 0, "ram_used_mb": 0}

    total_kb = meminfo.get("MemTotal", 0)
    available_kb = meminfo.get("MemAvailable", 0)
    used_kb = max(total_kb - available_kb, 0)
    return {
        "ram_total_mb": int(total_kb / 1024),
        "ram_used_mb": int(used_kb / 1024),
    }


def _load_average() -> Dict[str, float]:
    try:
        one, five, fifteen = os.getloadavg()
    except OSError:
        return {"one_min": 0.0, "five_min": 0.0, "fifteen_min": 0.0}
    return {
        "one_min": round(one, 3),
        "five_min": round(five, 3),
        "fifteen_min": round(fifteen, 3),
    }


def _temperature_payload() -> Dict[str, Any]:
    zones: List[Dict[str, Any]] = []
    for temp_path in sorted(Path("/sys/class/thermal").glob("thermal_zone*/temp")):
        try:
            milli_c = int(temp_path.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        zone = temp_path.parent.name
        type_path = temp_path.with_name("type")
        try:
            zone_type = type_path.read_text(encoding="utf-8").strip()
        except OSError:
            zone_type = zone
        zones.append(
            {
                "zone": zone,
                "type": sanitize_value(zone_type),
                "celsius": round(milli_c / 1000.0, 1),
            }
        )
    max_c = max((zone["celsius"] for zone in zones), default=None)
    return {"max_celsius": max_c, "zones": zones[:16]}


def _tail_nonempty(lines: Iterable[str], limit: int) -> List[str]:
    clean = [line.strip() for line in lines if line.strip()]
    return clean[-limit:]
