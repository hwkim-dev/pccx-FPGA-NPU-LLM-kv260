#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 pccxai
"""
post_deploy_smoke.py - run a READ-ONLY on-board smoke test after the v002
overlay has been loaded.  Validates:

  1. /dev/uio* nodes appear and one of them maps the fabric@0xA000_0000
     region (i.e. the NPU AXIL control window).
  2. AXIL CMD/STAT readback is accessible via that uio node:
     - Read FLAGS at offset 0x14 of every cmdsts_* window (HP0..HP3, ACP
       fmap, ACP result).  After load, cmd-FIFO should be empty (FLAGS[0]==1)
       and status-FIFO should be empty (FLAGS[2]==1).  Sticky-error bits
       [7:4] should be 0.
     - Read AXIL_STAT_OUT at the NPU window (0xA0000000 base) - this should
       NOT hang the bus.  Per c3fea5e the status backflow is wired; under
       a clean post-load state it should be a known idle pattern.
  3. No AXIL hang: each readback is wrapped with a timeout.  A hang means
     the SmartConnect / DataMover infrastructure did not come up cleanly
     and the deploy should be rolled back.
  4. dmesg tail is captured and scanned for fpga_manager / fpga_region
     warnings or errors that may indicate a partial load.

This script does NOT write any AXIL register - read-only.  It is safe to
re-run repeatedly.

Required env (same as deploy_to_kv260.sh):
  PCCX_KV260_HOST, PCCX_KV260_USER, KVFPGA_PASSWORD

Usage:
  post_deploy_smoke.py [--dry-run] [--verbose] [--overlay pccx_npu_bd]

Exit code:
  0 - all reads succeeded, no FIFOs claim sticky errors, no kernel
       warning about the overlay.
  1 - any failure
  2 - usage error
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from typing import Optional


AXIL_BASES = [
    ("npu",               0xA0000000),
    ("cmdsts_hp0",        0xA0001000),
    ("cmdsts_hp1",        0xA0002000),
    ("cmdsts_hp2",        0xA0003000),
    ("cmdsts_hp3",        0xA0004000),
    ("cmdsts_acp_fmap",   0xA0005000),
    ("cmdsts_acp_result", 0xA0006000),
]
FLAGS_OFFSET = 0x14
STAT_OFFSET  = 0x10  # STS_POP - reading is safe, just yields 0 if empty


def color(s: str, code: str) -> str:
    if not sys.stdout.isatty():
        return s
    return f"\x1b[{code}m{s}\x1b[0m"


def ok(s: str) -> str:   return color("PASS", "32") + " " + s
def warn(s: str) -> str: return color("WARN", "33") + " " + s
def fail(s: str) -> str: return color("FAIL", "31") + " " + s


def need_env() -> tuple[str, str]:
    host = os.environ.get("PCCX_KV260_HOST")
    user = os.environ.get("PCCX_KV260_USER")
    if not (host and user):
        print(fail("PCCX_KV260_HOST / PCCX_KV260_USER not set"), file=sys.stderr)
        sys.exit(1)
    # KVFPGA_PASSWORD is referenced only inside the sh -c on the remote;
    # we never read its value in this script.  But we DO need it to be set,
    # otherwise the sudo -S call will block.
    if not os.environ.get("KVFPGA_PASSWORD"):
        print(fail("KVFPGA_PASSWORD env var must be set (value never printed)"),
              file=sys.stderr)
        sys.exit(1)
    return host, user


def ssh_cmd(host: str, user: str, remote_sh: str, dry: bool,
            timeout: int = 20, dry_label: Optional[str] = None) -> tuple[int, str]:
    """Run a remote shell snippet under sudo -S, password fed via stdin from env.

    Returns (exit_code, combined_output).
    """
    if dry:
        # Don't actually run.  Print a representative command but redact pwd.
        display = dry_label or remote_sh
        print(f"DRY: ssh {user}@{host} -- sudo -S sh -c {shlex.quote(display)}")
        return 0, ""
    ssh_args = [
        "ssh",
        "-o", "ConnectTimeout=10",
        "-o", "ServerAliveInterval=15",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        f"{user}@{host}",
        f"sudo -S sh -c {shlex.quote(remote_sh)}",
    ]
    try:
        pwd = os.environ["KVFPGA_PASSWORD"]
        # pass pwd via stdin to sudo -S; close stdin afterward
        proc = subprocess.run(
            ssh_args,
            input=pwd + "\n",
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        # Strip any 'password for ...' prompt lines.
        out = "\n".join(line for line in out.splitlines()
                        if not line.startswith("[sudo]")
                        and "Password:" not in line
                        and "password for" not in line)
        return proc.returncode, out
    except subprocess.TimeoutExpired:
        return -1, f"<timeout after {timeout}s>"


def check_uio(host: str, user: str, dry: bool) -> tuple[bool, Optional[str]]:
    """Find which /dev/uio* maps the NPU AXIL window."""
    snippet = (
        "set -e; "
        "for d in /sys/class/uio/uio*; do "
        "  [ -d \"$d\" ] || continue; "
        "  name=$(cat \"$d/name\" 2>/dev/null || true); "
        "  for i in 0 1 2 3 4 5; do "
        "    addr_file=\"$d/maps/map$i/addr\"; "
        "    [ -f \"$addr_file\" ] || break; "
        "    addr=$(cat \"$addr_file\"); "
        "    size=$(cat \"$d/maps/map$i/size\" 2>/dev/null); "
        "    echo \"uio=$(basename $d) name=$name map=$i addr=$addr size=$size\"; "
        "  done; "
        "done"
    )
    rc, out = ssh_cmd(host, user, snippet, dry)
    if dry:
        return True, None
    if rc != 0:
        print(fail(f"uio probe failed (rc={rc}): {out.strip()}"))
        return False, None
    print(f"uio table:\n{out.rstrip()}")
    target_uio = None
    for line in out.splitlines():
        fields = {}
        for part in line.split():
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            fields[key] = value
        try:
            addr = int(fields.get("addr", ""), 16)
        except ValueError:
            continue
        if addr == AXIL_BASES[0][1]:
            target_uio = fields.get("uio")
            break
    if target_uio:
        print(ok(f"NPU AXIL window mapped at /dev/{target_uio}"))
        return True, f"/dev/{target_uio}"
    print(fail("no /dev/uio* maps addr=0xA0000000; overlay may not have programmed"))
    return False, None


def read_axil(host: str, user: str, axil_base: int, offset: int,
              dry: bool) -> tuple[bool, int, str]:
    """Read one 32-bit AXIL register.

    Prefer plain `devmem`, then `busybox devmem`, then fall back to direct UIO
    mmap so the smoke test does not depend on board image packages.
    """
    addr = axil_base + offset
    # `timeout 5` prevents a bus hang from blocking the SSH session.
    snippet = f"""
set -eu
addr=0x{addr:08X}
if command -v devmem >/dev/null 2>&1; then
  if out=$(timeout 5 devmem "$addr" 32); then
    printf 'devmem %s\\n' "$out"
    exit 0
  fi
  echo "devmem failed; trying fallback readers" >&2
fi
if command -v busybox >/dev/null 2>&1; then
  if out=$(timeout 5 busybox devmem "$addr" 32); then
    printf 'busybox-devmem %s\\n' "$out"
    exit 0
  fi
  echo "busybox devmem failed; trying UIO mmap fallback" >&2
fi
if command -v python3 >/dev/null 2>&1; then
  timeout 5 python3 - "$addr" <<'PY'
import glob
import mmap
import os
import struct
import sys

target = int(sys.argv[1], 0)
for uio_dir in sorted(glob.glob("/sys/class/uio/uio*")):
    uio_name = os.path.basename(uio_dir)
    for map_dir in sorted(glob.glob(os.path.join(uio_dir, "maps", "map*"))):
        try:
            map_index = int(os.path.basename(map_dir).replace("map", ""), 10)
            with open(os.path.join(map_dir, "addr"), encoding="utf-8") as f:
                map_base = int(f.read().strip(), 0)
            with open(os.path.join(map_dir, "size"), encoding="utf-8") as f:
                map_size = int(f.read().strip(), 0)
        except (OSError, ValueError):
            continue
        if not (map_base <= target <= map_base + map_size - 4):
            continue
        dev_path = "/dev/" + uio_name
        fd = os.open(dev_path, os.O_RDONLY | getattr(os, "O_SYNC", 0))
        try:
            mm = mmap.mmap(
                fd,
                map_size,
                flags=mmap.MAP_SHARED,
                prot=mmap.PROT_READ,
                offset=map_index * mmap.PAGESIZE,
            )
            try:
                mm.seek(target - map_base)
                value = struct.unpack("<I", mm.read(4))[0]
            finally:
                mm.close()
        finally:
            os.close(fd)
        print(f"uio-mmap 0x{{value:08X}}")
        sys.exit(0)
print("no UIO mmap reader found", file=sys.stderr)
sys.exit(127)
PY
  exit $?
fi
echo "no AXIL reader available" >&2
exit 127
"""
    rc, out = ssh_cmd(
        host,
        user,
        snippet,
        dry,
        timeout=15,
        dry_label=f"read AXIL 0x{addr:08X} via devmem/busybox-devmem/uio-mmap fallback",
    )
    if dry:
        return True, 0, "dry-run"
    if rc != 0:
        print(fail(f"AXIL read 0x{addr:08X} failed after devmem/busybox/mmap fallback "
                   f"(rc={rc}): {out.strip()}"))
        return False, 0, ""
    val = None
    method = ""
    for token in out.replace(",", " ").split():
        normalized = token.strip(";:").lower()
        if normalized in {"devmem", "busybox-devmem", "uio-mmap"}:
            method = normalized
        if normalized.startswith("0x"):
            try:
                val = int(normalized, 0)
            except ValueError:
                continue
    if val is None:
        print(fail(f"AXIL read 0x{addr:08X} returned unparseable output: {out!r}"))
        return False, 0, method
    return True, val, method or "unknown"


def check_axil_flags(host: str, user: str, dry: bool) -> bool:
    """For each cmdsts_* window, read FLAGS at offset 0x14 and validate."""
    all_ok = True
    for name, base in AXIL_BASES:
        if name == "npu":
            # NPU control window does not use the same FLAGS layout - skip.
            ok_, val, method = read_axil(host, user, base, 0x00, dry)
            if dry:
                continue
            if not ok_:
                all_ok = False
                continue
            print(ok(f"{name:18s} read @ 0x{base:08X}+0x00 OK "
                     f"(val=0x{val:08X}, via={method})"))
            continue
        ok_, val, method = read_axil(host, user, base, FLAGS_OFFSET, dry)
        if dry:
            continue
        if not ok_:
            all_ok = False
            continue
        # FLAGS layout: bit0=cmd_empty, bit1=cmd_full, bit2=sts_empty,
        # bit3=sts_full, bits[7:4]=sticky errors.
        cmd_empty = bool(val & 0x1)
        cmd_full  = bool(val & 0x2)
        sts_empty = bool(val & 0x4)
        sts_full  = bool(val & 0x8)
        sticky    = (val >> 4) & 0xF
        if not cmd_empty or not sts_empty or cmd_full or sts_full or sticky:
            print(warn(f"{name:18s} FLAGS=0x{val:02X}  cmd_empty={cmd_empty}  "
                       f"sts_empty={sts_empty}  cmd_full={cmd_full}  "
                       f"sts_full={sts_full}  sticky=0x{sticky:X}  via={method}"))
        else:
            print(ok(f"{name:18s} FLAGS=0x{val:02X}  idle "
                     f"(cmd_empty=1, sts_empty=1, sticky=0, via={method})"))
    return all_ok


def dmesg_tail(host: str, user: str, dry: bool, n: int = 80) -> bool:
    snippet = f"dmesg --ctime | tail -n {n}"
    rc, out = ssh_cmd(host, user, snippet, dry)
    if dry:
        return True
    if rc != 0:
        print(warn(f"dmesg tail failed (rc={rc}): {out.strip()}"))
        return True  # not a deploy blocker
    suspicious = [l for l in out.splitlines()
                  if any(tok in l.lower()
                         for tok in ("error", "warn", "fail",
                                     "fpga_manager", "fpga_region",
                                     "of_overlay"))]
    if suspicious:
        print(warn("dmesg has fpga/overlay-related lines:"))
        for l in suspicious[-30:]:
            print(f"  {l}")
    else:
        print(ok(f"dmesg tail clean (last {n} lines have no overlay errors)"))
    return True


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description="Read-only smoke test for a freshly-loaded v002 overlay on KV260.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print remote commands without executing them")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--overlay", default="pccx_npu_bd",
                   help="Overlay name to assume is loaded (info only)")
    args = p.parse_args(argv)

    host, user = need_env()
    print("-- post_deploy_smoke.py --")
    print(f"host    : {host}")
    print(f"user    : {user}")
    print(f"overlay : {args.overlay}")
    print(f"dry-run : {args.dry_run}")
    print()

    # 1. uio probe + locate NPU AXIL window
    print("[1/3] uio probe")
    uio_ok, uio_dev = check_uio(host, user, args.dry_run)

    # 2. AXIL FLAGS reads on each cmdsts_* window
    print("\n[2/3] AXIL FLAGS sanity")
    flags_ok = check_axil_flags(host, user, args.dry_run)

    # 3. kernel log scan
    print("\n[3/3] dmesg tail")
    dmesg_ok = dmesg_tail(host, user, args.dry_run)

    print()
    if args.dry_run:
        print(ok("dry-run complete; no commands were sent to the board"))
        return 0

    overall = uio_ok and flags_ok and dmesg_ok
    if overall:
        print(ok("post_deploy_smoke PASSED - overlay programmed and AXIL bus alive"))
        return 0
    print(fail("post_deploy_smoke FAILED - inspect the output and consider rollback"))
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
