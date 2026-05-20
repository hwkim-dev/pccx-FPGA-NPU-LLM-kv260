#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 pccxai
"""
pre_deploy_check.py - validate a v002 bitstream + dtbo + bit.bin set before
sending them to a KV260.

Runs entirely on the build host.  Does NOT connect to the board.  Use
post_deploy_smoke.py for on-board checks.

Checks performed:
  1. Bitstream file exists, size in expected range, MD5 reported.
  2. Bitstream header parses; design name, part name, build date and
     payload length are reported.
  3. Part name matches a configurable allow-list (default:
     xck26-sfvc784-2LV-c, the K26 SoM canonical).
  4. Optional: MD5 matches an expected value passed via --expected-md5.
  5. DTBO file exists; magic bytes match Devicetree blob.
  6. bit.bin file exists; size matches the bitstream payload tag and the
     bytes match the bitstream payload section.
  7. shell.json exists and parses, with shell_type == "XRT_FLAT".
  8. KV260 connection environment (PCCX_KV260_HOST, PCCX_KV260_USER) is
     present iff --check-env is set, but the script does NOT contact the
     board.

Exit code:
  0 - all checks PASS
  1 - any check FAIL or any required argument missing
  2 - usage error

The script never prints secrets.  $KVFPGA_PASSWORD and SSH key material
are deliberately not read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


DEFAULT_ALLOWED_PARTS = ("xck26-sfvc784-2LV-c",)
DT_MAGIC = b"\xd0\x0d\xfe\xed"
BIT_MAGIC = bytes.fromhex("0ff00ff00ff00ff000")


@dataclass
class BitHeader:
    design: str
    part: str
    date: str
    time: str
    payload_len: int
    header_len: int  # bytes consumed up to and including the 'e' length field


def parse_bit_header(path: Path) -> BitHeader:
    """Parse the ASCII metadata header of a Xilinx .bit file."""
    with path.open("rb") as f:
        data = f.read(512)
    if len(data) < 16:
        raise ValueError(f"{path}: too small to be a .bit file ({len(data)} B)")
    (magic_len,) = struct.unpack(">H", data[0:2])
    if magic_len != 9 or data[2:11] != BIT_MAGIC:
        raise ValueError(f"{path}: not a Xilinx .bit file (bad magic)")
    # sub-header 0x0001
    (sub,) = struct.unpack(">H", data[11:13])
    if sub != 1:
        raise ValueError(f"{path}: unexpected sub-header {sub}")
    i = 13
    fields = {}
    for _ in range(4):  # 'a','b','c','d'
        tag = data[i:i + 1]
        i += 1
        if tag not in (b"a", b"b", b"c", b"d"):
            raise ValueError(f"{path}: unexpected tag {tag!r}")
        (flen,) = struct.unpack(">H", data[i:i + 2])
        i += 2
        val = data[i:i + flen].rstrip(b"\x00").decode(errors="replace")
        fields[tag.decode()] = val
        i += flen
    # 'e' tag - payload length
    if data[i:i + 1] != b"e":
        raise ValueError(f"{path}: expected 'e' tag at offset {i}, got {data[i:i + 1]!r}")
    i += 1
    (plen,) = struct.unpack(">I", data[i:i + 4])
    i += 4
    design = fields.get("a", "")
    # design name is "name;UserID=...;Version=...;SW_CRC=..." - strip after first ';'
    design_name = design.split(";", 1)[0]
    return BitHeader(
        design=design_name,
        part=fields.get("b", ""),
        date=fields.get("c", ""),
        time=fields.get("d", ""),
        payload_len=plen,
        header_len=i,
    )


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def color(s: str, code: str) -> str:
    if not sys.stdout.isatty():
        return s
    return f"\x1b[{code}m{s}\x1b[0m"


def ok(s: str) -> str: return color("PASS", "32") + " " + s
def warn(s: str) -> str: return color("WARN", "33") + " " + s
def fail(s: str) -> str: return color("FAIL", "31") + " " + s


def check_bitstream(path: Path, allowed_parts: tuple[str, ...],
                    expected_md5: Optional[str], verbose: bool) -> tuple[bool, BitHeader, str]:
    failures = 0

    if not path.exists():
        print(fail(f"bitstream missing: {path}"))
        sys.exit(1)
    size = path.stat().st_size
    print(ok(f"bitstream present: {path}  size={size} B"))

    if size < 1_000_000 or size > 50_000_000:
        print(fail(f"bitstream size {size} outside sane range 1..50 MB"))
        failures += 1

    digest = md5_of(path)
    if expected_md5:
        if digest == expected_md5.lower():
            print(ok(f"bitstream MD5 matches expected: {digest}"))
        else:
            print(fail(f"bitstream MD5 mismatch: got {digest}, expected {expected_md5}"))
            failures += 1
    else:
        print(ok(f"bitstream MD5: {digest}"))

    try:
        hdr = parse_bit_header(path)
    except Exception as exc:
        print(fail(f"bitstream header parse failed: {exc}"))
        sys.exit(1)
    print(ok(f"bitstream design = {hdr.design}"))
    print(ok(f"bitstream part   = {hdr.part}"))
    print(ok(f"bitstream date   = {hdr.date} {hdr.time}"))
    print(ok(f"bitstream payload= {hdr.payload_len} bytes (header consumed {hdr.header_len} B)"))

    if hdr.part not in allowed_parts:
        print(fail(f"bitstream part {hdr.part!r} not in allow-list {allowed_parts}"))
        failures += 1
    expected_total = hdr.header_len + hdr.payload_len
    if size != expected_total:
        print(warn(f"bitstream size {size} != header_len {hdr.header_len} + payload_len {hdr.payload_len} = {expected_total}"))

    if failures:
        return False, hdr, digest
    return True, hdr, digest


def check_dtbo(path: Path) -> bool:
    if not path.exists():
        print(fail(f"dtbo missing: {path}"))
        return False
    with path.open("rb") as f:
        magic = f.read(4)
    if magic != DT_MAGIC:
        print(fail(f"dtbo magic mismatch: got {magic.hex()}, expected {DT_MAGIC.hex()}"))
        return False
    size = path.stat().st_size
    print(ok(f"dtbo present: {path}  size={size} B  magic OK"))
    return True


def check_bitbin(bitbin: Path, bit: Path, hdr: BitHeader) -> bool:
    if not bitbin.exists():
        print(fail(f"bit.bin missing: {bitbin}"))
        return False
    bb_size = bitbin.stat().st_size
    print(ok(f"bit.bin present: {bitbin}  size={bb_size} B"))
    # bootgen-emitted bit.bin should equal the .bit payload (no .bit header).
    if bb_size != hdr.payload_len:
        print(fail(f"bit.bin size {bb_size} != bit payload {hdr.payload_len}; regenerate bit.bin"))
        return False
    # Byte-compare bit.bin with bit's payload section.
    with bit.open("rb") as f:
        f.seek(hdr.header_len)
        bit_payload = f.read(hdr.payload_len)
    with bitbin.open("rb") as f:
        bb = f.read()
    if bb != bit_payload:
        # FPGA Manager on K26 SoM consumes .bit.bin in word-swapped LE form;
        # `bootgen -process_bitstream bin` emits this canonical layout, so
        # bb == swap32(.bit payload) is the EXPECTED relationship.
        bit_le = b"".join(bit_payload[i:i + 4][::-1] for i in range(0, len(bit_payload), 4))
        if bb == bit_le:
            print(ok("bit.bin matches bootgen LE-swapped payload of .bit "
                     "(canonical FPGA Manager format)"))
            return True
        print(fail("bit.bin does not match .bit payload in either byte order — "
                   "regenerate via `make -C sw/dtbo bitbin BIT_SRC=<v002.bit>`"))
        return False
    print(ok("bit.bin contents byte-identical to .bit payload (raw-bin format)"))
    return True


def check_shell_json(path: Path) -> bool:
    if not path.exists():
        print(fail(f"shell.json missing: {path}"))
        return False
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        print(fail(f"shell.json parse error: {exc}"))
        return False
    if data.get("shell_type") != "XRT_FLAT":
        print(fail(f"shell.json shell_type={data.get('shell_type')!r}, expected XRT_FLAT"))
        return False
    print(ok(f"shell.json valid: {data}"))
    return True


def check_env(check_env: bool) -> bool:
    if not check_env:
        return True
    needed = ("PCCX_KV260_HOST", "PCCX_KV260_USER")
    missing = [k for k in needed if not os.environ.get(k)]
    # Never log the values themselves to keep host/user out of evidence logs.
    if missing:
        print(fail(f"env missing: {missing}"))
        return False
    print(ok(f"env present: {list(needed)} (values redacted)"))
    return True


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Validate v002 bitstream / dtbo / bit.bin before KV260 deploy. "
                    "This script never connects to the board.")
    parser.add_argument("--bit", required=True, type=Path,
                        help="Path to .bit file (e.g. hw/build/v002-vivado-*/pccx_v002_system_wrapper.bit)")
    parser.add_argument("--dtbo", type=Path,
                        help="Path to .dtbo file. Defaults to sw/dtbo/build/pccx_npu_bd/pccx_npu_bd.dtbo")
    parser.add_argument("--bitbin", type=Path,
                        help="Path to .bit.bin file. Defaults to sw/dtbo/build/pccx_npu_bd/pccx_npu_bd.bit.bin")
    parser.add_argument("--shell-json", type=Path,
                        help="Path to shell.json. Defaults to sw/dtbo/build/pccx_npu_bd/shell.json")
    parser.add_argument("--expected-md5",
                        help="Expected MD5 of the .bit file. If omitted, just report.")
    parser.add_argument("--allowed-part", action="append",
                        help="Allowed part name(s) (default: xck26-sfvc784-2LV-c). Pass multiple times.")
    parser.add_argument("--check-env", action="store_true",
                        help="Also check that PCCX_KV260_HOST / PCCX_KV260_USER env vars are set "
                             "(values are NEVER printed)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    allowed = tuple(args.allowed_part) if args.allowed_part else DEFAULT_ALLOWED_PARTS

    # Auto-resolve sibling artifacts off the .bit path's worktree root.
    bit = args.bit.resolve()
    worktree = bit
    # Walk upward until we find sw/dtbo/build/pccx_npu_bd or hit fs root
    for parent in [bit.parent] + list(bit.parents):
        candidate = parent / "sw" / "dtbo" / "build" / "pccx_npu_bd"
        if candidate.is_dir():
            worktree = parent
            break

    dtbo = args.dtbo or (worktree / "sw" / "dtbo" / "build" / "pccx_npu_bd" / "pccx_npu_bd.dtbo")
    bitbin = args.bitbin or (worktree / "sw" / "dtbo" / "build" / "pccx_npu_bd" / "pccx_npu_bd.bit.bin")
    shell_json = args.shell_json or (worktree / "sw" / "dtbo" / "build" / "pccx_npu_bd" / "shell.json")

    print(f"-- pre_deploy_check.py --")
    print(f"bit       : {bit}")
    print(f"dtbo      : {dtbo}")
    print(f"bitbin    : {bitbin}")
    print(f"shell.json: {shell_json}")
    print(f"allowed-parts: {allowed}")
    if args.expected_md5:
        print(f"expected-md5 : {args.expected_md5}")
    print()

    ok_bit, hdr, digest = check_bitstream(bit, allowed, args.expected_md5, args.verbose)
    ok_dtbo = check_dtbo(dtbo)
    ok_bitbin = check_bitbin(bitbin, bit, hdr)
    ok_shell = check_shell_json(shell_json)
    ok_env = check_env(args.check_env)

    print()
    summary = {
        "bitstream": ok_bit,
        "dtbo": ok_dtbo,
        "bitbin": ok_bitbin,
        "shell.json": ok_shell,
        "env": ok_env,
    }
    failed = [k for k, v in summary.items() if not v]
    if failed:
        print(fail(f"pre-deploy check failed: {failed}"))
        return 1
    print(ok(f"pre-deploy check PASSED for {bit.name}"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
