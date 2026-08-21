#!/usr/bin/env python3
"""Decode NVIDIA Aftermath GPU crash dumps (.nv-gpudmp) to JSON + a summary.

Why this exists
---------------
`dxvk.enableAftermath` defaults to True, so the tree has been WRITING a
`<exe>_<date>-<time>_aftermath.nv-gpudmp` next to the game exe on every
VK_ERROR_DEVICE_LOST for months (see dxvk_instance.cpp, the
GFSDK_Aftermath_EnableGpuCrashDumps callback). Nothing has ever read one.
Each dump names the shader / warp / marker the GPU was executing when it
died, plus page-fault info -- which is exactly the "which submission hung"
question a fence spike leaves open, and it answers it retroactively for
every dump already on disk.

The decoder library (GFSDK_Aftermath_Lib.x64.dll) and its headers are
vendored at external/aftermath, so this needs no SDK install and no
compiler -- it is ctypes against the DLL already in the repo.

Usage
-----
  py scripts-common/aftermath_decode.py <dump.nv-gpudmp> [...]
  py scripts-common/aftermath_decode.py --latest <dir>     # newest dump in dir
  py scripts-common/aftermath_decode.py --all <dir>        # every dump in dir

Writes <dump>.json beside each dump (full decode) and prints the summary.
Add --quiet to skip the summary, --no-json to skip the sidecar.

Shader locations
----------------
When a dump carries an "Active Warps" section, the summary lists the shader
instruction offsets the stalled lanes were parked at, and the decoder is fed
the .nvdbg debug info and .spv binaries needed to resolve them further. Both
are found automatically; override with --debug-info-dir / --spv-dir.

Two things gate an actual source line, and the summary says which one bit:
  - "ACTIVE WARPS none"  -> the dump has no shader address at all. A DMA page
    fault that resets the engine leaves none, and nothing can be resolved.
  - offsets but no line  -> the SPIR-V has no debug info. See
    SHADER_SOURCE_DEBUG_INFO in compile_shaders.py, which turns it on per
    shader (it is ~5.9x the SPIR-V size, so not global).
"""

import argparse
import ctypes
import json
import os
import re
import sys
from ctypes import (CFUNCTYPE, POINTER, Structure, byref, c_char, c_char_p,
                    c_uint32, c_uint64, c_void_p)
from pathlib import Path

# GFSDK_Aftermath_Defines.h
AFTERMATH_VERSION_API = 0x0000212  # 2.18 -- must match the vendored lib
RESULT_SUCCESS = 0x1
RESULT_FAIL = 0xBAD00000

RESULT_NAMES = {
    0x1: "Success",
    0xBAD00000 | 1: "FAIL_VersionMismatch",
    0xBAD00000 | 2: "FAIL_NotInitialized",
    0xBAD00000 | 3: "FAIL_InvalidAdapter",
    0xBAD00000 | 4: "FAIL_InvalidParameter",
    0xBAD00000 | 5: "FAIL_Unknown",
    0xBAD00000 | 6: "FAIL_ApiError",
    0xBAD00000 | 7: "FAIL_NvApiIncompatible",
}

# GFSDK_Aftermath_GpuCrashDumpDecoderFlags
DECODER_FLAGS_ALL = 0x1FFF
DECODER_FLAG_SHADER_MAPPING = 0x100
DECODER_FLAGS_NO_MAPPING = DECODER_FLAGS_ALL & ~DECODER_FLAG_SHADER_MAPPING

# GFSDK_Aftermath_GpuCrashDumpFormatterFlags
FORMAT_UTF8 = 0x2


# --- shader address -> source mapping -------------------------------------
#
# SHADER_MAPPING_INFO turns a shader instruction offset into a source or
# intermediate-assembly line. It needs TWO things and gives nothing without
# either:
#
#   1. Something to map. The offsets live in the dump's "Active Warps"
#      section, which only exists when the GPU could still be asked for SM
#      state at crash time. A DMA page fault that tears the channel down
#      leaves no warps -- and then the decoder does not even CALL the
#      lookups below. If mapping produces nothing, check for warps first;
#      missing shader binaries are the second suspect, not the first.
#
#   2. The shader binaries and the driver's debug info:
#      - shaderDebugInfoLookupCb  <- the .nvdbg files aftermathShaderDebugInfoCallback
#        (dxvk_instance.cpp) writes to <game dir>/shaderDebugInfo/. Because the
#        runtime passes DeferDebugInfoCallbacks, only shaders involved in a
#        crash are dumped, so this directory stays small and IS the right one.
#      - shaderLookupCb           <- the .spv the shader build emits, keyed by
#        GFSDK_Aftermath_GetShaderHashSpirv.
#      - shaderSourceDebugInfoLookupCb is only needed when the shipped SPIR-V
#        was STRIPPED of debug info and the full blob lives elsewhere. This
#        tree never strips, so the full blob goes through shaderLookupCb and
#        this third callback is unnecessary -- which is just as well, since
#        GFSDK_Aftermath_GetDebugNameSpirv is not exported by the vendored lib.
#
# Resolution granularity depends on what the SPIR-V carries. Without OpLine /
# OpString you get "<shader> @ 0x4b730" -- an instruction offset. File:line
# needs the shaders compiled with debug info; see --check-spv-debug.


class _Ident(Structure):
    """GFSDK_Aftermath_ShaderDebugInfoIdentifier"""
    _fields_ = [("id", c_uint64 * 2)]


class _Hash(Structure):
    """GFSDK_Aftermath_ShaderBinaryHash"""
    _fields_ = [("hash", c_uint64)]


class _DebugName(Structure):
    """GFSDK_Aftermath_ShaderDebugName"""
    _fields_ = [("name", c_char * 128)]


class _SpirvCode(Structure):
    """GFSDK_Aftermath_SpirvCode -- pData is union'd with a uint64 for alignment"""
    _fields_ = [("pData", c_void_p), ("size", c_uint32), ("_pad", c_uint32)]


PFN_SET_DATA = CFUNCTYPE(None, c_void_p, c_uint32)
PFN_DEBUG_INFO_LOOKUP = CFUNCTYPE(None, POINTER(_Ident), PFN_SET_DATA, c_void_p)
PFN_SHADER_LOOKUP = CFUNCTYPE(None, POINTER(_Hash), PFN_SET_DATA, c_void_p)
PFN_SOURCE_LOOKUP = CFUNCTYPE(None, POINTER(_DebugName), PFN_SET_DATA, c_void_p)

REPO_ROOT = Path(__file__).resolve().parent.parent
LIB_DIR = REPO_ROOT / "external" / "aftermath" / "lib" / "x64"
LIB_NAME = "GFSDK_Aftermath_Lib.x64.dll"


def result_str(code: int) -> str:
    return RESULT_NAMES.get(code, f"0x{code:08X}")


class ShaderIndex:
    """Supplies the decoder with .nvdbg debug info and .spv binaries on demand.

    Both are keyed by identifiers only Aftermath can compute, so the index is
    built by hashing every candidate file once up front.
    """

    def __init__(self, decoder, debug_info_dirs, spv_dirs):
        self._decoder = decoder
        self.debug_info = {}   # (id0, id1) -> Path
        self.binaries = {}     # aftermath spirv hash -> Path

        # aftermathShaderDebugInfoCallback names these
        # "<id0 padded to 16>-<id1>-0000.nvdbg", so the key is in the filename
        # and no hashing is needed.
        for directory in debug_info_dirs:
            for path in Path(directory).glob("*.nvdbg"):
                match = re.match(r"([0-9A-Fa-f]+)-([0-9A-Fa-f]+)-0000$", path.stem)
                if match:
                    self.debug_info[(int(match.group(1), 16), int(match.group(2), 16))] = path

        for directory in spv_dirs:
            for path in Path(directory).rglob("*.spv"):
                shader_hash = decoder.spirv_hash(path.read_bytes())
                if shader_hash is not None:
                    self.binaries.setdefault(shader_hash, path)

        # Anything handed to setData must outlive the call; hold a reference.
        self._pinned = []
        self.debug_hits, self.debug_misses = [], []
        self.binary_hits, self.binary_misses = [], []
        self.source_requests = []

        # ctypes trampolines must be kept alive for as long as the decoder
        # may call them, so they are instance attributes, not locals.
        self.on_debug_info = PFN_DEBUG_INFO_LOOKUP(self._lookup_debug_info)
        self.on_shader = PFN_SHADER_LOOKUP(self._lookup_binary)
        self.on_source = PFN_SOURCE_LOOKUP(self._lookup_source)

    def _serve(self, path, set_data):
        blob = path.read_bytes()
        buf = (c_char * len(blob)).from_buffer_copy(blob)
        self._pinned.append(buf)
        set_data(ctypes.cast(buf, c_void_p), len(blob))

    def _lookup_debug_info(self, ident, set_data, user_data):
        key = (ident[0].id[0], ident[0].id[1])
        path = self.debug_info.get(key)
        if path is None:
            self.debug_misses.append(key)
            return
        self.debug_hits.append(path.name)
        self._serve(path, set_data)

    def _lookup_binary(self, shader_hash, set_data, user_data):
        key = shader_hash[0].hash
        path = self.binaries.get(key)
        if path is None:
            self.binary_misses.append(key)
            return
        self.binary_hits.append(path.name)
        self._serve(path, set_data)

    def _lookup_source(self, debug_name, set_data, user_data):
        # Only reached for stripped shaders, which this tree does not produce.
        self.source_requests.append(
            bytes(debug_name[0].name).split(b"\x00", 1)[0].decode("ascii", "replace"))

    def report(self):
        return ("shaderIndex: nvdbg={} spv={} | served nvdbg={} spv={}"
                " | missing nvdbg={} spv={}{}").format(
            len(self.debug_info), len(self.binaries),
            len(self.debug_hits), len(self.binary_hits),
            len(self.debug_misses), len(self.binary_misses),
            "" if not self.source_requests
            else f" | sourceDebugName requests={self.source_requests}")


class Decoder:
    """ctypes binding for the crash-dump decoder half of the Aftermath API."""

    def __init__(self, lib_dir: Path = LIB_DIR):
        dll_path = lib_dir / LIB_NAME
        if not dll_path.is_file():
            raise FileNotFoundError(f"Aftermath decoder library not found: {dll_path}")

        # llvm_7_0_1.dll sits next to the lib and is loaded lazily when the
        # decoder disassembles shader binaries. Put the directory on the DLL
        # search path so that load succeeds regardless of the cwd.
        if hasattr(os, "add_dll_directory"):
            self._dll_dir = os.add_dll_directory(str(lib_dir))
        self.lib = ctypes.CDLL(str(dll_path))

        self.lib.GFSDK_Aftermath_GpuCrashDump_CreateDecoder.argtypes = [
            c_uint32, c_void_p, c_uint32, POINTER(c_void_p)]
        self.lib.GFSDK_Aftermath_GpuCrashDump_CreateDecoder.restype = c_uint32

        self.lib.GFSDK_Aftermath_GpuCrashDump_DestroyDecoder.argtypes = [c_void_p]
        self.lib.GFSDK_Aftermath_GpuCrashDump_DestroyDecoder.restype = c_uint32

        self.lib.GFSDK_Aftermath_GpuCrashDump_GenerateJSON.argtypes = [
            c_void_p, c_uint32, c_uint32, PFN_DEBUG_INFO_LOOKUP,
            PFN_SHADER_LOOKUP, PFN_SOURCE_LOOKUP, c_void_p, POINTER(c_uint32)]
        self.lib.GFSDK_Aftermath_GpuCrashDump_GenerateJSON.restype = c_uint32

        self.lib.GFSDK_Aftermath_GpuCrashDump_GetJSON.argtypes = [
            c_void_p, c_uint32, c_char_p]
        self.lib.GFSDK_Aftermath_GpuCrashDump_GetJSON.restype = c_uint32

        self.lib.GFSDK_Aftermath_GetShaderHashSpirv.argtypes = [
            c_uint32, POINTER(_SpirvCode), POINTER(_Hash)]
        self.lib.GFSDK_Aftermath_GetShaderHashSpirv.restype = c_uint32

    def spirv_hash(self, spirv_bytes: bytes):
        """The shader identity Aftermath uses to key shaderLookupCb."""
        buf = (c_char * len(spirv_bytes)).from_buffer_copy(spirv_bytes)
        code = _SpirvCode(ctypes.cast(buf, c_void_p), len(spirv_bytes), 0)
        out = _Hash()
        rc = self.lib.GFSDK_Aftermath_GetShaderHashSpirv(
            AFTERMATH_VERSION_API, byref(code), byref(out))
        return out.hash if rc == RESULT_SUCCESS else None

    def decode(self, dump_bytes: bytes, index: "ShaderIndex" = None) -> dict:
        handle = c_void_p()
        buf = (c_char * len(dump_bytes)).from_buffer_copy(dump_bytes)

        rc = self.lib.GFSDK_Aftermath_GpuCrashDump_CreateDecoder(
            AFTERMATH_VERSION_API, ctypes.cast(buf, c_void_p),
            len(dump_bytes), byref(handle))
        if rc != RESULT_SUCCESS:
            raise RuntimeError(f"CreateDecoder failed: {result_str(rc)}")

        # Asking for SHADER_MAPPING_INFO without lookups is not merely useless,
        # it costs a decode pass -- so only set the flag when we can serve them.
        flags = DECODER_FLAGS_ALL if index is not None else DECODER_FLAGS_NO_MAPPING
        callbacks = ((index.on_debug_info, index.on_shader, index.on_source)
                     if index is not None
                     else (PFN_DEBUG_INFO_LOOKUP(), PFN_SHADER_LOOKUP(), PFN_SOURCE_LOOKUP()))

        try:
            json_size = c_uint32(0)
            rc = self.lib.GFSDK_Aftermath_GpuCrashDump_GenerateJSON(
                handle, flags, FORMAT_UTF8,
                callbacks[0], callbacks[1], callbacks[2], None, byref(json_size))
            if rc != RESULT_SUCCESS:
                raise RuntimeError(f"GenerateJSON failed: {result_str(rc)}")

            out = (c_char * json_size.value)()
            rc = self.lib.GFSDK_Aftermath_GpuCrashDump_GetJSON(
                handle, json_size.value, ctypes.cast(out, c_char_p))
            if rc != RESULT_SUCCESS:
                raise RuntimeError(f"GetJSON failed: {result_str(rc)}")

            text = bytes(out).split(b"\x00", 1)[0].decode("utf-8", "replace")
            return json.loads(text)
        finally:
            self.lib.GFSDK_Aftermath_GpuCrashDump_DestroyDecoder(handle)


def flatten(doc):
    """The decoder returns a LIST of single-key sections, not one dict.

    e.g. [{"Base info": {...}}, {"Page fault info": {...}}, ...]. Fold it into
    one dict so sections can be looked up by name. Sections that repeat are
    collected into a list rather than overwriting each other.
    """
    out = {}
    for section in doc if isinstance(doc, list) else [doc]:
        if not isinstance(section, dict):
            continue
        for key, value in section.items():
            if key in out:
                if not isinstance(out[key], list):
                    out[key] = [out[key]]
                out[key].append(value)
            else:
                out[key] = value
    return out


def as_list(value):
    """Sections are a dict when there is one entry and a list when there are many."""
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def marker_text(event):
    """Aftermath event markers carry the bytes we passed to vkCmdSetCheckpointNV.

    dxvk_scoped_annotation.cpp sets those from the ScopedGpuProfileZone name,
    so this decodes to a zone name like "Primary Rays" or "Reflection PSR".
    """
    lib = event.get("Library")
    if not isinstance(lib, dict):
        return None
    chunk = lib.get("Data", {}).get("Data chunk")
    if not isinstance(chunk, list):
        return None
    raw = bytes(b & 0xFF for b in chunk)
    end = raw.find(0)
    return raw[:end if end >= 0 else len(raw)].decode("ascii", "replace")


def summarize(doc, path):
    """The lines that answer 'what was the GPU doing when it died'."""
    sec = flatten(doc)
    lines = [f"=== {path.name} ==="]

    base = sec.get("Base info", {})
    lines.append("  app={} pid={} api={} dumped={}".format(
        base.get("Application name", "?"), base.get("PID", "?"),
        base.get("Graphics API", "?"), base.get("Dump date", "?")))

    dev = sec.get("Device info", {})
    gpu = sec.get("GPU info", {})
    lines.append("  gpu={} driver={} state={} adapterReset={} engineReset={}".format(
        gpu.get("Adapter name", "?"),
        sec.get("Display driver info", {}).get("Version", "?"),
        dev.get("Device state", "?"),
        dev.get("Adapter reset occurred", "?"),
        dev.get("Engine reset occurred", "?")))

    # Device state is the single most load-bearing field. Error_DMA_PageFault
    # means the GPU dereferenced a bad address -- a different failure from a
    # timeout, and it rules out "the work was merely slow".
    pf = sec.get("Page fault info")
    if pf:
        va = pf.get("GPU virtual address", 0)
        lines.append("  PAGEFAULT va=0x{:016X} engine={} client={}".format(
            va, pf.get("Engine", "?"), pf.get("Client", "?")))
        # A plausible allocation is low and aligned. A huge unaligned VA is a
        # garbage pointer -- i.e. an address READ FROM DATA, not computed from
        # a valid base, which points at corrupted buffer/AS device addresses
        # rather than an out-of-bounds index into a real allocation.
        if va > (1 << 48) or (va & 0xFF):
            lines.append("    ^ implausible address (high and/or unaligned): "
                         "reads as a corrupted device address, not an OOB index")
        res = pf.get("Resource info")
        if res:
            lines.append(f"    resource {res}")
    else:
        lines.append("  PAGEFAULT none")

    shaders = as_list(sec.get("Shader infos"))
    if shaders:
        lines.append(f"  ACTIVE SHADERS ({len(shaders)}):")
        for entry in shaders[:24]:
            info = entry.get("Info", entry) if isinstance(entry, dict) else {}
            lines.append("    name={:<20} type={:<12} hash=0x{:X} size={}".format(
                str(info.get("Shader name", "?")), str(info.get("Shader type", "?")),
                info.get("Shader hash", 0), info.get("Shader size", "?")))
    else:
        lines.append("  ACTIVE SHADERS none")

    # Markers are the breadcrumb trail: the LAST one that is not Finished is
    # the pass the GPU was inside.
    events = []
    for ctx in as_list(sec.get("Aftermath markers")):
        if isinstance(ctx, dict):
            events.extend(ctx.get("Context", {}).get("Events", []))
    marks = []
    for item in events:
        event = item.get("Event", {}) if isinstance(item, dict) else {}
        text = marker_text(event)
        if text is not None:
            marks.append((text, event.get("Status", "?")))
    if marks:
        lines.append(f"  MARKERS ({len(marks)}):")
        for text, status in marks:
            lines.append(f"    {status:<12} {text!r}")
        unfinished = [t for t, s in marks if s != "Finished"]
        if unfinished:
            lines.append("    -> died in: " + ", ".join(repr(t) for t in unfinished))
    else:
        lines.append("  MARKERS none")

    modules = []
    for item in events:
        event = item.get("Event", {}) if isinstance(item, dict) else {}
        stack = event.get("Callstack", {}).get("Stack") if isinstance(event, dict) else None
        for frame in stack or []:
            name = frame.get("Entry", {}).get("Module name")
            if name and name not in modules:
                modules.append(name)
    if modules:
        lines.append("  CALLSTACK MODULES: " + ", ".join(modules))

    # The only section carrying shader instruction offsets. Its presence is
    # what decides whether a line can be pinpointed at all: no warps, no
    # address, and SHADER_MAPPING_INFO has nothing to resolve. Entries read
    # "<shader> @ 0x<offset>" bare, or as a source location once the decoder
    # has both the .nvdbg and a .spv carrying debug info.
    warps = []
    for group in as_list(sec.get("Active Warps")):
        warps.extend(group if isinstance(group, list) else [group])
    if warps:
        # Rank by warp count: the address most lanes are parked at is the
        # one that stalled, not merely one that happened to be in flight.
        def count_of(w):
            return w.get("Warp count", 0) if isinstance(w, dict) else 0

        lines.append(f"  ACTIVE WARPS ({len(warps)} addresses):")
        for warp in sorted(warps, key=count_of, reverse=True)[:12]:
            if isinstance(warp, dict):
                lines.append("    warps={:<4} at {}".format(
                    count_of(warp), warp.get("GPU Address", "?")))
        if len(warps) > 12:
            lines.append(f"    ... {len(warps) - 12} more addresses")
    else:
        lines.append("  ACTIVE WARPS none - no shader address in this dump, "
                     "so no line can be resolved from it")

    known = {"Base info", "Display driver info", "OS info", "GPU info",
             "Page fault info", "Device info", "Aftermath markers",
             "Shader infos", "Active Warps"}
    extra = sorted(set(sec) - known)
    if extra:
        lines.append("  other sections: " + ", ".join(extra))

    return lines


def collect_dumps(args) -> list:
    if args.latest:
        d = Path(args.latest)
        dumps = sorted(d.glob("*.nv-gpudmp"), key=lambda p: p.stat().st_mtime)
        return dumps[-1:]
    if args.all:
        d = Path(args.all)
        return sorted(d.glob("*.nv-gpudmp"), key=lambda p: p.stat().st_mtime)
    return [Path(p) for p in args.dumps]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dumps", nargs="*", help="paths to .nv-gpudmp files")
    ap.add_argument("--latest", metavar="DIR", help="decode the newest dump in DIR")
    ap.add_argument("--all", metavar="DIR", help="decode every dump in DIR")
    ap.add_argument("--lib-dir", default=str(LIB_DIR),
                    help="directory holding GFSDK_Aftermath_Lib.x64.dll")
    ap.add_argument("--spv-dir", action="append", default=[], metavar="DIR",
                    help="directory of built .spv shaders, searched recursively "
                         "(repeatable; defaults to the newest _Comp64* build output)")
    ap.add_argument("--debug-info-dir", action="append", default=[], metavar="DIR",
                    help="directory of .nvdbg files (defaults to "
                         "<dump dir>/shaderDebugInfo)")
    ap.add_argument("--no-map", action="store_true",
                    help="skip shader address -> source mapping")
    ap.add_argument("--no-json", action="store_true", help="do not write the .json sidecar")
    ap.add_argument("--quiet", action="store_true", help="write the sidecar, print nothing")
    args = ap.parse_args()

    paths = collect_dumps(args)
    if not paths:
        ap.error("no dumps selected (pass paths, or --latest DIR / --all DIR)")

    decoder = Decoder(Path(args.lib_dir))

    index = None
    if not args.no_map:
        debug_dirs = [Path(d) for d in args.debug_info_dir]
        if not debug_dirs:
            # Written next to the game exe, i.e. beside the dumps themselves.
            debug_dirs = [p.parent / "shaderDebugInfo" for p in paths]
        debug_dirs = [d for d in dict.fromkeys(debug_dirs) if d.is_dir()]

        spv_dirs = [Path(d) for d in args.spv_dir]
        if not spv_dirs:
            builds = [REPO_ROOT / name / "src" / "dxvk" / "rtx_shaders"
                      for name in ("_Comp64Release", "_Comp64Debug")]
            spv_dirs = [d for d in builds if d.is_dir()]
        spv_dirs = [d for d in spv_dirs if d.is_dir()]

        if debug_dirs or spv_dirs:
            index = ShaderIndex(decoder, debug_dirs, spv_dirs)

    failures = 0

    for path in paths:
        if not path.is_file():
            print(f"!! missing: {path}", file=sys.stderr)
            failures += 1
            continue
        try:
            doc = decoder.decode(path.read_bytes(), index)
        except Exception as exc:  # decode failure on one dump must not stop the rest
            print(f"!! {path.name}: {exc}", file=sys.stderr)
            failures += 1
            continue

        if not args.no_json:
            sidecar = path.with_suffix(path.suffix + ".json")
            sidecar.write_text(json.dumps(doc, indent=2), encoding="utf-8")

        if not args.quiet:
            print("\n".join(summarize(doc, path)))
            if index is not None:
                print("  " + index.report())
            print()

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
