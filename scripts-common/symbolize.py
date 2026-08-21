#!/usr/bin/env python3
"""Resolve `module+0xRVA` addresses to symbol + file:line using dbghelp and the PDB.

Why this exists
---------------
The in-DLL crash handlers ([UnhandledException], [StallWatch], the various
stack-capture probes) all emit raw `d3d11.dll+0x98be37` offsets, and reading
those by hand is how a session gets spent guessing. d3d11.pdb ships next to
the built DLL, so the answer is one dbghelp call away. Note the llvm-symbolizer
in this tree is an ARM64 build and cannot load these PDBs -- dbghelp is the
route that works here.

Usage
-----
  py scripts-common/symbolize.py 0x98be37 0x710084 ...
  py scripts-common/symbolize.py --module <path.dll> 0x1234 ...
  echo "<log lines>" | py scripts-common/symbolize.py --stdin

--stdin scrapes every `<module>+0xHEX` it can find, so a whole
[UnhandledException] block or a [StallWatch] line can be piped in as-is.

Defaults to the deployed Titanfall2 d3d11.dll. The PDB must sit beside the DLL
and match the build -- mismatched symbols are worse than none, so the tool
prints the DLL/PDB mtimes and refuses to guess if the PDB is missing.
"""

import argparse
import ctypes
import os
import re
import sys
from ctypes import wintypes
from pathlib import Path

DEFAULT_MODULE = Path(
    r"C:\Users\Friss\Downloads\Compressed\Titanfall-2-Digital-Deluxe-Edition"
    r"-AnkerGames\Titanfall2\bin\x64_retail\d3d11.dll")

MAX_SYM_NAME = 2000

SYMOPT_LOAD_LINES = 0x00000010
SYMOPT_UNDNAME = 0x00000002
SYMOPT_DEFERRED_LOADS = 0x00000004
SYMOPT_NO_PROMPTS = 0x00080000

# Any base works; we resolve base + rva. Pick one far from real modules.
FAKE_BASE = 0x0000700000000000


class SYMBOL_INFO(ctypes.Structure):
    _fields_ = [
        ("SizeOfStruct", wintypes.ULONG),
        ("TypeIndex", wintypes.ULONG),
        ("Reserved", ctypes.c_ulonglong * 2),
        ("Index", wintypes.ULONG),
        ("Size", wintypes.ULONG),
        ("ModBase", ctypes.c_ulonglong),
        ("Flags", wintypes.ULONG),
        ("Value", ctypes.c_ulonglong),
        ("Address", ctypes.c_ulonglong),
        ("Register", wintypes.ULONG),
        ("Scope", wintypes.ULONG),
        ("Tag", wintypes.ULONG),
        ("NameLen", wintypes.ULONG),
        ("MaxNameLen", wintypes.ULONG),
        ("Name", ctypes.c_char * (MAX_SYM_NAME + 1)),
    ]


class IMAGEHLP_LINE64(ctypes.Structure):
    _fields_ = [
        ("SizeOfStruct", wintypes.DWORD),
        ("Key", ctypes.c_void_p),
        ("LineNumber", wintypes.DWORD),
        ("FileName", ctypes.c_char_p),
        ("Address", ctypes.c_ulonglong),
    ]


class Symbolizer:
    def __init__(self, module_path: Path):
        self.module_path = module_path
        if not module_path.is_file():
            raise FileNotFoundError(f"module not found: {module_path}")

        pdb = module_path.with_suffix(".pdb")
        if not pdb.is_file():
            raise FileNotFoundError(
                f"no PDB beside the module ({pdb}). Symbols from a mismatched "
                f"PDB are worse than none, so refusing to continue.")
        self.pdb_path = pdb

        self.dbghelp = ctypes.WinDLL("dbghelp.dll")
        self.dbghelp.SymInitialize.argtypes = [wintypes.HANDLE, ctypes.c_char_p, wintypes.BOOL]
        self.dbghelp.SymInitialize.restype = wintypes.BOOL
        self.dbghelp.SymSetOptions.argtypes = [wintypes.DWORD]
        self.dbghelp.SymSetOptions.restype = wintypes.DWORD
        self.dbghelp.SymLoadModuleEx.argtypes = [
            wintypes.HANDLE, wintypes.HANDLE, ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_ulonglong, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
        self.dbghelp.SymLoadModuleEx.restype = ctypes.c_ulonglong
        self.dbghelp.SymFromAddr.argtypes = [
            wintypes.HANDLE, ctypes.c_ulonglong, ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(SYMBOL_INFO)]
        self.dbghelp.SymFromAddr.restype = wintypes.BOOL
        self.dbghelp.SymGetLineFromAddr64.argtypes = [
            wintypes.HANDLE, ctypes.c_ulonglong, ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(IMAGEHLP_LINE64)]
        self.dbghelp.SymGetLineFromAddr64.restype = wintypes.BOOL

        self.proc = ctypes.windll.kernel32.GetCurrentProcess()
        self.dbghelp.SymSetOptions(
            SYMOPT_LOAD_LINES | SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS | SYMOPT_NO_PROMPTS)

        search = str(module_path.parent).encode()
        if not self.dbghelp.SymInitialize(self.proc, search, False):
            raise OSError(f"SymInitialize failed: {ctypes.GetLastError()}")

        size = module_path.stat().st_size
        self.base = self.dbghelp.SymLoadModuleEx(
            self.proc, None, str(module_path).encode(), None, FAKE_BASE, size, None, 0)
        if self.base == 0:
            raise OSError(f"SymLoadModuleEx failed: {ctypes.GetLastError()}")

    def resolve(self, rva: int) -> str:
        addr = self.base + rva

        sym = SYMBOL_INFO()
        # Real SYMBOL_INFO ends in CHAR Name[1]; ours reserves MAX_SYM_NAME+1, so
        # the declared size is our sizeof minus the extra MAX_SYM_NAME bytes.
        sym.SizeOfStruct = ctypes.sizeof(SYMBOL_INFO) - MAX_SYM_NAME
        sym.MaxNameLen = MAX_SYM_NAME
        displacement = ctypes.c_ulonglong(0)

        name = "<no symbol>"
        if self.dbghelp.SymFromAddr(self.proc, addr, ctypes.byref(displacement), ctypes.byref(sym)):
            name = sym.Name.decode("utf-8", "replace")
            if displacement.value:
                name += f"+0x{displacement.value:x}"

        line_info = IMAGEHLP_LINE64()
        line_info.SizeOfStruct = ctypes.sizeof(IMAGEHLP_LINE64)
        line_disp = wintypes.DWORD(0)
        where = ""
        if self.dbghelp.SymGetLineFromAddr64(
                self.proc, addr, ctypes.byref(line_disp), ctypes.byref(line_info)):
            filename = (line_info.FileName or b"").decode("utf-8", "replace")
            where = f"   {filename}:{line_info.LineNumber}"

        return f"{name}{where}"


# `d3d11.dll+0x98be37`, `d3d11+0x98be37`, or a bare `0x98be37`
ADDR_RE = re.compile(r"(?:([A-Za-z0-9_.\-]+?)(?:\.dll)?\+)?0x([0-9A-Fa-f]+)")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("addresses", nargs="*", help="RVAs, e.g. 0x98be37 or d3d11.dll+0x98be37")
    ap.add_argument("--module", default=str(DEFAULT_MODULE), help="path to the .dll")
    ap.add_argument("--stdin", action="store_true",
                    help="scrape every module+0xHEX out of stdin (paste a whole log block)")
    args = ap.parse_args()

    module = Path(args.module)
    sym = Symbolizer(module)

    print(f"module: {module}")
    print(f"  dll mtime: {module.stat().st_mtime_ns // 10**9}"
          f"   pdb mtime: {sym.pdb_path.stat().st_mtime_ns // 10**9}"
          f"   (these must match the crashing build)")
    print()

    items = []
    if args.stdin:
        text = sys.stdin.read()
        own = module.stem.lower()
        for match in ADDR_RE.finditer(text):
            mod, rva = match.group(1), int(match.group(2), 16)
            # Only our module has symbols here; skip ntdll/kernelbase noise, but
            # keep bare addresses since those are ours by convention.
            if mod is not None and mod.lower() != own:
                continue
            items.append(rva)
    for token in args.addresses:
        match = ADDR_RE.search(token)
        if match:
            items.append(int(match.group(2), 16))

    if not items:
        ap.error("no addresses given")

    seen = set()
    for rva in items:
        if rva in seen:
            continue
        seen.add(rva)
        print(f"+0x{rva:<10x} {sym.resolve(rva)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
