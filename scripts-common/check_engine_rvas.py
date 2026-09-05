#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# NV-DXVK [EngineSymbols] -- enforce the no-hardcoded-RVA rule.
#
# THE RULE
#   The solution must not be hardcoded; if things move, Remix must not break.
#
# WHY THIS SCRIPT EXISTS
#   Remix reaches into the game's own modules (client.dll, engine.dll,
#   studiorender.dll) to read camera state, patch cull sites and install
#   trampolines. Historically every reach was a raw
#   `GetModuleHandleA("client.dll") + 0xRVA` reverse-engineered from ONE
#   compilation of the game.
#
#   That went wrong exactly as you would expect. On the shipped v2.0.11.0
#   client.dll every one of those RVAs lands mid-instruction, by inconsistent
#   deltas -- a different compilation, not a rebase, so no amount of adjusting
#   fixes it. Most sites byte-checked their target and silently switched
#   themselves off. One CALLED its address, entered a function body having
#   skipped the prologue, and crashed on the epilogue's `movaps xmm6,[rsp]`
#   every frame.
#
#   A reviewer cannot catch a re-introduced RVA by reading a diff -- it looks
#   exactly like the code around it. So it is checked mechanically.
#
# WHAT IS ENFORCED
#   1. HARD FAILURE: any new `<module base> + 0xNNNN` literal, anywhere in
#      src/, that is not in the baseline below. New reaches into a game module
#      must go through dxvk::EngineSymbols and be declared in
#      src/dxvk/rtx_render/rtx_engine_symbols_tf2.h.
#   2. HARD FAILURE, ZERO TOLERANCE: any client.dll RVA literal at all. Those
#      have all been migrated; there is no baseline to grandfather.
#   3. The baseline may only shrink. If it is stale (a listed file now has
#      fewer hits), the script says so and asks you to lower the number, so
#      migrating code ratchets the limit down and cannot silently regress.
#
# USAGE
#   python scripts-common/check_engine_rvas.py            # check
#   python scripts-common/check_engine_rvas.py --list     # show every hit
#
# Exit status 0 = clean, 1 = violation.

import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, 'src')

# The resolver and its symbol table are the one place allowed to talk about
# where things live -- and even there it is signatures and anchors, not RVAs.
EXEMPT_BASENAMES = {
    'rtx_engine_symbols.h',
    'rtx_engine_symbols.cpp',
    'rtx_engine_symbols_tf2.h',
}

SCANNED_EXTENSIONS = ('.cpp', '.h', '.hpp', '.inl')

# Vendored/generated trees. None of them reach into a game module, and
# usd-plugins alone is ~300 MB, which turns a sub-second check into minutes.
PRUNED_DIRS = {
    '.git', 'tracy', 'nvtx3', 'usd-plugins',
    'rtx_shaders', 'generated',
}

# `<something that holds a module base> + 0xNNNN`
ANY_BASE_RVA = re.compile(
    r'\b(clBase|cliBase|clientBase|enBase|engBase|srBase|base)\s*\+\s*0x[0-9A-Fa-f]{4,}')

# client.dll specifically: fully migrated, so any hit is a regression.
CLIENT_BASE_RVA = re.compile(
    r'\b(clBase|cliBase|clientBase)\s*\+\s*0x[0-9A-Fa-f]{4,}')

# Known-outstanding hits, by repo-relative path. These are engine.dll and
# studiorender.dll reaches that have NOT been migrated yet.
#
# They are listed rather than fixed because they cannot be verified blind:
# unlike the client.dll sites (all provably dead on the shipped build, so
# disabling them cost nothing), several of these are LIVE -- notably the
# engine.dll R_DrawWorldMeshes trampoline, whose byte check passes today and
# which feeds the main camera. Replacing a working hook with a signature that
# has not been proven unique in that module would trade a silent staleness bug
# for an immediate visible regression. They need the same treatment as
# client.dll: identify each target on the shipped binary, register an anchor or
# signature in rtx_engine_symbols_tf2.h, drop the literal, lower this number.
BASELINE = {
    'src/d3d11/d3d11_rtx.cpp': 99,
}


def iter_source_files():
    for dirpath, dirnames, filenames in os.walk(SRC_ROOT):
        dirnames[:] = [d for d in dirnames if d not in PRUNED_DIRS]
        for name in filenames:
            if not name.endswith(SCANNED_EXTENSIONS):
                continue
            if name in EXEMPT_BASENAMES:
                continue
            yield os.path.join(dirpath, name)


def rel(path):
    return os.path.relpath(path, REPO_ROOT).replace(os.sep, '/')


def scan():
    any_hits = {}
    client_hits = []
    for path in iter_source_files():
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as handle:
                text = handle.read()
        except OSError as exc:
            print('warning: could not read %s: %s' % (rel(path), exc))
            continue

        # Cheap prefilter: a file with no hex literal at all cannot match, and
        # that is the overwhelming majority of them.
        if '0x' not in text:
            continue

        for number, line in enumerate(text.splitlines(), 1):
            if '0x' not in line:
                continue
            stripped = line.lstrip()
            if stripped.startswith('//') or stripped.startswith('*'):
                continue  # a comment describing history is not a reach
            for match in ANY_BASE_RVA.finditer(line):
                any_hits.setdefault(rel(path), []).append(
                    (number, match.group(0), line.strip()))
            if CLIENT_BASE_RVA.search(line):
                client_hits.append((rel(path), number, line.strip()))
    return any_hits, client_hits


def main():
    show_all = '--list' in sys.argv
    any_hits, client_hits = scan()
    failures = []

    if client_hits:
        failures.append(
            'client.dll RVA literals are not allowed -- every client.dll site '
            'has been migrated to dxvk::EngineSymbols:')
        for path, number, text in client_hits:
            failures.append('    %s:%d: %s' % (path, number, text))

    for path, hits in sorted(any_hits.items()):
        allowed = BASELINE.get(path, 0)
        if len(hits) > allowed:
            failures.append(
                '%s has %d module+RVA literals, baseline allows %d. New reaches '
                'into a game module must go through dxvk::EngineSymbols and be '
                'declared in src/dxvk/rtx_render/rtx_engine_symbols_tf2.h.'
                % (path, len(hits), allowed))
            for number, token, text in hits[allowed:]:
                failures.append('    %s:%d: %s' % (path, number, text))

    for path, allowed in sorted(BASELINE.items()):
        found = len(any_hits.get(path, []))
        if found < allowed:
            failures.append(
                'Baseline for %s is stale: %d literals remain but it allows %d. '
                'Lower it to %d so the limit ratchets down.'
                % (path, found, allowed, found))
        if found == 0 and allowed > 0:
            failures.append('  (%s is now clean -- drop its baseline entry.)' % path)

    if show_all:
        for path, hits in sorted(any_hits.items()):
            print('%s: %d' % (path, len(hits)))
            for number, token, text in hits:
                print('    %d: %s' % (number, text))

    if failures:
        print('check_engine_rvas: FAILED')
        for line in failures:
            print(line)
        return 1

    total = sum(len(v) for v in any_hits.values())
    print('check_engine_rvas: OK (%d known-outstanding literals, 0 new, '
          '0 in client.dll)' % total)
    return 0


if __name__ == '__main__':
    sys.exit(main())
