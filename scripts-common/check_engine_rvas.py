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
#   2. HARD FAILURE, ZERO TOLERANCE: any RVA literal attributed to a module in
#      ZERO_TOLERANCE_MODULES (client.dll, engine.dll). Those are fully
#      migrated; there is no baseline to grandfather them.
#
#      Attribution follows the ASSIGNMENT (`base = (uintptr_t)cl`), not the
#      variable's name. An earlier name-based version of this check reported
#      client.dll clean while ~21 client.dll sites sat in functions that spell
#      the same variable `base` -- a false all-clear.
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

# Which module a base variable refers to is decided by the nearest preceding
# GetModuleHandleA in the same function -- NOT by the variable's name.
#
# This matters: an earlier pass checked only for names like `clBase` and
# reported client.dll as clean, while ~21 client.dll sites were sitting in
# functions that spell the same thing `base`. A name-based rule cannot see
# those, so it gave a false all-clear.
# `HMODULE cl = GetModuleHandleA("client.dll")` / `s_engine = GetModuleHandleA(...)`
HANDLE_ASSIGN = re.compile(
    r'\b(\w+)\s*=\s*GetModuleHandleA\(\s*"([\w.]+)"\s*\)')

# `const uintptr_t base = reinterpret_cast<uintptr_t>(cl);` -- this is what
# actually decides which module a `base + 0xRVA` refers to. Attributing by the
# nearest preceding GetModuleHandleA instead is wrong: in a long function the
# handle from an unrelated earlier block leaks forward, which mislabelled a
# block of materialsystem reads off `s_msdx11` as client.dll.
BASE_ASSIGN = re.compile(
    r'\b(?:const\s+)?uintptr_t\s+(\w+)\s*=\s*reinterpret_cast<uintptr_t>\(\s*&?(\w+)\s*\)')

# Statics whose module is fixed by declaration rather than by a nearby call.
STATIC_HANDLES = {
    's_msdx11': 'materialsystem_dx11.dll',
    's_engine': 'engine.dll',
}

# A function definition at file or namespace scope, used to reset attribution
# so one function's module never leaks into the next.
FUNCTION_START = re.compile(r'^\s{0,4}(static\s+|inline\s+)*[\w:<>*&]+\s+\w+\s*\([^;]*$')

# Modules whose sites are fully migrated. Any hit attributed to one of these
# is a regression, with no baseline to grandfather it.
ZERO_TOLERANCE_MODULES = {'client.dll', 'engine.dll'}

# Known-outstanding hits, by repo-relative path.
#
# client.dll and engine.dll are DONE -- see ZERO_TOLERANCE_MODULES. What is
# left is studiorender.dll (22) and materialsystem_dx11.dll (44).
#
# Those are a genuinely different case from the others, and the reason they are
# listed rather than disabled: their RVAs are still CORRECT on the shipped
# build. Checked against studiorender.dll v2.0.11.0, 0xDE10, 0x11CB0, 0x120B0,
# 0x15A60, 0x15C00 and 0x15D10 are all exact function entries, so those hooks
# install and work today. client.dll and engine.dll had drifted to a different
# compilation; these two modules had not.
#
# So there is nothing to repair here, only brittleness to remove -- and
# swapping a working hook for an unverified signature would trade a latent
# problem for an immediate one. They need anchors derived against their own
# binaries, then the literal dropped and this number lowered.
BASELINE = {
    'src/d3d11/d3d11_rtx.cpp': 66,
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

        handle_module = dict(STATIC_HANDLES)   # handle var -> module
        base_module = {}                       # base var   -> module
        for number, line in enumerate(text.splitlines(), 1):
            stripped = line.lstrip()
            is_comment = stripped.startswith('//') or stripped.startswith('*')

            if not is_comment:
                if FUNCTION_START.match(line) and 'GetModuleHandleA' not in line:
                    handle_module = dict(STATIC_HANDLES)
                    base_module = {}
                for var, module in HANDLE_ASSIGN.findall(line):
                    handle_module[var] = module
                for base_var, src_var in BASE_ASSIGN.findall(line):
                    if src_var in handle_module:
                        base_module[base_var] = handle_module[src_var]
                    else:
                        base_module.pop(base_var, None)

            if '0x' not in line or is_comment:
                continue  # a comment describing history is not a reach

            for match in ANY_BASE_RVA.finditer(line):
                module = base_module.get(match.group(1))
                any_hits.setdefault(rel(path), []).append(
                    (number, match.group(0), line.strip(), module))
                if module in ZERO_TOLERANCE_MODULES:
                    client_hits.append(
                        (rel(path), number, module, line.strip()))
    return any_hits, client_hits


def main():
    show_all = '--list' in sys.argv
    any_hits, client_hits = scan()
    failures = []

    if client_hits:
        failures.append(
            'RVA literals are not allowed for these modules -- their sites have '
            'been migrated to dxvk::EngineSymbols:')
        for path, number, module, text in client_hits:
            failures.append('    %s:%d [%s]: %s' % (path, number, module, text))

    for path, hits in sorted(any_hits.items()):
        allowed = BASELINE.get(path, 0)
        if len(hits) > allowed:
            failures.append(
                '%s has %d module+RVA literals, baseline allows %d. New reaches '
                'into a game module must go through dxvk::EngineSymbols and be '
                'declared in src/dxvk/rtx_render/rtx_engine_symbols_tf2.h.'
                % (path, len(hits), allowed))
            for number, token, text, module in hits[allowed:]:
                failures.append('    %s:%d [%s]: %s'
                                % (path, number, module or 'unknown', text))

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
            for number, token, text, module in hits:
                print('    %d [%s]: %s' % (number, module or 'unknown', text))

    if failures:
        print('check_engine_rvas: FAILED')
        for line in failures:
            print(line)
        return 1

    total = sum(len(v) for v in any_hits.values())
    by_module = {}
    for hits in any_hits.values():
        for _, _, _, module in hits:
            by_module[module or 'unknown'] = by_module.get(module or 'unknown', 0) + 1
    summary = ', '.join('%s=%d' % kv for kv in sorted(by_module.items()))
    print('check_engine_rvas: OK (%d known-outstanding literals, 0 new; %s)'
          % (total, summary or 'none'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
