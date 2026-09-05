"""
PSDT consistency checker.

The CPU reference in psdt_ref.py can establish that the maths is right. It
cannot establish that the shader, the header it shares with the C++, and the
C++ that fills that header still agree with each other - and that is where the
bugs actually live in a pass like this. A state slot written under one name and
read under another, a push-constant field the CPU never sets, a binding index
used twice, a constant that drifted between the shader and the reference: none
of those change the reference's answers and all of them break the build or,
worse, silently produce a different transform on the GPU.

v0.1 shipped two such divergences (the shoulder clamp and the black-level
epsilon) and nothing noticed, because nothing was looking.

    python3 tools/psdt/psdt_check.py

Exits non-zero if anything is wrong.
"""

import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SHADER_DIR = os.path.join(ROOT, 'src/dxvk/shaders/rtx/pass/psdt')
TONEMAP_DIR = os.path.join(ROOT, 'src/dxvk/shaders/rtx/pass/tonemap')
CPP = os.path.join(ROOT, 'src/dxvk/rtx_render/rtx_tone_mapping.cpp')
HDR = os.path.join(ROOT, 'src/dxvk/rtx_render/rtx_tone_mapping.h')
PSDT_H = os.path.join(SHADER_DIR, 'psdt.h')
REF = os.path.join(os.path.dirname(__file__), 'psdt_ref.py')

failures = []
notes = []


def read(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def fail(msg):
    failures.append(msg)


def check(label, ok, detail=''):
    mark = 'ok  ' if ok else 'FAIL'
    print(f'  [{mark}] {label}' + (f'   {detail}' if detail else ''))
    if not ok:
        fail(label + (f': {detail}' if detail else ''))


# ---------------------------------------------------------------------------

def check_state_slots(psdt_h):
    """Every AdaptationState slot must be unique and inside the texture."""
    slots = {}
    for name, value in re.findall(r'#define\s+(PSDT_STATE_[A-Z_0-9]+)\s+(\d+)\b', psdt_h):
        # Flag bits, shifts and the pass's own binding indices share the
        # prefix but are not slots.
        if ('FLAG' in name or 'SHIFT' in name or 'BINDING' in name
                or name in ('PSDT_STATE_SIZE', 'PSDT_STATE_GROUP_SIZE')):
            continue
        slots.setdefault(int(value), []).append(name)

    size = int(re.search(r'#define\s+PSDT_STATE_SIZE\s+(\d+)', psdt_h).group(1))
    dupes = {v: n for v, n in slots.items() if len(n) > 1}
    check('state slots are unique', not dupes,
          '; '.join(f'{v}: {", ".join(n)}' for v, n in sorted(dupes.items())))
    over = [f'{n[0]}={v}' for v, n in sorted(slots.items()) if v >= size]
    check(f'state slots fit PSDT_STATE_SIZE ({size})', not over, ', '.join(over))
    return {n[0] for n in slots.values()}


def check_state_flag_packing(psdt_h, cpp):
    """
    The flags word must not have two fields on the same bit.

    PsdtStateArgs is exactly at the 128-byte push constant limit, so every
    boolean, enum and small scalar is bit-packed into one uint. That is a
    perfectly reasonable thing to do and a completely silent thing to get
    wrong: two overlapping fields produce a transform that is wrong in a way
    that looks like a tuning problem, on a machine where the shader cannot be
    compiled to notice. v0.3 repacked the whole word to fit an 8-bit scale
    coherence in, which is exactly the kind of edit this exists for.

    Field widths are read from the trailing "// N bits" comment, so the comment
    is load-bearing rather than decorative.
    """
    fields = {}
    for name, bit in re.findall(r'#define\s+PSDT_STATE_FLAG_(\w+)\s+\(1u\s*<<\s*(\d+)\)', psdt_h):
        fields[f'FLAG_{name}'] = (int(bit), 1)
    for name, shift, width in re.findall(
            r'#define\s+PSDT_STATE_SHIFT_(\w+)\s+(\d+)\s*//\s*(\d+)\s+bit', psdt_h):
        fields[f'SHIFT_{name}'] = (int(shift), int(width))

    check('every packed flags field declares a width', len(fields) >= 9,
          f'{len(fields)} found')

    owner = {}
    clashes = []
    for name, (shift, width) in sorted(fields.items()):
        for bit in range(shift, shift + width):
            if bit >= 32:
                clashes.append(f'{name} runs past bit 31')
            elif bit in owner:
                clashes.append(f'bit {bit}: {owner[bit]} and {name}')
            else:
                owner[bit] = name
    check('no two flags fields share a bit', not clashes, '; '.join(sorted(set(clashes))))

    # And the masks the shader and the C++ apply have to match the declared
    # widths, or a field is silently truncated or reads its neighbour's bits.
    state = read(os.path.join(SHADER_DIR, 'psdt_state.comp.slang'))
    bad = []
    for name, (shift, width) in sorted(fields.items()):
        if not name.startswith('SHIFT_'):
            continue
        want = (1 << width) - 1
        macro = 'PSDT_STATE_' + name
        for text, label in ((state, 'shader'), (cpp, 'C++')):
            for mask in re.findall(r'>>\s*' + macro + r'\)\s*&\s*0x([0-9A-Fa-f]+)u', text):
                if int(mask, 16) != want:
                    bad.append(f'{label} masks {macro} with 0x{mask} not 0x{want:X}')
            for mask in re.findall(r'&\s*0x([0-9A-Fa-f]+)u\)\s*<<\s*' + macro, text):
                if int(mask, 16) != want:
                    bad.append(f'{label} masks {macro} with 0x{mask} not 0x{want:X}')
    check('packed field masks match their declared widths', not bad, '; '.join(bad))

    free = [b for b in range(32) if b not in owner]
    notes.append(f'flags word: {32 - len(free)} of 32 bits used, free: '
                 + (', '.join(str(b) for b in free) if free else 'none'))


def check_bindings(psdt_h):
    """Binding indices must be unique within each descriptor set."""
    groups = {
        'PSDT_ANALYSIS_': 'psdt_analysis',
        'PSDT_DOWNSAMPLE_': 'psdt_downsample',
        'PSDT_STATE_': 'psdt_state',
        'TONEMAPPING_APPLY_': 'tonemap apply',
    }
    tonemap_h = read(os.path.join(TONEMAP_DIR, 'tonemapping.h'))
    source = psdt_h + tonemap_h
    for prefix, label in groups.items():
        found = {}
        for name, value in re.findall(
                r'#define\s+(' + prefix + r'[A-Z_0-9]*(?:INPUT|OUTPUT|INOUT|INPUT_OUTPUT))\s+(\d+)\b',
                source):
            found.setdefault(int(value), []).append(name)
        dupes = {v: n for v, n in found.items() if len(n) > 1}
        check(f'{label} binding indices unique', not dupes,
              '; '.join(f'{v}: {", ".join(n)}' for v, n in sorted(dupes.items())))


def check_state_names_used(psdt_h, slot_names):
    """No shader may read a state slot that does not exist, and no slot should
    be written and never read (a dead slot is a lie in the layout comment)."""
    shader_text = ''
    for name in os.listdir(SHADER_DIR):
        if name.endswith(('.slang', '.slangh')):
            shader_text += read(os.path.join(SHADER_DIR, name))

    used = set(re.findall(r'\b(PSDT_STATE_[A-Z_0-9]+)\b', shader_text))
    used = {u for u in used
            if 'FLAG' not in u and 'SHIFT' not in u and 'BINDING' not in u
            and u not in ('PSDT_STATE_SIZE', 'PSDT_STATE_GROUP_SIZE')}
    unknown = sorted(used - slot_names)
    check('every state slot a shader names is defined', not unknown, ', '.join(unknown))

    written = set(re.findall(r'InOutState\[\s*(PSDT_STATE_[A-Z_0-9]+)\s*\]\s*=', shader_text))
    read_ = set()
    for pat in (r'state\[\s*(PSDT_STATE_[A-Z_0-9]+)\s*\]',
                r'InState\[\s*(PSDT_STATE_[A-Z_0-9]+)\s*\]',
                r'InOutState\[\s*(PSDT_STATE_[A-Z_0-9]+)\s*\](?!\s*=)'):
        read_ |= set(re.findall(pat, shader_text))

    # Read but never written is a real bug: it means a stage is consuming
    # uninitialised state, which on the first frame is whatever the clear left
    # behind and afterwards is last frame's value under a different name.
    unwritten = sorted(read_ - written)
    check('no state slot is read before anything writes it', not unwritten,
          ', '.join(unwritten))

    # Written but never read is not a bug - a state texture carrying frame
    # statistics for inspection is the point of having one - but it should be
    # deliberate, so the list is printed rather than asserted on.
    dead = sorted(written - read_)
    notes.append(f'{len(dead)} state slots are diagnostic-only (written, never read by a stage): '
                 + ', '.join(s.replace('PSDT_STATE_', '') for s in dead))


def check_push_constants(psdt_h, cpp):
    """Every push-constant field must be set by the CPU and read by a shader."""
    shader_text = ''
    for name in os.listdir(SHADER_DIR):
        if name.endswith(('.slang', '.slangh')):
            shader_text += read(os.path.join(SHADER_DIR, name))

    for struct in ('PsdtAnalysisArgs', 'PsdtStateArgs', 'PsdtDownsampleArgs'):
        body = re.search(r'struct\s+' + struct + r'\s*\{(.*?)\n\};', psdt_h, re.S).group(1)
        fields = [m.group(2) for m in re.finditer(
            r'^\s*(float|uint|uvec2|uvec3|uvec4)\s+(\w+)\s*;', body, re.M)]
        fields = [f for f in fields if not f.startswith('pad')]

        unset = [f for f in fields if not re.search(r'pushArgs\.' + f + r'\s*=', cpp)]
        check(f'{struct}: every field is set by the CPU', not unset, ', '.join(unset))

        unread = [f for f in fields if not re.search(r'cb\.' + f + r'\b', shader_text)]
        check(f'{struct}: every field is read by a shader', not unread, ', '.join(unread))

        # 128 bytes is MaxPushConstantSize and the Vulkan guaranteed minimum.
        size = 0
        for m in re.finditer(r'^\s*(float|uint|uvec2|uvec3|uvec4)\s+(\w+)\s*;', body, re.M):
            size += {'float': 4, 'uint': 4, 'uvec2': 8, 'uvec3': 12, 'uvec4': 16}[m.group(1)]
        check(f'{struct}: {size} bytes fits the 128-byte push constant limit', size <= 128,
              f'{size} bytes')


def check_shared_constants():
    """
    Constants that appear in both the shader and the CPU reference have to
    match, or the reference is measuring something the GPU does not run. This
    is the check that would have caught v0.1's shoulder clamp and black-level
    epsilon.
    """
    transform = read(os.path.join(SHADER_DIR, 'psdt_transform.slangh'))
    analysis = read(os.path.join(SHADER_DIR, 'psdt_analysis.comp.slang'))
    state = read(os.path.join(SHADER_DIR, 'psdt_state.comp.slang'))
    space = read(os.path.join(SHADER_DIR, 'psdt_perceptual_space.slangh'))
    ref = read(REF)

    def shader_const(text, name):
        m = re.search(r'static const float\s+' + name + r'\s*=\s*([-0-9.eE]+)\s*;', text)
        return float(m.group(1)) if m else None

    def ref_const(name):
        m = re.search(r'^' + name + r'\s*=\s*([-0-9.eE]+)\s*$', ref, re.M)
        return float(m.group(1)) if m else None

    pairs = [
        ('kPsdtMaxDetailBoost', transform, 'MAX_DETAIL_BOOST'),
        ('kPsdtGlareTintFraction', transform, 'GLARE_TINT_FRACTION'),
        ('kPsdtDepthDeadzone', transform, 'DEPTH_DEADZONE'),
        ('kPsdtScaleDeadzone', transform, 'SCALE_DEADZONE'),
        ('kSkyAdaptationWeight', analysis, 'SKY_ADAPTATION_WEIGHT'),
        ('kEmissiveThresholdDiscount', analysis, 'EMISSIVE_THRESHOLD_DISCOUNT'),
    ]
    bad = []
    for shader_name, text, ref_name in pairs:
        a, b = shader_const(text, shader_name), ref_const(ref_name)
        if a is None or b is None or abs(a - b) > 1e-9:
            bad.append(f'{shader_name}={a} vs {ref_name}={b}')
    check('shared constants match between shader and reference', not bad, '; '.join(bad))

    # Chroma references, per (space, gamut).
    shader_refs = dict(re.findall(
        r'static const float kPsdtChromaRef(\w+)\s*=\s*([0-9.]+)\s*;', space))
    key_map = {'ICtCp709': ('SPACE_ICTCP', 'GAMUT_709'), 'ICtCpP3': ('SPACE_ICTCP', 'GAMUT_P3'),
               'ICtCp2020': ('SPACE_ICTCP', 'GAMUT_2020'), 'Jzazbz709': ('SPACE_JZAZBZ', 'GAMUT_709'),
               'JzazbzP3': ('SPACE_JZAZBZ', 'GAMUT_P3'), 'Jzazbz2020': ('SPACE_JZAZBZ', 'GAMUT_2020')}
    ref_block = re.search(r'CHROMA_REF\s*=\s*\{(.*?)\}', ref, re.S).group(1)
    bad = []
    for key, (sp, gm) in key_map.items():
        want = shader_refs.get(key)
        m = re.search(r'\(' + sp + r',\s*' + gm + r'\):\s*([0-9.]+)', ref_block)
        got = m.group(1) if m else None
        if want is None or got is None or abs(float(want) - float(got)) > 1e-9:
            bad.append(f'{key}: shader {want} vs reference {got}')
    check('chroma references match between shader and reference', not bad, '; '.join(bad))

    # The shoulder clamp: present in both, or in neither.
    clamp_shader = 'clamp(rolloff * highlightRange' in state
    clamp_ref = 'clamp(rolloff * highlightRange' in ref
    check('shoulder clamp present in both shader and reference',
          clamp_shader and clamp_ref,
          f'shader={clamp_shader} reference={clamp_ref}')

    # Black-level epsilon.
    m_s = re.search(r'cb\.displayBlackNits\s*/\s*refWhite,\s*([0-9.eE-]+)\)', state)
    m_r = re.search(r"o\['displayBlackNits'\]\s*/\s*ref,\s*([0-9.eE-]+)\)", ref)
    same = m_s and m_r and abs(float(m_s.group(1)) - float(m_r.group(1))) < 1e-30
    check('black-level epsilon matches', bool(same),
          f'shader={m_s.group(1) if m_s else "?"} reference={m_r.group(1) if m_r else "?"}')


def check_gt7_port():
    """
    The GT7 control has to be a port of the shader that runs.

    This does not verify gt7.slangh against Polyphony's published reference -
    that needs the reference source and is a separate job. What it verifies is
    the layer underneath: that the numbers psdt_suite prints beside PSDT's come
    from the same operator the GPU would run, constant for constant. A control
    that has quietly drifted from the shipped shader is worse than no control.
    """
    gt7 = read(os.path.join(TONEMAP_DIR, 'gt7.slangh'))
    ref = read(REF)

    pairs = [
        ('gt7_midPoint', 'GT7_MID'), ('gt7_linearSection', 'GT7_LINEAR'),
        ('gt7_toeStrength', 'GT7_TOE'), ('gt7_curveKA', 'GT7_KA'),
        ('gt7_curveKB', 'GT7_KB'), ('gt7_curveKC', 'GT7_KC'),
        ('gt7_fadeStart', 'GT7_FADE0'), ('gt7_fadeEnd', 'GT7_FADE1'),
        ('gt7_blendRatio', 'GT7_BLEND'), ('gt7_targetUcs', 'GT7_TARGET_UCS'),
    ]
    bad = []
    for shader_name, ref_name in pairs:
        m = re.search(r'static const float\s+' + shader_name + r'\s*=\s*([-0-9.]+)\s*;', gt7)
        # The reference declares several on one line, e.g. "A, B, C = 1, 2, 3".
        r = re.search(r'^([A-Z0-9_, ]*\b' + ref_name + r'\b[A-Z0-9_, ]*)=\s*(.+)$', ref, re.M)
        got = None
        if r:
            names = [n.strip() for n in r.group(1).split(',')]
            values = [v.strip() for v in r.group(2).split(',')]
            if ref_name in names and len(names) == len(values):
                got = values[names.index(ref_name)]
        if m is None or got is None or abs(float(m.group(1)) - float(got)) > 1e-9:
            bad.append(f'{shader_name}={m.group(1) if m else "?"} vs {ref_name}={got}')
    check('GT7 curve constants match between shader and reference', not bad, '; '.join(bad))

    # The inverse ICtCp constants. gt7.slangh spells its own out to six digits
    # rather than reusing the exact ones; the port has to use the shader's.
    shader_inv = re.findall(r'ictCp\.x\s*[-+]\s*([0-9.]+)\s*\*\s*ictCp\.y', gt7)
    ref_inv = re.search(r'GT7_ICTCP_TO_LMS\s*=\s*\[\[1\.0,\s*([0-9.]+)', ref)
    ok = shader_inv and ref_inv and abs(float(shader_inv[0]) - float(ref_inv.group(1))) < 1e-12
    check('GT7 inverse ICtCp constants match', bool(ok),
          f'shader={shader_inv[0] if shader_inv else "?"} reference={ref_inv.group(1) if ref_inv else "?"}')

    # The Rec.2020 input assumption, which is the thing the gt7space suite
    # section measures. Assert that it is still what the shader says, so that
    # if someone changes it the harness stops describing the old behaviour.
    assumes_2020 = 'assumed to be linear Rec.2020' in gt7
    ref_documents = 'gt7_in_rec2020' in ref
    check('GT7\'s input-space assumption is documented and modelled',
          assumes_2020 and ref_documents,
          f'shader says Rec.2020={assumes_2020}, reference models both={ref_documents}')


def check_probe_math():
    """
    Compile the TonemapProbe's ICtCp out of rtx_context.cpp and check it against
    the reference.

    The probe carries its own small colour-space implementation - it runs on a
    worker thread with no shader includes in scope - and the whole value of it
    is that a number logged from the game is comparable with a number from this
    harness. That only holds if the two agree, and "it looks like the same
    formula" is not agreement.

    This lifts the three lambdas verbatim out of the .cpp, so what is compiled
    is the text that is in the file. It also happens to be the only part of the
    C++ in this change that can be compiled at all on this machine, which makes
    it worth more than its size.
    """
    import shutil
    import subprocess
    import tempfile

    cxx = shutil.which('g++') or shutil.which('clang++')
    if cxx is None:
        notes.append('probe maths not cross-checked: no C++ compiler available')
        return

    src = read(os.path.join(ROOT, 'src/dxvk/rtx_render/rtx_context.cpp'))
    try:
        start = src.index('        auto lum = [](float r, float g, float b) {')
        end = src.index('        // Sparse normalized grid')
    except ValueError:
        check('TonemapProbe maths can be located in rtx_context.cpp', False,
              'the lambdas moved; update check_probe_math')
        return

    probes = [(1, 1, 1), (0.5, 0.1, 0.1), (0.1, 0.5, 0.1),
              (0.1, 0.1, 0.5), (0.18, 0.18, 0.18), (2, 0.5, 0.25)]
    literal = ', '.join('{%ff,%ff,%ff}' % p for p in probes)
    program = (
        '#include <cmath>\n#include <cstdio>\n#include <algorithm>\nint main(){\n'
        + src[start:end]
        + f'\n  const float probes[{len(probes)}][3] = {{{literal}}};\n'
        '  for (int i = 0; i < ' + str(len(probes)) + '; ++i) {\n'
        '    float c = 0.f, h = 0.f;\n'
        '    const float I = ictcp(probes[i][0], probes[i][1], probes[i][2], c, h);\n'
        '    std::printf("%.8f %.8f %.8f %.8f\\n", I, c, h,\n'
        '                lum(probes[i][0], probes[i][1], probes[i][2]));\n'
        '  }\n  return 0;\n}\n')

    with tempfile.TemporaryDirectory() as tmp:
        cpp = os.path.join(tmp, 'probe.cpp')
        exe = os.path.join(tmp, 'probe')
        with open(cpp, 'w') as f:
            f.write(program)
        build = subprocess.run([cxx, '-O2', '-o', exe, cpp],
                               capture_output=True, text=True)
        if build.returncode != 0:
            check('TonemapProbe maths compiles', False,
                  build.stderr.strip().splitlines()[0] if build.stderr else 'unknown error')
            return
        check('TonemapProbe maths compiles', True)
        out = subprocess.run([exe], capture_output=True, text=True).stdout

    sys.path.insert(0, os.path.dirname(__file__))
    import psdt_ref as P

    worst_i = worst_c = worst_h = worst_y = 0.0
    for line, rgb in zip(out.strip().splitlines(), probes):
        ci, cc, ch, cy = (float(x) for x in line.split())
        iab = P.rec709_to_ictcp(list(rgb), 100.0)
        worst_i = max(worst_i, abs(ci - iab[0]))
        worst_c = max(worst_c, abs(cc - P.chroma_of(iab)))
        worst_y = max(worst_y, abs(cy - P.luminance(rgb)))
        # Hue is undefined at zero chroma; the probe's own guard excludes those
        # samples from its statistics, so they are excluded here too.
        if min(cc, P.chroma_of(iab)) > 1e-3:
            worst_h = max(worst_h, abs(ch - P.hue_of(iab)))

    ok = worst_i < 1e-4 and worst_c < 1e-4 and worst_h < 1e-3 and worst_y < 1e-5
    check('TonemapProbe ICtCp agrees with the reference', ok,
          f'I {worst_i:.1e}, C {worst_c:.1e}, H {worst_h:.1e} (chromatic only), Y {worst_y:.1e}')


def check_operators():
    """Operator ids must agree between the shader header and the C++ enum, and
    every id the dispatcher handles must be in the UI."""
    tonemap_h = read(os.path.join(TONEMAP_DIR, 'tonemapping.h'))
    ops_shader = dict((n, int(v)) for n, v in re.findall(
        r'static const uint32_t tonemapOperator(\w+)\s*=\s*(\d+)\s*;', tonemap_h))
    hdr = read(HDR)
    enum_body = re.search(r'enum class TonemapOperator\s*:\s*uint32_t\s*\{(.*?)\n\s*\};',
                          hdr, re.S).group(1)
    ops_cpp = dict((n, int(v)) for n, v in re.findall(r'^\s*(\w+)\s*=\s*(\d+)\s*,', enum_body, re.M))

    mismatch = []
    for name, value in ops_cpp.items():
        if ops_shader.get(name) != value:
            mismatch.append(f'{name}: cpp {value} vs shader {ops_shader.get(name)}')
    check('operator ids agree between the C++ enum and the shader header',
          not mismatch, '; '.join(mismatch))

    cpp = read(CPP)
    missing_ui = [n for n in ops_cpp if f'TonemapOperator::{n}' not in cpp]
    check('every operator appears in the UI dropdown', not missing_ui, ', '.join(missing_ui))

    # Every non-PSDT operator must be reachable from the dispatcher.
    dispatcher = read(os.path.join(TONEMAP_DIR, 'tonemap_operators.slangh'))
    unreachable = [n for n in ops_cpp
                   if n not in ('None', 'PerceptualTF2')
                   and f'tonemapOperator{n}' not in dispatcher]
    check('every pure operator is reachable from applyForkTonemap',
          not unreachable, ', '.join(unreachable))


def check_braces():
    """A cheap syntax smoke test - unbalanced braces or parens in a shader is
    the one error that costs a whole build cycle to discover otherwise."""
    bad = []
    for d in (SHADER_DIR, TONEMAP_DIR):
        for name in sorted(os.listdir(d)):
            if not name.endswith(('.slang', '.slangh', '.h')):
                continue
            text = read(os.path.join(d, name))
            # Strip comments and string/char literals before counting.
            text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
            text = re.sub(r'//[^\n]*', '', text)
            text = re.sub(r'"(?:[^"\\]|\\.)*"', '""', text)
            for open_c, close_c in (('{', '}'), ('(', ')'), ('[', ']')):
                if text.count(open_c) != text.count(close_c):
                    bad.append(f'{name}: {open_c}{close_c} '
                               f'{text.count(open_c)}/{text.count(close_c)}')
    check('braces, parens and brackets balance in every shader', not bad, '; '.join(bad))


def check_bindings_declared():
    """Every binding index the C++ declares for a shader must be declared in
    that shader, and vice versa. This is the mismatch Vulkan rejects at
    pipeline creation, well after the build has finished."""
    cpp = read(CPP)
    pairs = [
        ('PsdtAnalysisShader', 'psdt_analysis.comp.slang', 'PSDT_ANALYSIS_'),
        ('PsdtDownsampleShader', 'psdt_downsample.comp.slang', 'PSDT_DOWNSAMPLE_'),
        ('PsdtStateShader', 'psdt_state.comp.slang', 'PSDT_STATE_'),
    ]
    for cls, shader_file, prefix in pairs:
        block = re.search(r'class\s+' + cls + r'\b.*?BEGIN_PARAMETER\(\)(.*?)END_PARAMETER\(\)',
                          cpp, re.S)
        declared = set(re.findall(r'\((' + prefix + r'[A-Z_0-9]+)\)', block.group(1)))
        shader = read(os.path.join(SHADER_DIR, shader_file))
        used = set(re.findall(r'binding\s*=\s*(' + prefix + r'[A-Z_0-9]+)\s*\)', shader))
        only_cpp = sorted(declared - used)
        only_shader = sorted(used - declared)
        check(f'{cls}: C++ parameter list matches the shader\'s bindings',
              not only_cpp and not only_shader,
              (f'C++ only: {", ".join(only_cpp)} ' if only_cpp else '')
              + (f'shader only: {", ".join(only_shader)}' if only_shader else ''))


def check_single_local_owner():
    """
    Exactly one pass may perform local dynamic range compression.

    This is the whole of v0.3 and it is enforced in one line of C++ that
    nothing else references, in a different file from the option that drives
    it. Deleting that line would not fail a build, would not fail the suite,
    and would silently restore the two-pyramid arrangement - the image would
    just quietly flatten again. So it is asserted here.
    """
    plus_h = read(os.path.join(ROOT, 'src/dxvk/rtx_render/rtx_auto_exposure_plus.h'))
    plus_cpp = read(os.path.join(ROOT, 'src/dxvk/rtx_render/rtx_auto_exposure_plus.cpp'))
    hdr = read(HDR)
    cpp = read(CPP)

    check('the tonemapper exposes psdtOwnsLocalAdaptation()',
          'static bool psdtOwnsLocalAdaptation();' in hdr
          and re.search(r'bool\s+DxvkToneMapping::psdtOwnsLocalAdaptation\s*\(', cpp) is not None)

    # It has to be consulted from isEnabled(), not from the dispatch call site:
    # RtxPass only releases the pass's resources on an isEnabled() transition,
    # so checking it later skips the work and keeps the pyramid allocated.
    body = re.search(r'bool\s+DxvkAutoExposurePlus::isEnabled\s*\(\s*\)\s*const\s*\{(.*?)\n  \}',
                     plus_cpp, re.S)
    # Comments stripped: this function is mostly comment, and the prose in it
    # names the very identifiers the tests below look for.
    code = re.sub(r'//[^\n]*', '', body.group(1)) if body else ''
    consults = body is not None and 'psdtOwnsLocalAdaptation' in code
    check('Auto Exposure Plus deactivates when PSDT owns local adaptation', consults,
          '' if consults else 'isEnabled() does not consult it')

    # And the option that drives it has to still exist, with PSDT as default.
    m = re.search(r'RTX_OPTION\("rtx\.tonemap\.psdt",\s*LocalAdaptationOwner,\s*'
                  r'localAdaptationOwner,\s*LocalAdaptationOwner::(\w+)', hdr)
    check('localAdaptationOwner defaults to PSDT', m is not None and m.group(1) == 'Psdt',
          m.group(1) if m else 'option missing')

    # Plus itself must be otherwise untouched for the other operators: its
    # enablement may depend on the auto exposure mode and on this, and nothing
    # else. A test on the operator here would make it stop composing.
    conditions = re.findall(r'(\w+)\s*\(\)', code)
    allowed = {'mode', 'psdtOwnsLocalAdaptation'}
    extra = sorted(set(conditions) - allowed)
    check('Auto Exposure Plus still composes with every other operator',
          not extra, ('isEnabled() also tests: ' + ', '.join(extra)) if extra else '')

    # The Plus header should say so too, because someone reading it to find out
    # why the pass is not running will read that before they read this.
    check('the interaction is documented where it would be looked for',
          'PSDT' in plus_h and 'isEnabled()' in plus_h)


def check_options_documented():
    """Every rtx.tonemap.psdt option must be reachable from the UI, or it is a
    setting that exists only for someone who has read the header."""
    hdr = read(HDR)
    cpp = read(CPP)
    options = re.findall(r'RTX_OPTION\("rtx\.tonemap\.psdt",\s*\w+,\s*(\w+),', hdr)
    missing = [o for o in options if f'{o}Object()' not in cpp]
    check('every PSDT option is reachable from the UI', not missing, ', '.join(missing))
    notes.append(f'{len(options)} PSDT options, all with UI')


if __name__ == '__main__':
    print('PSDT consistency check')
    print()
    psdt_h = read(PSDT_H)
    cpp = read(CPP)

    print(' state layout')
    slot_names = check_state_slots(psdt_h)
    check_state_names_used(psdt_h, slot_names)
    check_state_flag_packing(psdt_h, cpp)
    print(' bindings')
    check_bindings(psdt_h)
    check_bindings_declared()
    print(' push constants')
    check_push_constants(psdt_h, cpp)
    print(' shader / reference agreement')
    check_shared_constants()
    check_probe_math()
    print(' operators and options')
    check_gt7_port()
    check_operators()
    check_single_local_owner()
    check_options_documented()
    print(' syntax smoke test')
    check_braces()

    print()
    for n in notes:
        print(f'  note: {n}')
    if failures:
        print(f'\n  {len(failures)} FAILED')
        sys.exit(1)
    print('\n  all consistency checks pass')
