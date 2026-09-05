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
    print(' bindings')
    check_bindings(psdt_h)
    check_bindings_declared()
    print(' push constants')
    check_push_constants(psdt_h, cpp)
    print(' shader / reference agreement')
    check_shared_constants()
    print(' operators and options')
    check_operators()
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
