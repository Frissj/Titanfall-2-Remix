#!/usr/bin/env python3
# Post-build patcher that injects SPIR-V capability declarations slangc fails
# to emit. Specifically, slang-compiled SPIR-V uses OpImageRead/OpImageWrite on
# storage images declared with Unknown format but does not declare the matching
# StorageImageReadWithoutFormat / StorageImageWriteWithoutFormat capabilities.
# spirv-val rejects this at vkCreateShaderModule time when validation layers
# are enabled. The matching device features are enabled in dxvk_adapter.cpp.
#
# Usage: add_spirv_capabilities.py <input.spv> [<input2.spv> ...]
# Patches files in place. Idempotent.
import struct
import sys

SPIRV_MAGIC = 0x07230203
OP_CAPABILITY = 17

# SPIR-V capability IDs (from the SPIR-V spec).
CAP_STORAGE_IMAGE_READ_WITHOUT_FORMAT  = 55
CAP_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT = 56

CAPS_TO_ENSURE = (
    CAP_STORAGE_IMAGE_READ_WITHOUT_FORMAT,
    CAP_STORAGE_IMAGE_WRITE_WITHOUT_FORMAT,
)


def patch_spv(path):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) < 20:
        return False
    magic = struct.unpack("<I", data[0:4])[0]
    if magic != SPIRV_MAGIC:
        # Not little-endian SPIR-V; skip silently. (Big-endian SPIR-V is rare
        # and the build tools we use produce little-endian.)
        return False

    header = data[:20]
    body   = data[20:]

    # Walk instructions starting at offset 0 of body. Each instruction's first
    # word: high 16 bits = word count, low 16 bits = opcode.
    existing_caps = set()
    insert_offset = 0
    pos = 0
    while pos + 4 <= len(body):
        word = struct.unpack_from("<I", body, pos)[0]
        wc = (word >> 16) & 0xFFFF
        op = word & 0xFFFF
        if wc == 0:
            # Malformed; bail.
            return False
        instr_bytes = wc * 4
        if pos + instr_bytes > len(body):
            return False
        if op == OP_CAPABILITY:
            if wc >= 2:
                cap = struct.unpack_from("<I", body, pos + 4)[0]
                existing_caps.add(cap)
            insert_offset = pos + instr_bytes
        else:
            # Capabilities must come first per SPIR-V layout. Stop on the
            # first non-OpCapability instruction.
            break
        pos += instr_bytes

    to_add = [c for c in CAPS_TO_ENSURE if c not in existing_caps]
    if not to_add:
        return False

    # Build new OpCapability instructions: word count 2, opcode 17, then cap id.
    new_chunk = b"".join(
        struct.pack("<II", (2 << 16) | OP_CAPABILITY, c) for c in to_add
    )

    new_body = body[:insert_offset] + new_chunk + body[insert_offset:]
    new_data = header + new_body

    with open(path, "wb") as f:
        f.write(new_data)
    return True


def main():
    paths = sys.argv[1:]
    if not paths:
        print("usage: add_spirv_capabilities.py <input.spv> [...]", file=sys.stderr)
        return 1
    patched = 0
    for p in paths:
        try:
            if patch_spv(p):
                patched += 1
        except Exception as e:
            print(f"add_spirv_capabilities: {p}: {e}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
