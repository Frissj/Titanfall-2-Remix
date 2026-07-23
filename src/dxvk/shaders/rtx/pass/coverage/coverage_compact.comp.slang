/*
* Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a
* copy of this software and associated documentation files (the "Software"),
* to deal in the Software without restriction, including without limitation
* the rights to use, copy, modify, merge, publish, distribute, sublicense,
* and/or sell copies of the Software, and to permit persons to whom the
* Software is furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/
#include "rtx/pass/coverage/coverage_compact.h"

// NV-DXVK [Coverage compact]: see coverage_compact.h for why this exists.
// One grid-stride pass over the full coverage buffer; nonzero slots are
// appended as (flatIndex, value) pairs via an atomic cursor. The cursor is
// zeroed by the CPU (clearBuffer) before this dispatch each frame.

layout(binding = COVERAGE_COMPACT_INPUT)
RWStructuredBuffer<uint> Coverage;

layout(binding = COVERAGE_COMPACT_OUTPUT)
RWStructuredBuffer<uint> Compact;

layout(push_constant)
ConstantBuffer<CoverageCompactArgs> cb;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
  // Grid-stride loop: consecutive threads read consecutive slots each
  // iteration, so reads stay fully coalesced across the whole buffer.
  for (uint i = tid.x; i < cb.totalElements; i += cb.threadCount)
  {
    const uint v = Coverage[i];
    if (v == 0u)
    {
      continue;
    }

    uint dst;
    InterlockedAdd(Compact[0], 1u, dst);
    // Cursor keeps counting past capacity on purpose: the overflow amount
    // tells the CPU how many entries were dropped.
    if (dst < cb.entryCapacity)
    {
      Compact[COVERAGE_COMPACT_HEADER_UINTS + dst * 2u]      = i;
      Compact[COVERAGE_COMPACT_HEADER_UINTS + dst * 2u + 1u] = v;
    }
  }
}
