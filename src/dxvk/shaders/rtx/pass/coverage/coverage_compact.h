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

#ifndef COVERAGE_COMPACT_H
#define COVERAGE_COMPACT_H

#include "rtx/utility/shader_types.h"

// NV-DXVK [Coverage compact]: GPU compaction of the 71-region x 262144-slot
// surface-coverage buffer. The CPU used to scan all ~74MB of that buffer
// every frame from HOST_VISIBLE|HOST_COHERENT memory — uncached
// (write-combined) on every driver, so each 4-byte read is a full memory
// transaction and the scan alone cost ~1 second/frame while the GPU sat
// idle (measured: [Perf.Frame] finalBlit ~0.9-1.4s, GPUIDLEms ~= frame).
// This pass makes the GPU (which reads the buffer at full VRAM bandwidth,
// sub-millisecond) emit only the NONZERO entries as (flatIndex, value)
// pairs into a small host-CACHED readback buffer. The CPU then rebuilds a
// sparse shadow copy from just those pairs — thousands of entries instead
// of 18.6 million slots.

#define COVERAGE_COMPACT_INPUT  0
#define COVERAGE_COMPACT_OUTPUT 1

// Output-buffer layout: header, then entryCapacity (index, value) pairs.
//   uint[0] = write cursor (atomic). May exceed entryCapacity — the excess
//             is the number of DROPPED entries, which the CPU logs.
//   uint[1] = pad / reserved.
#define COVERAGE_COMPACT_HEADER_UINTS 2

// Capacity in (index, value) pairs. Real gameplay frames populate on the
// order of 10 dense regions x ~13k live surfaces ~= 130k nonzero slots;
// 1M pairs (8 MB host-cached) leaves generous headroom before dropping.
#define COVERAGE_COMPACT_MAX_ENTRIES (1024u * 1024u)

struct CoverageCompactArgs {
  uint totalElements;  // COVERAGE_TOTAL_REGIONS * COVERAGE_SURFACE_SLOTS
  uint threadCount;    // total dispatched threads (grid-stride loop step)
  uint entryCapacity;  // COVERAGE_COMPACT_MAX_ENTRIES
  uint pad0;
};

#endif  // COVERAGE_COMPACT_H
