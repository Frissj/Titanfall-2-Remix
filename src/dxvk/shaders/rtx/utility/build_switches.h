/*
* Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
* THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*/
#pragma once

// NV-DXVK [Perf.CodeVolume]: compile-time switches that remove whole BLOCKS of
// shader code, as opposed to the cb.perf* knobs which remove executed
// instructions while leaving the code in the binary.
//
// BOTH DEFAULT TO 1. The hypothesis they were built to test is REFUTED. Read
// this whole comment before setting either to 0 - the experiment has been run.
//
// ---------------------------------------------------------------------------
// THE HYPOTHESIS (2026-07-26)
//
// From remix-dxvk.log, [Perf.Shader] for the primary gbuffer kernel:
//
//   cs=gbuffer_rayquery_nrc_wboit_opaque_translucent
//     Register Count=255  Binary Size=826880  Local Memory Size=144
//
// 255 is the hardware maximum VGPR count on Ada. Across every pass in that log,
// time appeared to track (binary size, register count):
//
//   composite                  62 KB /  72 regs -> 0.44 ms
//   volume_integrate_rayquery  154 KB / 128 regs -> 2.0  ms
//   integrate_direct_rayquery  835 KB / 128 regs -> 15.4 ms
//   gbuffer_rayquery_...       827 KB / 255 regs -> 24   ms
//
// The reasoning was that 255 registers caps occupancy at 8 warps/SM (~16%),
// too little to hide memory latency, and that 827 KB thrashes the instruction
// cache - and that this explained why every ablation in
// HANDOFF_TF2_PRIMARY_RAY_COST_V4 measured null (each removed instructions that
// EXECUTE, none removed code volume or register pressure).
//
// ---------------------------------------------------------------------------
// THE RESULT: REFUTED
//
// Setting both switches to 0 did exactly what it was supposed to do to the
// binary, and nothing at all to the time:
//
//   Register Count  255    -> 168 (at 128 threads) / 250 (at 256 threads)
//   Binary Size     826880 -> 350848  (-58%)
//
//   switches   rtx.perfGbufferBlockThreads   gb_primaryRays
//   1          0                             24.085 ms
//   1          256                           23.788 ms
//   0          256                           23.89  ms   <- -58% binary, +0.1 ms
//   0          0                             28.8   ms   <- 20% WORSE
//
// A 58% smaller binary and 87 fewer registers are worth 0.1 ms, i.e. noise.
// The 256-thread variant of the small shader compiles with Shared Memory = 0,
// so it runs with the maximum possible L1 (the full 128 KB pool) and is still
// not faster - L1 capacity above ~113 KB does nothing either.
//
// WHY THE 128-THREAD CASE IS 20% WORSE. Block is 16x8 = 128 threads. AD104 has
// 65536 registers/SM and ONE 128 KB pool split between L1 data cache and shared
// memory:
//   255 regs: 128x255 = 32640/block -> 2 blocks =  8 warps, 2x6656  =  13 KB
//             shared -> ~115 KB left as L1
//   168 regs: 128x168 = 21504/block -> 3 blocks = 12 warps, 3x33280 = 100 KB
//             shared -> ~28 KB left as L1
// Freeing registers let a third block become resident, and the driver pays for
// the extra blocks with shared memory (it picks register count AND shared
// allocation together from a target-occupancy heuristic - at 256 threads only
// one block fits, so it takes 250 registers and allocates no shared at all).
// The instruction cache got better; the data cache lost 75% while 50% more
// warps competed for it.
//
// ---------------------------------------------------------------------------
// WHAT IS NOW REFUTED AS A LEVER ON gb_primaryRays
//
// The pass sits at 23.8-24.1 ms in every configuration anyone has tried.
// Independently refuted: occupancy (the perfGbufferBlockThreads ladder, the
// compile-time material rung at 168 regs, and this change), code volume,
// instruction cache, L1 capacity, and the ~20 work ablations in the V4 handoff.
// A pass insensitive to all of those is not limited by anything in the shader.
// Do not spend another build on shader-side micro-optimisation until something
// explains that insensitivity.
//
// ---------------------------------------------------------------------------
// IF YOU DO SET THESE TO 0
//
// Pair it with rtx.perfGbufferBlockThreads=256 or you will eat the 20% cliff.
// Note that option is a DIAGNOSTIC guarded to the NRC + WBOIT + no-portals
// RayQuery permutation, so any permutation change silently drops back to 128
// threads. That fragility is why 1 is the default here rather than shipping the
// smaller shader plus a knob to make it safe.
//
// The falsifiable output of flipping these is the [Perf.Shader] line, NOT the
// perf sweep (which is compiled out at 0 and reports the baseline for every
// rung). Read Register Count and Shared Memory Size for the gbuffer variant.

// NV-DXVK [Perf.CodeVolume]: the per-pixel debug view switch.
//
// geometry_resolver.slangh calls geometryResolverVertexOutputDebugView()
// unconditionally on every primary hit. That function is a 66-case switch over
// cb.debugView spanning ~815 lines, and several of its cases call
// surfaceInteractionCreate<SurfaceGenerateTangents> a SECOND time, run POM
// raymarch loops, and rebuild material interactions. In production
// cb.debugView == DEBUG_VIEW_DISABLED so none of it runs - but all of it is in
// the binary, and the register allocator budgets for the worst path through a
// function, not the taken one. It is the single largest removable block in the
// kernel, and removing it is worth nothing (see above).
//
// Set to 0 to compile the gbuffer debug views out. THIS DISABLES THE DEBUG VIEW
// DROPDOWN for everything sourced from the gbuffer pass (albedo, normals,
// geometry hash, texcoord gradients, PSR/POM views, ...). Passes that own their
// own debug output (debug_view.comp.slang, demodulate, composite) are
// unaffected.
#ifndef REMIX_ENABLE_GBUFFER_DEBUG_VIEWS
#define REMIX_ENABLE_GBUFFER_DEBUG_VIEWS 1
#endif

// NV-DXVK [Perf.CodeVolume]: the cb.perf* ablation ladders.
//
// Twelve runtime knobs (perfGbStopAfter, perfUnorderedStopAfter,
// perfSurfaceInteractionStopAfter, perfMaterialStopAfter, perfSkipGeometryFetch,
// perfSkipMaterialTextures, perfSkipPom, perfSkipThinFilm,
// perfCheapTextureGradients, perfCoherentUnorderedFetch, perfUnorderedStepCensus,
// perfCoverageWrites). They are deliberately RUNTIME branches so that both sides
// stay alive and register allocation is identical between rungs - see the note
// at surface_interaction.slangh's perfSurfaceInteractionStopAfter ladder.
//
// With this at 0 every ladder condition folds to a compile-time constant, so
// slang/DXC dead-code-eliminates the probe branch AND its body. The RtxOptions
// and the CPU-side sweep driver are untouched; the knobs simply stop doing
// anything, which means rtx.perfAutoSweep reports the baseline for every rung.
#ifndef REMIX_ENABLE_PERF_LADDERS
#define REMIX_ENABLE_PERF_LADDERS 1
#endif
