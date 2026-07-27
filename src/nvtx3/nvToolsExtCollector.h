/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Licensed under the Apache License v2.0 with LLVM Exceptions.
 * See https://nvidia.github.io/NVTX/LICENSE.txt for license information.
 */

#if defined(NVTX_AS_SYSTEM_HEADER)
#if defined(__clang__)
#pragma clang system_header
#elif defined(__GNUC__) || defined(__NVCOMPILER)
#pragma GCC system_header
#elif defined(_MSC_VER)
#pragma system_header
#endif
#endif

#include "nvtx3/nvToolsExt.h"

/**
 * \brief The compatibility ID is used for versioning of this extension.
 */
#ifndef NVTX_EXT_COLLECTOR_COMPATID
#define NVTX_EXT_COLLECTOR_COMPATID 0x0101
#endif

/**
 * \brief The module ID identifies the plugins extension. It has to be unique
 * among the extension modules.
 *
 * Since this extension is private, instead of continuing the sequence number
 * from the beginning, start the numbering backwards from USHRT_MAX - 1 to avoid
 * clashes with future public extensions.
 */
#ifndef NVTX_EXT_COLLECTOR_MODULEID
#define NVTX_EXT_COLLECTOR_MODULEID 0xFFFE
#endif


#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */


#ifndef NVTX_COLLECTOR_API_FUNCTIONS_V1
#define NVTX_COLLECTOR_API_FUNCTIONS_V1

typedef enum nvtxCollectorState_t {
    NVTX_COLLECTOR_STATE_INVALID      = 0,
    NVTX_COLLECTOR_STATE_IDLE         = 1,
    NVTX_COLLECTOR_STATE_COLLECTING   = 2,
    NVTX_COLLECTOR_STATE_FINALIZING   = 3,
    NVTX_COLLECTOR_STATE_SHUTDOWN     = 4,
} nvtxCollectorState_t;


typedef uint64_t nvtxCollectorFinalizationHandle;

/**
 * \brief Get the current attached collector state.
 *
 * @return Collector current state as nvtxCollectorState_t enum value.
 * The returned value is NVTX_COLLECTOR_STATE_INVALID when no tool is attached.
 */
NVTX_DECLSPEC nvtxCollectorState_t NVTX_API nvtxCollectorGetState(void);

/**
 * \brief Request a finalization stage from the attached collector.
 *
 * If the acquired token is non-zero then the tool will enter NVTX_COLLECTOR_STATE_FINALIZING
 * state after exiting normally from NVTX_COLLECTOR_STATE_COLLECTING until the
 * acquired token is released via a call to nvtxCollectorReleaseFinalizationToken
 * or the process that acquired the token terminates.
 * A single process may acquire multiple tokens and the collector will be in
 * NVTX_COLLECTOR_STATE_FINALIZING state until all such tokens have been released.
 *
 * The function will fail and return a zero-value token when the collector state
 * is not NVTX_COLLECTOR_STATE_IDLE or NVTX_COLLECTOR_STATE_COLLECTING at the
 * moment of invocation.
 *
 * @return non-zero handle if the operation was successful, a zero value otherwise.
 */
NVTX_DECLSPEC nvtxCollectorFinalizationHandle NVTX_API nvtxCollectorAcquireFinalizationToken(void);

/**
 * \brief Release a token obtained by call to nvtxCollectorAcquireFinalizationToken.
 *
 * Once all acquired tokens have been released the collector may leave the
 * NVTX_COLLECTOR_STATE_FINALIZING state.
 *
 * @param token The nvtxCollectorFinalizationHandle token value.
 *
 * @return NVTX_SUCCESS if the given token was released, NVTX_FAIL otherwise.
 */
NVTX_DECLSPEC int NVTX_API nvtxCollectorReleaseFinalizationToken(nvtxCollectorFinalizationHandle token);

/**
 * \brief Block until the desired collector state is entered, the tool is shut down, or the timeout has expired.
 *
 * If collector state is NVTX_COLLECTOR_STATE_INVALID or
 * NVTX_COLLECTOR_STATE_SHUTDOWN the function will return immediately.
 *
 * @param state The requested collector state.
 * @param timeoutNsec Timeout in nanoseconds. Set to zero to wait indefinitely.
 *
 * @return NVTX_SUCCESS if collector state equals the requested one, NVTX_FAIL otherwise.
 */
NVTX_DECLSPEC int NVTX_API nvtxCollectorWaitForState(nvtxCollectorState_t state,
    uint64_t timeoutNsec);

#endif /* NVTX_COLLECTOR_API_FUNCTIONS_V1 */

#ifndef NVTX_COLLECTOR_CALLBACK_ID_V1
#define NVTX_COLLECTOR_CALLBACK_ID_V1

#define NVTX3EXT_CBID_nvtxCollectorGetState                  0
#define NVTX3EXT_CBID_nvtxCollectorAcquireFinalizationToken  1
#define NVTX3EXT_CBID_nvtxCollectorReleaseFinalizationToken  2
#define NVTX3EXT_CBID_nvtxCollectorWaitForState              3

#endif /* NVTX_COLLECTOR_CALLBACK_ID_V1 */

/* Macros to create versioned symbols. */
#ifndef NVTX_EXT_COLLECTOR_VERSIONED_IDENTIFIERS_V1
#define NVTX_EXT_COLLECTOR_VERSIONED_IDENTIFIERS_V1
#define NVTX_EXT_COLLECTOR_VERSIONED_IDENTIFIER_L3(NAME, VERSION, COMPATID) \
    NAME##_v##VERSION##_col##COMPATID
#define NVTX_EXT_COLLECTOR_VERSIONED_IDENTIFIER_L2(NAME, VERSION, COMPATID) \
    NVTX_EXT_COLLECTOR_VERSIONED_IDENTIFIER_L3(NAME, VERSION, COMPATID)
#define NVTX_EXT_COLLECTOR_VERSIONED_ID(NAME) \
    NVTX_EXT_COLLECTOR_VERSIONED_IDENTIFIER_L2(NAME, NVTX_VERSION, NVTX_EXT_COLLECTOR_COMPATID)
#endif /* NVTX_EXT_COLLECTOR_VERSIONED_IDENTIFIERS_V1 */

#ifdef __GNUC__
#pragma GCC visibility push(internal)
#endif

/* Extension types are required for the implementation and the NVTX handler. */
#define NVTX_EXT_TYPES_GUARD
#include "nvtxDetail/nvtxExtTypes.h"
#undef NVTX_EXT_TYPES_GUARD

#ifndef NVTX_NO_IMPL
#define NVTX_EXT_IMPL_COLLECTOR_GUARD
#include "nvtxDetail/nvtxExtImplCollector_v1.h"
#undef NVTX_EXT_IMPL_COLLECTOR_GUARD
#endif /* NVTX_NO_IMPL */

#ifdef __GNUC__
#pragma GCC visibility pop
#endif

#ifdef __cplusplus
}
#endif /* __cplusplus */
