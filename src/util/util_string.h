/*
* Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
#pragma once

#include <charconv>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

// NV-DXVK [Perf.FmtSite]: call-site attribution for the float branch of
// appendOne below. Self-contained and Win32-guarded internally.
#include "util_fmt_diag.h"
#include <string_view>
#include <sstream>
#include <type_traits>
#include <vector>

#include "./com/com_include.h"

// Converts a string defined by a macro e.g. #define FOO "bar" to a wide string, e.g. L"bar"
#define WIDEN_MACRO_LITERAL2(x) L ## x
#define WIDEN_MACRO_LITERAL(x) WIDEN_MACRO_LITERAL2(x)

namespace dxvk::str {

  std::string fromws(const WCHAR *ws);

  void tows(const char* mbs, WCHAR* wcs, size_t wcsLen);

  template <size_t N>
  void tows(const char* mbs, WCHAR (&wcs)[N]) {
    return tows(mbs, wcs, N);
  }

  std::wstring tows(const char* mbs);

  // NV-DXVK [perf]: str::format used to construct a std::stringstream per call.
  // On MSVC that is a basic_ios init + a locale copy (atomic refcount traffic on
  // every facet) + a stringbuf allocation, and each operator<< then goes through
  // num_put facet lookup — measured in the microseconds for a typical 15-20 arg
  // diagnostic line. SubmitDraw alone has 233 str::format sites, and the cost is
  // paid at the CALL SITE even when Logger::emitMsg later drops the message, so
  // the emitMsg denylist never recovered it (that is why gating the log output
  // did not move fps).
  //
  // This version appends straight into one std::string. Types whose ostream
  // insertion semantics are non-trivial to replicate exactly (parameterised
  // manipulators like std::setw/std::setfill, pointers, long double, and any
  // user-defined operator<<(std::ostream&, T) such as Vector3 / Matrix4) fall
  // back to a real std::ostringstream. The fallback is sticky: once armed, every
  // remaining argument goes through the stream, so stream-state manipulators
  // keep applying to the arguments that follow them exactly as before.
  namespace detail {

    using IosManipFn = std::ios_base& (std::ios_base&);

    struct FormatSink {
      std::string                         out;
      std::unique_ptr<std::ostringstream> slow;
      int                                 base = 10;  // std::dec / std::hex / std::oct

      // Arm the exact-ostream fallback, carrying over both the text produced so
      // far and the current base flag so output is identical either way.
      std::ostringstream& toSlow() {
        if (slow == nullptr) {
          slow = std::make_unique<std::ostringstream>();

          if (base == 16)
            slow->setf(std::ios_base::hex, std::ios_base::basefield);
          else if (base == 8)
            slow->setf(std::ios_base::oct, std::ios_base::basefield);

          slow->write(out.data(), static_cast<std::streamsize>(out.size()));
          out.clear();
        }

        return *slow;
      }
    };

    template<typename T>
    void appendOne(FormatSink& sink, const T& arg) {
      // Once the fallback is armed everything must keep going through it, or the
      // pieces would come out in the wrong order.
      if (sink.slow != nullptr) {
        *sink.slow << arg;
        return;
      }

      if constexpr (std::is_same_v<T, IosManipFn>) {
        // std::hex / std::dec / std::oct. Anything else with this signature is
        // not something we can model, so hand it to the stream.
        if (&arg == static_cast<IosManipFn*>(std::hex))
          sink.base = 16;
        else if (&arg == static_cast<IosManipFn*>(std::dec))
          sink.base = 10;
        else if (&arg == static_cast<IosManipFn*>(std::oct))
          sink.base = 8;
        else
          sink.toSlow() << arg;
      } else if constexpr (std::is_array_v<T>
                        && std::is_same_v<std::remove_cv_t<std::remove_extent_t<T>>, char>) {
        // String literal — ostream inserts it as a C string (stops at the NUL),
        // not as a fixed-size array.
        sink.out.append(static_cast<const char*>(arg));
      } else if constexpr (std::is_same_v<std::decay_t<T>, char*>
                        || std::is_same_v<std::decay_t<T>, const char*>) {
        const char* s = arg;
        if (s != nullptr)
          sink.out.append(s);
      } else if constexpr (std::is_same_v<T, std::string>) {
        sink.out.append(arg);
      } else if constexpr (std::is_same_v<T, std::string_view>) {
        sink.out.append(arg.data(), arg.size());
      } else if constexpr (std::is_same_v<T, char>
                        || std::is_same_v<T, signed char>
                        || std::is_same_v<T, unsigned char>) {
        // ostream inserts all three of these as characters, not as numbers.
        sink.out.push_back(static_cast<char>(arg));
      } else if constexpr (std::is_same_v<T, bool>) {
        // Default (noboolalpha) inserts 1 / 0 in every base.
        sink.out.push_back(arg ? '1' : '0');
      } else if constexpr (std::is_same_v<T, wchar_t>
                        || std::is_same_v<T, char16_t>
                        || std::is_same_v<T, char32_t>) {
        // Inserting these into a narrow stream is deleted; keep that a compile
        // error instead of silently printing a code point.
        sink.toSlow() << arg;
      } else if constexpr (std::is_integral_v<T>) {
        char buf[24];
        char* end;

        if (sink.base == 10) {
          end = std::to_chars(buf, buf + sizeof(buf), arg).ptr;
        } else {
          // num_put converts signed values to their unsigned equivalent for the
          // non-decimal bases, so `std::hex << -1` prints ffffffff, not -1.
          end = std::to_chars(buf, buf + sizeof(buf),
                              static_cast<std::make_unsigned_t<T>>(arg), sink.base).ptr;
        }

        sink.out.append(buf, static_cast<size_t>(end - buf));
      } else if constexpr (std::is_enum_v<T> && std::is_convertible_v<T, int>) {
        // Unscoped enums promote to their integer type on insertion. Scoped
        // enums are not convertible and are left to fail exactly as before.
        appendOne(sink, static_cast<std::underlying_type_t<T>>(arg));
      } else if constexpr ((std::is_same_v<T, float> || std::is_same_v<T, double>)) {
        // ostream promotes to double and defaults to defaultfloat with
        // precision 6, i.e. exactly "%.6g".
        //
        // std::to_chars(general, 6) is specified to produce the same characters
        // as printf("%.6g") in the C locale, and it does: verified byte-for-byte
        // against both printf and ostream over ~800k random float/double bit
        // patterns, at ~6x the speed of printf and ~9x the speed of ostream.
        // This matters because the diagnostic lines here carry a dozen-plus
        // floats each, and float insertion was the single largest cost in the
        // old stringstream path.
        //
        // Non-finite values are the one thing to_chars spells differently
        // ("nan" vs the UCRT's "-nan(ind)"), so those keep going through the CRT
        // to leave existing log text unchanged. Done inline rather than via the
        // sticky fallback so a single NaN doesn't drag the rest of the line onto
        // the slow path.
        const double d = static_cast<double>(arg);
        char buf[64];

        if (d == d && d - d == 0.0) {
          const auto res = std::to_chars(buf, buf + sizeof(buf), d,
                                         std::chars_format::general, 6);
          sink.out.append(buf, static_cast<size_t>(res.ptr - buf));
        } else {
          const int n = std::snprintf(buf, sizeof(buf), "%.6g", d);
          if (n > 0)
            sink.out.append(buf, static_cast<size_t>(n < int(sizeof(buf)) ? n : sizeof(buf) - 1));
        }
      } else {
        // Pointers, long double, std::setw/setfill/setprecision, and every
        // user-defined operator<<(std::ostream&, T).
        sink.toSlow() << arg;
      }
    }

    inline void formatInto(FormatSink&) { }

    template<typename... Tx>
    void formatInto(FormatSink& sink, const WCHAR* arg, const Tx&... args);

    template<typename T, typename... Tx>
    void formatInto(FormatSink& sink, const T& arg, const Tx&... args);

    template<typename... Tx>
    void formatInto(FormatSink& sink, const WCHAR* arg, const Tx&... args) {
      if (sink.slow != nullptr)
        *sink.slow << fromws(arg);
      else
        sink.out.append(fromws(arg));

      formatInto(sink, args...);
    }

    template<typename T, typename... Tx>
    void formatInto(FormatSink& sink, const T& arg, const Tx&... args) {
      appendOne(sink, arg);
      formatInto(sink, args...);
    }

  }

  template<typename... Args>
  std::string format(const Args&... args) {
    detail::FormatSink sink;

    // NV-DXVK [Perf.FmtSite]: name the CALL SITE, and note it HERE rather than
    // in appendOne. The first version of this probe sat in appendOne's float
    // branch, and _ReturnAddress() there resolved to formatInto<float,char[2],
    // ...> - appendOne had been inlined into the recursive formatInto, so the
    // probe named its own caller inside this header instead of the diagnostic
    // that issued the format. Six sites, ~4.7M calls/window, and not one of
    // them identifiable.
    //
    // format() is the outermost frame, so its return address is the real
    // caller. The float count is folded in at compile time, which also means
    // the whole probe compiles away for the (many) format() calls that pass no
    // floats at all - strictly cheaper than the per-float version it replaces.
#ifdef _WIN32
    constexpr uint32_t kFloatArgs = (0u + ... + uint32_t(
      std::is_same_v<std::decay_t<Args>, float> ||
      std::is_same_v<std::decay_t<Args>, double>));
    if constexpr (kFloatArgs != 0u)
      fmt_diag::noteFloat(_ReturnAddress(), kFloatArgs);
#endif

    // Long diagnostic lines are the ones that reallocate repeatedly; short
    // fragments (the `str::format(" t", slot, "=", size)` pieces built inside
    // loops) stay inside the small-string buffer, so don't reserve them away.
    if constexpr (sizeof...(Args) > 6)
      sink.out.reserve(256);

    detail::formatInto(sink, args...);

    if (sink.slow != nullptr)
      return sink.slow->str();

    return std::move(sink.out);
  }

  std::vector<std::string> split(std::string s, const char delimiter = ',');

  std::string sanitizeUtf8(const std::string& input);

  std::string formatBytes(size_t bytes);

  // Note: Constructs a string view including the null terminator unlike the standard library's std::basic_string_view
  // which does not include the null terminator. This makes it easier to work with APIs that expect null terminated strings
  // using string views.
  template<typename T, std::size_t L>
  constexpr auto string_viewz(const T(&t)[L]) {
    return std::basic_string_view<T>(t, L);
  }
}
