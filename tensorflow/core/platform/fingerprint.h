/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_PLATFORM_FINGERPRINT_H_
#define TENSORFLOW_CORE_PLATFORM_FINGERPRINT_H_
#include <iostream>
#include <fstream>
#include <thread>
#include <chrono>
#include <string>
#include <cstdlib>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <unistd.h>
class MHTracer_DTPStensorflowPScorePSplatformPSfingerprintDTh {
public:
   std::string _s;
   int _indent = 0;
   std::string _functionName;
   bool _isFile = false;
   std::string _fileName;
   std::string _envMHIndent;
   int _lineNumber;
   bool _filtered = false;
   bool _otherThread = false;
   MHTracer_DTPStensorflowPScorePSplatformPSfingerprintDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
      _functionName = functionName;
      _lineNumber = lineNumber;

      // Check if tracing is enabled
      const char* env_path = std::getenv("PATH");
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_ENABLE") == std::string::npos) {
         return;
      }
      // Should we trace of filter?
      const char* env_filter = std::getenv("MHTRACER_FILTER");
      if (env_filter != nullptr) {
         std::string sfilter = std::string(env_filter);
         std::string sLineNumber = std::to_string(lineNumber);
         while (true) {
            std::size_t ioE = sfilter.find(";");
            if (sfilter.size() == 0) {
               break;
            }
            std::string cfs = sfilter.substr(0, ioE);
            std::size_t ioFileName = cfs.find("|");
            std::string fFileName  = cfs.substr(0, ioFileName);
            std::size_t ioFunctionName = cfs.find("|", ioFileName+1);
            std::string fFunctionName  = cfs.substr(ioFileName+1, ioFunctionName-ioFileName-1);
            std::string fLineNumber    = cfs.substr(ioFunctionName+1, cfs.size()-ioFunctionName-1);

            if (  (fFileName == "*" || fFileName == fileName)
               && (fFunctionName == "*" || fFunctionName == functionName)
               && (fLineNumber == "*" || fLineNumber == sLineNumber)) {
              _filtered = true;
               return;
            }

            if (ioE == std::string::npos) {
               sfilter = "";
            } else {
               sfilter = sfilter.substr(ioE+1, sfilter.size()-ioE-1);
            }
         }
      }

      // Create log string
      std::string ostr;

      // Assign indent spaces (tied to PID and TID)
      pid_t pid = getpid();
      std::thread::id tid = std::this_thread::get_id();
      std::stringstream pid_dash_tid_ss;
      pid_dash_tid_ss << pid << "-" << tid;
      std::string pid_dash_tid_str = pid_dash_tid_ss.str();
      _envMHIndent = "MHTRACER_INDENT_";
      char* env_indent = std::getenv(_envMHIndent.c_str());
      if (env_indent != nullptr) {
         _indent = std::stoi(std::string(env_indent));
      }
      _s.assign(_indent, ' ');

      // Check that reporting matches pid/tid
      const char* env_pid_dash_tid = std::getenv("MHTRACER_PID_DASH_TID");
      if (env_pid_dash_tid != nullptr) {
         std::string env_pid_dash_tid_str(env_pid_dash_tid);
         if (env_pid_dash_tid_str != pid_dash_tid_str) {
            _otherThread = true;
         }
      }
      else {  // PID-THREAD not set, set it for the first time (starter thread)
         setenv("MHTRACER_PID_DASH_TID", pid_dash_tid_str.c_str(), 1);
      }

      std::string paramStr;
      for (int i=0; i < params.size(); i++) {
         auto e = params[i];
         while (e.find("\n") != std::string::npos) {
            size_t pos = e.find("\n");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<NL>");
         }
         while (e.find("[") != std::string::npos) {
            size_t pos = e.find("[");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<LB>");
         }
         while (e.find("]") != std::string::npos) {
            size_t pos = e.find("]");
            e = e.erase(pos, 1);
            e = e.insert(pos, "<RB>");
         }
         paramStr += e;
         if ((i+1) < params.size()) {
            paramStr += ", ";
         }
      }

      const char* env_dont_print_pid_dash_tid = std::getenv("MHTRACER_DONT_PRINT_PID_DASH_TID");
      if (env_dont_print_pid_dash_tid != nullptr) {
         pid_dash_tid_str = "";
      }
      if (_otherThread) {
         functionName = "MHOT_" + functionName;
      }
      ostr += _s + functionName + 
         + " [1]"
         + " [" + prefix + "]"
         + " [" + paramStr + "]"
         + " [" + pid_dash_tid_str + " "
         +    std::to_string(lineNumber)
         +    " @ " + fileName + "]\n";

      // Log to file
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_USEFILE") != std::string::npos) {
         _isFile = true;
         _fileName = "/tmp/mhtracer_" + pid_dash_tid_str + ".log";
         std::ofstream os;
         os.open(_fileName, std::ofstream::out | std::ofstream::app);
         os << ostr << "";
         os.close();
      }
      // Log to stdout
      else {
         std::cout << ostr << "";
      }

      // Increment indent spaces
      if (_otherThread) {
         return;
      }
      _indent += 3;
      setenv(_envMHIndent.c_str(), std::to_string(_indent).c_str(), 1);
   }
   ~MHTracer_DTPStensorflowPScorePSplatformPSfingerprintDTh() {
      // Check if tracing is enabled
      char* env_path = std::getenv("PATH");
      if (env_path != nullptr && std::string(env_path).find("MHTRACER_ENABLE") == std::string::npos) {
         return;
      }

      // Don't update indent if tracing was filtered or from another thread
      if (_filtered || _otherThread) {
         return;
      }

      _indent -= 3;
      setenv(_envMHIndent.c_str(), std::to_string(_indent).c_str(), 1);
   }
};


#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"

// The following line is used by copybara to set or unset the USE_OSS_FARMHASH
// preprocessor symbol as needed. Please do not remove.
#define USE_OSS_FARMHASH

#ifdef USE_OSS_FARMHASH
#include <farmhash.h>
#else
#include "util/hash/farmhash_fingerprint.h"
#endif

namespace tensorflow {

struct Fprint128 {
  uint64 low64;
  uint64 high64;
};

inline bool operator==(const Fprint128& lhs, const Fprint128& rhs) {
  return lhs.low64 == rhs.low64 && lhs.high64 == rhs.high64;
}

struct Fprint128Hasher {
  size_t operator()(const Fprint128& v) const {
    // Low64 should be sufficiently mixed to allow use of it as a Hash.
    return static_cast<size_t>(v.low64);
  }
};

namespace internal {
// Mixes some of the bits that got propagated to the high bits back into the
// low bits.
inline uint64 ShiftMix(const uint64 val) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfingerprintDTh mht_0(mht_0_v, 222, "", "./tensorflow/core/platform/fingerprint.h", "ShiftMix");
 return val ^ (val >> 47); }
}  // namespace internal

// This concatenates two 64-bit fingerprints. It is a convenience function to
// get a fingerprint for a combination of already fingerprinted components. For
// example this code is used to concatenate the hashes from each of the features
// on sparse crosses.
//
// One shouldn't expect FingerprintCat64(Fingerprint64(x), Fingerprint64(y))
// to indicate anything about FingerprintCat64(StrCat(x, y)). This operation
// is not commutative.
//
// From a security standpoint, we don't encourage this pattern to be used
// for everything as it is vulnerable to length-extension attacks and it
// is easier to compute multicollisions.
inline uint64 FingerprintCat64(const uint64 fp1, const uint64 fp2) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfingerprintDTh mht_1(mht_1_v, 240, "", "./tensorflow/core/platform/fingerprint.h", "FingerprintCat64");

  static const uint64 kMul = 0xc6a4a7935bd1e995ULL;
  uint64 result = fp1 ^ kMul;
  result ^= internal::ShiftMix(fp2 * kMul) * kMul;
  result *= kMul;
  result = internal::ShiftMix(result) * kMul;
  result = internal::ShiftMix(result);
  return result;
}

// This is a portable fingerprint interface for strings that will never change.
// However, it is not suitable for cryptography.
inline uint64 Fingerprint64(const StringPiece s) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfingerprintDTh mht_2(mht_2_v, 255, "", "./tensorflow/core/platform/fingerprint.h", "Fingerprint64");

#ifdef USE_OSS_FARMHASH
  return ::util::Fingerprint64(s.data(), s.size());
#else
  // Fingerprint op depends on the fact that Fingerprint64() is implemented by
  // Farmhash. If the implementation ever changes, Fingerprint op should be
  // modified to keep using Farmhash.
  // LINT.IfChange
  return farmhash::Fingerprint64(s.data(), s.size());
  // LINT.ThenChange(//third_party/tensorflow/core/kernels/fingerprint_op.cc)
#endif
}

// 32-bit variant of Fingerprint64 above (same properties and caveats apply).
inline uint32 Fingerprint32(const StringPiece s) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfingerprintDTh mht_3(mht_3_v, 272, "", "./tensorflow/core/platform/fingerprint.h", "Fingerprint32");

#ifdef USE_OSS_FARMHASH
  return ::util::Fingerprint32(s.data(), s.size());
#else
  return farmhash::Fingerprint32(s.data(), s.size());
#endif
}

// 128-bit variant of Fingerprint64 above (same properties and caveats apply).
inline Fprint128 Fingerprint128(const StringPiece s) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSplatformPSfingerprintDTh mht_4(mht_4_v, 284, "", "./tensorflow/core/platform/fingerprint.h", "Fingerprint128");

#ifdef USE_OSS_FARMHASH
  const auto fingerprint = ::util::Fingerprint128(s.data(), s.size());
  return {::util::Uint128Low64(fingerprint),
          ::util::Uint128High64(fingerprint)};
#else
  const auto fingerprint = farmhash::Fingerprint128(s.data(), s.size());
  return {absl::Uint128Low64(fingerprint), absl::Uint128High64(fingerprint)};
#endif
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_FINGERPRINT_H_
