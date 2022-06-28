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
class MHTracer_DTPStensorflowPScorePSlibPShashPScrc32c_accelerateDTcc {
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
   MHTracer_DTPStensorflowPScorePSlibPShashPScrc32c_accelerateDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPShashPScrc32c_accelerateDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <stddef.h>
#include <stdint.h>

// SSE4.2 accelerated CRC32c.

// See if the SSE4.2 crc32c instruction is available.
#undef USE_SSE_CRC32C
#ifdef __SSE4_2__
#if defined(__x86_64__) && defined(__GNUC__) && \
    (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 8))
#define USE_SSE_CRC32C 1
#elif defined(__x86_64__) && defined(__clang__)
#if __has_builtin(__builtin_cpu_supports)
#define USE_SSE_CRC32C 1
#endif
#endif
#endif /* __SSE4_2__ */

// This version of Apple clang has a bug:
// https://llvm.org/bugs/show_bug.cgi?id=25510
#if defined(__APPLE__) && (__clang_major__ <= 8)
#undef USE_SSE_CRC32C
#endif

#ifdef USE_SSE_CRC32C
#include <nmmintrin.h>
#endif

namespace tensorflow {
namespace crc32c {

#ifndef USE_SSE_CRC32C

bool CanAccelerate() { return false; }
uint32_t AcceleratedExtend(uint32_t crc, const char *buf, size_t size) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("buf: \"" + (buf == nullptr ? std::string("nullptr") : std::string((char*)buf)) + "\"");
   MHTracer_DTPStensorflowPScorePSlibPShashPScrc32c_accelerateDTcc mht_0(mht_0_v, 220, "", "./tensorflow/core/lib/hash/crc32c_accelerate.cc", "AcceleratedExtend");

  // Should not be called.
  return 0;
}

#else

// SSE4.2 optimized crc32c computation.
bool CanAccelerate() { return __builtin_cpu_supports("sse4.2"); }

uint32_t AcceleratedExtend(uint32_t crc, const char *buf, size_t size) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("buf: \"" + (buf == nullptr ? std::string("nullptr") : std::string((char*)buf)) + "\"");
   MHTracer_DTPStensorflowPScorePSlibPShashPScrc32c_accelerateDTcc mht_1(mht_1_v, 234, "", "./tensorflow/core/lib/hash/crc32c_accelerate.cc", "AcceleratedExtend");

  const uint8_t *p = reinterpret_cast<const uint8_t *>(buf);
  const uint8_t *e = p + size;
  uint32_t l = crc ^ 0xffffffffu;

  // Advance p until aligned to 8-bytes..
  // Point x at first 7-byte aligned byte in string.  This might be
  // just past the end of the string.
  const uintptr_t pval = reinterpret_cast<uintptr_t>(p);
  const uint8_t *x = reinterpret_cast<const uint8_t *>(((pval + 7) >> 3) << 3);
  if (x <= e) {
    // Process bytes until finished or p is 8-byte aligned
    while (p != x) {
      l = _mm_crc32_u8(l, *p);
      p++;
    }
  }

  // Process bytes 16 at a time
  uint64_t l64 = l;
  while ((e - p) >= 16) {
    l64 = _mm_crc32_u64(l64, *reinterpret_cast<const uint64_t *>(p));
    l64 = _mm_crc32_u64(l64, *reinterpret_cast<const uint64_t *>(p + 8));
    p += 16;
  }

  // Process remaining bytes one at a time.
  l = l64;
  while (p < e) {
    l = _mm_crc32_u8(l, *p);
    p++;
  }

  return l ^ 0xffffffffu;
}

#endif

}  // namespace crc32c
}  // namespace tensorflow
