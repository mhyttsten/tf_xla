/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_LIB_CORE_BITS_H_
#define TENSORFLOW_CORE_LIB_CORE_BITS_H_
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
class MHTracer_DTPStensorflowPScorePSlibPScorePSbitsDTh {
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
   MHTracer_DTPStensorflowPScorePSlibPScorePSbitsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSlibPScorePSbitsDTh() {
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


#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Return floor(log2(n)) for positive integer n.  Returns -1 iff n == 0.
int Log2Floor(uint32 n);
int Log2Floor64(uint64 n);

// Return ceiling(log2(n)) for positive integer n.  Returns -1 iff n == 0.
int Log2Ceiling(uint32 n);
int Log2Ceiling64(uint64 n);

// ------------------------------------------------------------------------
// Implementation details follow
// ------------------------------------------------------------------------

#if defined(__GNUC__)

// Return floor(log2(n)) for positive integer n.  Returns -1 iff n == 0.
inline int Log2Floor(uint32 n) { return n == 0 ? -1 : 31 ^ __builtin_clz(n); }

// Return floor(log2(n)) for positive integer n.  Returns -1 iff n == 0.
inline int Log2Floor64(uint64 n) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSbitsDTh mht_0(mht_0_v, 211, "", "./tensorflow/core/lib/core/bits.h", "Log2Floor64");

  return n == 0 ? -1 : 63 ^ __builtin_clzll(n);
}

#else

// Return floor(log2(n)) for positive integer n.  Returns -1 iff n == 0.
inline int Log2Floor(uint32 n) {
  if (n == 0) return -1;
  int log = 0;
  uint32 value = n;
  for (int i = 4; i >= 0; --i) {
    int shift = (1 << i);
    uint32 x = value >> shift;
    if (x != 0) {
      value = x;
      log += shift;
    }
  }
  assert(value == 1);
  return log;
}

// Return floor(log2(n)) for positive integer n.  Returns -1 iff n == 0.
// Log2Floor64() is defined in terms of Log2Floor32()
inline int Log2Floor64(uint64 n) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSbitsDTh mht_1(mht_1_v, 239, "", "./tensorflow/core/lib/core/bits.h", "Log2Floor64");

  const uint32 topbits = static_cast<uint32>(n >> 32);
  if (topbits == 0) {
    // Top bits are zero, so scan in bottom bits
    return Log2Floor(static_cast<uint32>(n));
  } else {
    return 32 + Log2Floor(topbits);
  }
}

#endif

inline int Log2Ceiling(uint32 n) {
  int floor = Log2Floor(n);
  if (n == (n & ~(n - 1)))  // zero or a power of two
    return floor;
  else
    return floor + 1;
}

inline int Log2Ceiling64(uint64 n) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSbitsDTh mht_2(mht_2_v, 262, "", "./tensorflow/core/lib/core/bits.h", "Log2Ceiling64");

  int floor = Log2Floor64(n);
  if (n == (n & ~(n - 1)))  // zero or a power of two
    return floor;
  else
    return floor + 1;
}

inline uint32 NextPowerOfTwo(uint32 value) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSbitsDTh mht_3(mht_3_v, 273, "", "./tensorflow/core/lib/core/bits.h", "NextPowerOfTwo");

  int exponent = Log2Ceiling(value);
  DCHECK_LT(exponent, std::numeric_limits<uint32>::digits);
  return 1 << exponent;
}

inline uint64 NextPowerOfTwo64(uint64 value) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSlibPScorePSbitsDTh mht_4(mht_4_v, 282, "", "./tensorflow/core/lib/core/bits.h", "NextPowerOfTwo64");

  int exponent = Log2Ceiling(value);
  DCHECK_LT(exponent, std::numeric_limits<uint64>::digits);
  return 1LL << exponent;
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_CORE_BITS_H_
