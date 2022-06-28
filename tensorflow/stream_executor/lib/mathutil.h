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

#ifndef TENSORFLOW_STREAM_EXECUTOR_LIB_MATHUTIL_H_
#define TENSORFLOW_STREAM_EXECUTOR_LIB_MATHUTIL_H_
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
class MHTracer_DTPStensorflowPSstream_executorPSlibPSmathutilDTh {
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
   MHTracer_DTPStensorflowPSstream_executorPSlibPSmathutilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPSlibPSmathutilDTh() {
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


#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include "tensorflow/stream_executor/platform/logging.h"
#include "tensorflow/stream_executor/platform/port.h"

namespace stream_executor {
namespace port {

class MathUtil {
 public:
  template <typename IntegralType>
  static IntegralType CeilOfRatio(IntegralType numerator,
                                  IntegralType denominator) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPSlibPSmathutilDTh mht_0(mht_0_v, 204, "", "./tensorflow/stream_executor/lib/mathutil.h", "CeilOfRatio");

    return CeilOrFloorOfRatio<IntegralType, true>(numerator, denominator);
  }
  template <typename IntegralType>
  static IntegralType FloorOfRatio(IntegralType numerator,
                                   IntegralType denominator) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPSlibPSmathutilDTh mht_1(mht_1_v, 212, "", "./tensorflow/stream_executor/lib/mathutil.h", "FloorOfRatio");

    return CeilOrFloorOfRatio<IntegralType, false>(numerator, denominator);
  }
  template <typename IntegralType, bool ceil>
  static IntegralType CeilOrFloorOfRatio(IntegralType numerator,
                                         IntegralType denominator);
};

// ---- CeilOrFloorOfRatio ----
// This is a branching-free, cast-to-double-free implementation.
//
// Casting to double is in general incorrect because of loss of precision
// when casting an int64_t into a double.
//
// There's a bunch of 'recipes' to compute a integer ceil (or floor) on the web,
// and most of them are incorrect.
template<typename IntegralType, bool ceil>
IntegralType MathUtil::CeilOrFloorOfRatio(IntegralType numerator,
                                          IntegralType denominator) {
  static_assert(std::is_integral<IntegralType>::value,
                 "CeilOfRatio_is_only_defined_for_integral_types");
  assert(denominator != 0);
  // Dividing the smallest signed integer by -1 is not supported: it would
  // SIGFPE
  assert(!std::is_signed<IntegralType>::value ||
         numerator != std::numeric_limits<IntegralType>::min() ||
         denominator != -1);

  const IntegralType rounded_toward_zero = numerator / denominator;
  const IntegralType intermediate_product = rounded_toward_zero * denominator;

  if (ceil) {  // Compile-time condition: not an actual branching
    // When rounded_toward_zero is negative, then an adjustment is never needed:
    // the real ratio is negative, and so rounded toward zero is the ceil.
    // When rounded_toward_zero is non-negative, an adjustment is needed if the
    // sign of the difference numerator - intermediate_product is the same as
    // the sign of the denominator.
    //
    // Using a bool and then a static_cast to IntegralType is not strictly
    // necessary, but it makes the code clear, and anyway the compiler should
    // get rid of it.
    const bool needs_adjustment = (rounded_toward_zero >= 0) &&
        ((denominator > 0 && numerator > intermediate_product) ||
            (denominator < 0 && numerator < intermediate_product));
    const IntegralType adjustment = static_cast<IntegralType>(needs_adjustment);
    const IntegralType ceil_of_ratio = rounded_toward_zero + adjustment;
    return ceil_of_ratio;
  } else {
    // Floor case: symmetrical to the previous one
    const bool needs_adjustment = (rounded_toward_zero <= 0) &&
        ((denominator > 0 && numerator < intermediate_product) ||
         (denominator < 0 && numerator > intermediate_product));
    const IntegralType adjustment = static_cast<IntegralType>(needs_adjustment);
    const IntegralType floor_of_ratio = rounded_toward_zero - adjustment;
    return floor_of_ratio;
  }
}

}  // namespace port
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_LIB_MATHUTIL_H_
