/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LOGISTIC_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LOGISTIC_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlogisticDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlogisticDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlogisticDTh() {
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


#include <cmath>

#include "fixedpoint/fixedpoint.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace reference_ops {

inline void Logistic(const RuntimeShape& input_shape, const float* input_data,
                     const RuntimeShape& output_shape, float* output_data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlogisticDTh mht_0(mht_0_v, 200, "", "./tensorflow/lite/kernels/internal/reference/logistic.h", "Logistic");

  const float cutoff_upper = 16.619047164916992188f;
  const float cutoff_lower = -9.f;

  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  // Rational for using approximation in reference kernel.
  // 0. This approximation gives enough precision for float.
  // 1. This works around an issue on an embedded chipset where exp() does not
  // return correctly as expected - exp(x) should return inf when overflown
  // not 1.701417   IEEE 754 defines representation for inf.
  // 2. This will speed up calculation and is matching the behavior in the
  // optimized kernels. (check the definition of scalar_logistic_op<float>)

  for (int i = 0; i < flat_size; i++) {
    float val = input_data[i];
    float result;
    if (val > cutoff_upper) {
      result = 1.0f;
    } else if (val < cutoff_lower) {
      result = std::exp(val);
    } else {
      result = 1.f / (1.f + std::exp(-val));
    }
    output_data[i] = result;
  }
}

// Convenience version that allows, for example, generated-code calls to be
// uniform between data types.
inline void Logistic(const LogisticParams&, const RuntimeShape& input_shape,
                     const float* input_data, const RuntimeShape& output_shape,
                     float* output_data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlogisticDTh mht_1(mht_1_v, 235, "", "./tensorflow/lite/kernels/internal/reference/logistic.h", "Logistic");

  // Drop params: not needed.
  Logistic(input_shape, input_data, output_shape, output_data);
}

inline void Logistic(const LogisticParams& params,
                     const RuntimeShape& input_shape, const int16_t* input_data,
                     const RuntimeShape& output_shape, int16_t* output_data) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlogisticDTh mht_2(mht_2_v, 245, "", "./tensorflow/lite/kernels/internal/reference/logistic.h", "Logistic");

  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  for (int i = 0; i < flat_size; i++) {
    // F0 uses 0 integer bits, range [-1, 1].
    // This is the return type of math functions such as tanh, logistic,
    // whose range is in [-1, 1].
    using F0 = gemmlowp::FixedPoint<std::int16_t, 0>;
    // F3 uses 3 integer bits, range [-8, 8], the input range expected here.
    using F3 = gemmlowp::FixedPoint<std::int16_t, 3>;

    const F3 input = F3::FromRaw(input_data[i]);
    F0 output = gemmlowp::logistic(input);
    output_data[i] = output.raw();
  }
}

// Quantized int8_t logistic activation.  Cheats by dequantizing and
// requantizing around the floating point logistic method.  This implementation
// is slow on platforms without a floating point unit.

// TODO(b/141211002): Delete this int8_t implementation once we can reuse the
// approach used in TFLite for int8_t Logistic.
inline void Logistic(const RuntimeShape& input_shape, const int8_t* input_data,
                     float input_scale, int input_zero_point,
                     const RuntimeShape& output_shape, int8_t* output_data,
                     float output_scale, int output_zero_point) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSreferencePSlogisticDTh mht_3(mht_3_v, 274, "", "./tensorflow/lite/kernels/internal/reference/logistic.h", "Logistic");

  const float cutoff_upper = 16.619047164916992188f;
  const float cutoff_lower = -9.f;

  const int flat_size = MatchingFlatSize(input_shape, output_shape);

  // Rational for using approximation in reference kernel.
  // 0. This approximation gives enough precision for float.
  // 1. This works around an issue on an embedded chipset where exp() does not
  // return correctly as expected - exp(x) should return inf when overflown
  // not 1.701417   IEEE 754 defines representation for inf.
  // 2. This will speed up calculation and is matching the behavior in the
  // optimized kernels. (check the definition of scalar_logistic_op<float>)

  for (int i = 0; i < flat_size; i++) {
    // Dequantize.
    float val =
        static_cast<float>((input_data[i] - input_zero_point) * input_scale);
    float result;
    if (val > cutoff_upper) {
      result = 1.0f;
    } else if (val < cutoff_lower) {
      result = std::exp(val);
    } else {
      result = 1.f / (1.f + std::exp(-val));
    }
    // Requantize
    int8_t output =
        static_cast<int8_t>(result / output_scale + output_zero_point);
    output_data[i] = output;
  }
}

}  // namespace reference_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_REFERENCE_LOGISTIC_H_
