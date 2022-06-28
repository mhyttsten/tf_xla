/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_TEST_UTIL_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_TEST_UTIL_H_
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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStest_utilDTh {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStest_utilDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStest_utilDTh() {
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
#include <functional>
#include <iterator>
#include <limits>
#include <random>
#include <vector>

#include "tensorflow/lite/kernels/internal/types.h"

namespace tflite {

// Computes output and padding dimensions.
bool ComputeConvSizes(const RuntimeShape& input_shape, int output_depth,
                      int filter_width, int filter_height, int stride,
                      int dilation_width_factor, int dilation_height_factor,
                      PaddingType padding_type, RuntimeShape* output_shape,
                      int* pad_width, int* pad_height);

// Returns a mt19937 random engine.
std::mt19937& RandomEngine();

// Returns a random integer uniformly distributed between |min| and |max|.
int UniformRandomInt(int min, int max);

// Returns a random float uniformly distributed between |min| and |max|.
float UniformRandomFloat(float min, float max);

// Returns a random element in |v|.
template <typename T>
const T& RandomElement(const std::vector<T>& v) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStest_utilDTh mht_0(mht_0_v, 216, "", "./tensorflow/lite/kernels/internal/test_util.h", "RandomElement");

  return v[UniformRandomInt(0, v.size() - 1)];
}

// Returns a random exponentially distributed integer.
int ExponentialRandomPositiveInt(float percentile, int percentile_val,
                                 int max_val);

// Returns a random exponentially distributed float.
float ExponentialRandomPositiveFloat(float percentile, float percentile_val,
                                     float max_val);

// Fills a vector with random floats between |min| and |max|.
void FillRandomFloat(std::vector<float>* vec, float min, float max);

template <typename T>
void FillRandom(typename std::vector<T>::iterator begin_it,
                typename std::vector<T>::iterator end_it, T min, T max) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStest_utilDTh mht_1(mht_1_v, 236, "", "./tensorflow/lite/kernels/internal/test_util.h", "FillRandom");

  // Workaround for compilers that don't support (u)int8_t uniform_distribution.
  typedef typename std::conditional<sizeof(T) >= sizeof(int16_t), T,
                                    std::int16_t>::type rand_type;
  std::uniform_int_distribution<rand_type> dist(min, max);
  // TODO(b/154540105): use std::ref to avoid copying the random engine.
  auto gen = std::bind(dist, RandomEngine());
  std::generate(begin_it, end_it, [&gen] { return static_cast<T>(gen()); });
}

// Fills a vector with random numbers between |min| and |max|.
template <typename T>
void FillRandom(std::vector<T>* vec, T min, T max) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStest_utilDTh mht_2(mht_2_v, 251, "", "./tensorflow/lite/kernels/internal/test_util.h", "FillRandom");

  FillRandom(std::begin(*vec), std::end(*vec), min, max);
}

// Template specialization for float.
template <>
inline void FillRandom<float>(std::vector<float>* vec, float min, float max) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStest_utilDTh mht_3(mht_3_v, 260, "", "./tensorflow/lite/kernels/internal/test_util.h", "FillRandom<float>");

  FillRandomFloat(vec, min, max);
}

// Fills a vector with random numbers.
template <typename T>
void FillRandom(std::vector<T>* vec) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStest_utilDTh mht_4(mht_4_v, 269, "", "./tensorflow/lite/kernels/internal/test_util.h", "FillRandom");

  FillRandom(vec, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
}

// Fill with a "skyscraper" pattern, in which there is a central section (across
// the depth) with higher values than the surround.
template <typename T>
void FillRandomSkyscraper(std::vector<T>* vec, int depth,
                          double middle_proportion, uint8 middle_min,
                          uint8 sides_max) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStest_utilDTh mht_5(mht_5_v, 281, "", "./tensorflow/lite/kernels/internal/test_util.h", "FillRandomSkyscraper");

  for (auto base_it = std::begin(*vec); base_it != std::end(*vec);
       base_it += depth) {
    auto left_it = base_it + std::ceil(0.5 * depth * (1.0 - middle_proportion));
    auto right_it =
        base_it + std::ceil(0.5 * depth * (1.0 + middle_proportion));
    FillRandom(base_it, left_it, std::numeric_limits<T>::min(), sides_max);
    FillRandom(left_it, right_it, middle_min, std::numeric_limits<T>::max());
    FillRandom(right_it, base_it + depth, std::numeric_limits<T>::min(),
               sides_max);
  }
}

}  // namespace tflite
#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_TEST_UTIL_H_
