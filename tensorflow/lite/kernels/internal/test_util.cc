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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStest_utilDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStest_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStest_utilDTcc() {
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
#include "tensorflow/lite/kernels/internal/test_util.h"

#include <cmath>
#include <iterator>

namespace tflite {

// this is a copied from an internal function in propagate_fixed_sizes.cc
bool ComputeConvSizes(const RuntimeShape& input_shape, int output_depth,
                      int filter_width, int filter_height, int stride,
                      int dilation_width_factor, int dilation_height_factor,
                      PaddingType padding_type, RuntimeShape* output_shape,
                      int* pad_width, int* pad_height) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStest_utilDTcc mht_0(mht_0_v, 196, "", "./tensorflow/lite/kernels/internal/test_util.cc", "ComputeConvSizes");

  const int input_width = input_shape.Dims(2);
  const int input_height = input_shape.Dims(1);
  const int batch = input_shape.Dims(0);

  int dilated_filter_width = dilation_width_factor * (filter_width - 1) + 1;
  int dilated_filter_height = dilation_height_factor * (filter_height - 1) + 1;

  int output_height = 0;
  int output_width = 0;
  if (padding_type == PaddingType::kValid) {
    // Official TF is
    // ceil((input_height - (dilated_filter_height - 1)) / stride),
    // implemented as
    // floor(
    //   (input_height - (dilated_filter_height - 1) + (stride - 1)) / stride).
    output_height = (input_height + stride - dilated_filter_height) / stride;
    output_width = (input_width + stride - dilated_filter_width) / stride;
  } else if (padding_type == PaddingType::kSame) {
    output_height = (input_height + stride - 1) / stride;
    output_width = (input_width + stride - 1) / stride;
  } else {
    return false;
  }

  if (output_width <= 0 || output_height <= 0) {
    return false;
  }

  *pad_height = std::max(
      0, ((output_height - 1) * stride + dilated_filter_height - input_height) /
             2);
  *pad_width = std::max(
      0,
      ((output_width - 1) * stride + dilated_filter_width - input_width) / 2);

  output_shape->BuildFrom({batch, output_height, output_width, output_depth});
  return true;
}

std::mt19937& RandomEngine() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStest_utilDTcc mht_1(mht_1_v, 239, "", "./tensorflow/lite/kernels/internal/test_util.cc", "RandomEngine");

  static std::mt19937 engine;
  return engine;
}

int UniformRandomInt(int min, int max) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStest_utilDTcc mht_2(mht_2_v, 247, "", "./tensorflow/lite/kernels/internal/test_util.cc", "UniformRandomInt");

  std::uniform_int_distribution<int> dist(min, max);
  return dist(RandomEngine());
}

float UniformRandomFloat(float min, float max) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStest_utilDTcc mht_3(mht_3_v, 255, "", "./tensorflow/lite/kernels/internal/test_util.cc", "UniformRandomFloat");

  std::uniform_real_distribution<float> dist(min, max);
  return dist(RandomEngine());
}

int ExponentialRandomPositiveInt(float percentile, int percentile_val,
                                 int max_val) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStest_utilDTcc mht_4(mht_4_v, 264, "", "./tensorflow/lite/kernels/internal/test_util.cc", "ExponentialRandomPositiveInt");

  const float lambda =
      -std::log(1.f - percentile) / static_cast<float>(percentile_val);
  std::exponential_distribution<float> dist(lambda);
  float val;
  do {
    val = dist(RandomEngine());
  } while (!val || !std::isfinite(val) || val > max_val);
  return static_cast<int>(std::ceil(val));
}

float ExponentialRandomPositiveFloat(float percentile, float percentile_val,
                                     float max_val) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStest_utilDTcc mht_5(mht_5_v, 279, "", "./tensorflow/lite/kernels/internal/test_util.cc", "ExponentialRandomPositiveFloat");

  const float lambda =
      -std::log(1.f - percentile) / static_cast<float>(percentile_val);
  std::exponential_distribution<float> dist(lambda);
  float val;
  do {
    val = dist(RandomEngine());
  } while (!std::isfinite(val) || val > max_val);
  return val;
}

void FillRandomFloat(std::vector<float>* vec, float min, float max) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPStest_utilDTcc mht_6(mht_6_v, 293, "", "./tensorflow/lite/kernels/internal/test_util.cc", "FillRandomFloat");

  std::uniform_real_distribution<float> dist(min, max);
  // TODO(b/154540105): use std::ref to avoid copying the random engine.
  auto gen = std::bind(dist, RandomEngine());
  std::generate(std::begin(*vec), std::end(*vec), gen);
}

}  // namespace tflite
