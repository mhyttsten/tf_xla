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
class MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSaveragepool_quantized_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSaveragepool_quantized_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSaveragepool_quantized_testDTcc() {
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
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <iterator>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/internal/optimized/integer_ops/pooling.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h"
#include "tensorflow/lite/kernels/internal/test_util.h"

namespace tflite {
namespace {

// Runs the reference and optimized AveragePool functions and asserts the values
// are the same.
void RunOneAveragePoolTest(const PoolParams& params,
                           const RuntimeShape& input_shape,
                           const int8* input_data,
                           const RuntimeShape& output_shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSaveragepool_quantized_testDTcc mht_0(mht_0_v, 207, "", "./tensorflow/lite/kernels/internal/averagepool_quantized_test.cc", "RunOneAveragePoolTest");

  const int buffer_size = output_shape.FlatSize();
  std::vector<int8> optimized_averagePool_output(buffer_size);
  std::vector<int8> reference_averagePool_output(buffer_size);

  bool reference_success = reference_integer_ops::AveragePool(
      params, input_shape, input_data, output_shape,
      reference_averagePool_output.data());
  bool optimized_success = optimized_integer_ops::AveragePool(
      params, input_shape, input_data, output_shape,
      optimized_averagePool_output.data());
  EXPECT_TRUE(reference_success);
  EXPECT_TRUE(optimized_success);

  for (int i = 0; i < buffer_size; i++) {
    EXPECT_TRUE(reference_averagePool_output[i] ==
                optimized_averagePool_output[i]);
  }
}

// Creates random input shape (batch, height, width, depth), then computes
// output shape based on value of `padding_same`:
// `padding_same` == true, calculate output with padding == "SAME"
// `padding_same` == false, calculate output with padding == "VALID"
// With input/output shapes computed, fills the input data and calls the
// test function.
void CreateDataAndRunAveragePool(bool padding_same) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSaveragepool_quantized_testDTcc mht_1(mht_1_v, 236, "", "./tensorflow/lite/kernels/internal/averagepool_quantized_test.cc", "CreateDataAndRunAveragePool");

  const int batch = UniformRandomInt(1, 2);
  const int input_depth = UniformRandomInt(1, 700);
  const int output_depth = input_depth;
  const int input_width_offset = UniformRandomInt(1, 30);
  const int input_height_offset = UniformRandomInt(1, 30);
  const int stride_width = UniformRandomInt(1, 10);
  const int stride_height = UniformRandomInt(1, 10);
  const int filter_width = UniformRandomInt(1, 10);
  const int filter_height = UniformRandomInt(1, 10);
  const int input_width = input_width_offset + filter_width;
  const int input_height = input_height_offset + filter_height;
  const int output_width =
      padding_same ? (input_width + stride_width - 1) / stride_width
                   : (input_width - filter_width + stride_width) / stride_width;
  const int output_height =
      padding_same
          ? (input_height + stride_height - 1) / stride_height
          : (input_height - filter_height + stride_height) / stride_height;

  auto input_shape =
      RuntimeShape({batch, input_height, input_width, input_depth});
  auto output_shape =
      RuntimeShape({batch, output_height, output_width, output_depth});
  const int buffer_size = input_shape.FlatSize();
  std::vector<int8> input_data(buffer_size);
  FillRandom(&input_data);

  PoolParams params;
  params.stride_height = stride_height;
  params.stride_width = stride_width;
  params.filter_height = filter_height;
  params.filter_width = filter_width;
  params.quantized_activation_min =
      static_cast<int8_t>(std::numeric_limits<int8_t>::lowest());
  params.quantized_activation_max =
      static_cast<int8_t>(std::numeric_limits<int8_t>::max());
  auto compute_padding = [](int stride, int in_size, int filter_size,
                            int out_size) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSaveragepool_quantized_testDTcc mht_2(mht_2_v, 277, "", "./tensorflow/lite/kernels/internal/averagepool_quantized_test.cc", "lambda");

    int padding = ((out_size - 1) * stride + filter_size - in_size) / 2;
    return padding > 0 ? padding : 0;
  };
  params.padding_values.width =
      compute_padding(stride_width, input_width, filter_width, output_width);
  params.padding_values.height = compute_padding(stride_height, input_height,
                                                 filter_height, output_height);
  RunOneAveragePoolTest(params, input_shape, input_data.data(), output_shape);
}

TEST(TestAveragePool, SymmetricQuantAveragePool) {
  const int kTestsToRun = 10;
  for (int i = 0; i < kTestsToRun; i++) {
    CreateDataAndRunAveragePool(/*padding_same=*/true);
    CreateDataAndRunAveragePool(/*padding_same=*/false);
  }
}

// Creates random input shape (batch, height, width, depth), then computes
// output shape based on value of `padding_same`:
// `padding_same` == true, calculate output with padding == "SAME"
// `padding_same` == false, calculate output with padding == "VALID"
// With input/output shapes computed, fills the input data and calls the
// test function.
void CreateExtremalDataAndRunAveragePool(bool padding_same) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSaveragepool_quantized_testDTcc mht_3(mht_3_v, 305, "", "./tensorflow/lite/kernels/internal/averagepool_quantized_test.cc", "CreateExtremalDataAndRunAveragePool");

  const int batch = UniformRandomInt(1, 2);
  const int input_depth = UniformRandomInt(1, 700);
  const int output_depth = input_depth;
  const int input_width_offset = UniformRandomInt(1, 30);
  const int input_height_offset = UniformRandomInt(1, 30);
  const int stride_width = UniformRandomInt(1, 128);
  const int stride_height = UniformRandomInt(1, 128);
  const int filter_width = UniformRandomInt(1, 28);
  const int filter_height = UniformRandomInt(1, 28);
  if (filter_width * filter_height > 64) {
    std::cout << "should test 32 version" << std::endl;
  }
  const int input_width = input_width_offset + filter_width;
  const int input_height = input_height_offset + filter_height;
  const int output_width =
      padding_same ? (input_width + stride_width - 1) / stride_width
                   : (input_width - filter_width + stride_width) / stride_width;
  const int output_height =
      padding_same
          ? (input_height + stride_height - 1) / stride_height
          : (input_height - filter_height + stride_height) / stride_height;

  auto input_shape =
      RuntimeShape({batch, input_height, input_width, input_depth});
  auto output_shape =
      RuntimeShape({batch, output_height, output_width, output_depth});

  PoolParams params;
  params.stride_height = stride_height;
  params.stride_width = stride_width;
  params.filter_height = filter_height;
  params.filter_width = filter_width;
  params.quantized_activation_min =
      static_cast<int8_t>(std::numeric_limits<int8_t>::lowest());
  params.quantized_activation_max =
      static_cast<int8_t>(std::numeric_limits<int8_t>::max());
  auto compute_padding = [](int stride, int in_size, int filter_size,
                            int out_size) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSinternalPSaveragepool_quantized_testDTcc mht_4(mht_4_v, 346, "", "./tensorflow/lite/kernels/internal/averagepool_quantized_test.cc", "lambda");

    int padding = ((out_size - 1) * stride + filter_size - in_size) / 2;
    return padding > 0 ? padding : 0;
  };
  params.padding_values.width =
      compute_padding(stride_width, input_width, filter_width, output_width);
  params.padding_values.height = compute_padding(stride_height, input_height,
                                                 filter_height, output_height);

  const int buffer_size = input_shape.FlatSize();
  std::vector<int8> input_data(buffer_size);

  // Test small values
  int8 min = std::numeric_limits<int8>::min();
  int8 max = std::numeric_limits<int8>::min() + 10;
  FillRandom(&input_data, min, max);
  RunOneAveragePoolTest(params, input_shape, input_data.data(), output_shape);

  // Test large values
  min = std::numeric_limits<int8>::max() - 10;
  max = std::numeric_limits<int8>::max();
  FillRandom(&input_data, min, max);
  RunOneAveragePoolTest(params, input_shape, input_data.data(), output_shape);
}

TEST(TestAveragePool, SymmetricQuantExtremalAveragePool) {
  CreateExtremalDataAndRunAveragePool(/*padding_same=*/true);
  CreateExtremalDataAndRunAveragePool(/*padding_same=*/false);
}

}  // namespace
}  // namespace tflite
