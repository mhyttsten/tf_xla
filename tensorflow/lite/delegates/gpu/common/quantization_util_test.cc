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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSquantization_util_testDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSquantization_util_testDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSquantization_util_testDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/gpu/common/quantization_util.h"

#include <stdint.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/util.h"

using ::testing::Eq;
using ::testing::FloatNear;
using ::testing::Pointwise;

namespace tflite {
namespace gpu {
namespace {

std::unique_ptr<TfLiteIntArray, TfLiteIntArrayDeleter> BuildTfLiteIntArray(
    const std::vector<int>& data) {
  std::unique_ptr<TfLiteIntArray, TfLiteIntArrayDeleter> result(
      TfLiteIntArrayCreate(data.size()));
  std::copy(data.begin(), data.end(), result->data);
  return result;
}

// TODO(b/158578883): this function is copied from the Micro codebase. Consider
// moving to a shared location.
void PopulateContext(std::vector<TfLiteTensor>& tensors,
                     TfLiteContext& context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSquantization_util_testDTcc mht_0(mht_0_v, 220, "", "./tensorflow/lite/delegates/gpu/common/quantization_util_test.cc", "PopulateContext");

  context.tensors_size = tensors.size();
  context.tensors = tensors.data();
  context.recommended_num_threads = 1;
}

// TODO(b/158578883): this function is copied from the Micro codebase. Consider
// moving to a shared location.
int ElementCount(const TfLiteIntArray& dims) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSquantization_util_testDTcc mht_1(mht_1_v, 231, "", "./tensorflow/lite/delegates/gpu/common/quantization_util_test.cc", "ElementCount");

  int result = 1;
  for (int i = 0; i < dims.size; ++i) {
    result *= dims.data[i];
  }
  return result;
}

// TODO(b/158578883): this function is copied from the Micro codebase. Consider
// moving to a shared location.
template <typename T>
inline float ScaleFromMinMax(const float min, const float max) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSquantization_util_testDTcc mht_2(mht_2_v, 245, "", "./tensorflow/lite/delegates/gpu/common/quantization_util_test.cc", "ScaleFromMinMax");

  return (max - min) / ((std::numeric_limits<T>::max() * 1.0) -
                        std::numeric_limits<T>::min());
}

// TODO(b/158578883): this function is copied from the Micro codebase. Consider
// moving to a shared location.
template <typename T>
inline int ZeroPointFromMinMax(const float min, const float max) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSquantization_util_testDTcc mht_3(mht_3_v, 256, "", "./tensorflow/lite/delegates/gpu/common/quantization_util_test.cc", "ZeroPointFromMinMax");

  return static_cast<int>(std::numeric_limits<T>::min()) +
         static_cast<int>(-min / ScaleFromMinMax<T>(min, max) + 0.5f);
}

// TODO(b/158578883): this function is copied from the Micro codebase. Consider
// moving to a shared location.
TfLiteTensor CreateQuantizedTensor(const int8_t* data, TfLiteIntArray* dims,
                                   const char* name, float min, float max,
                                   bool is_variable) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSquantization_util_testDTcc mht_4(mht_4_v, 269, "", "./tensorflow/lite/delegates/gpu/common/quantization_util_test.cc", "CreateQuantizedTensor");

  TfLiteTensor result;
  result.type = kTfLiteInt8;
  result.data.int8 = const_cast<int8_t*>(data);
  result.dims = dims;
  result.params = {ScaleFromMinMax<int8_t>(min, max),
                   ZeroPointFromMinMax<int8_t>(min, max)};
  result.allocation_type = kTfLiteMemNone;
  result.bytes = ElementCount(*dims) * sizeof(int8_t);
  result.allocation = nullptr;
  result.name = name;
  result.is_variable = is_variable;
  return result;
}

// TODO(b/158578883): this function is copied from the Micro codebase. Consider
// moving to a shared location.
TfLiteTensor CreateQuantizedTensor(const uint8_t* data, TfLiteIntArray* dims,
                                   const char* name, float min, float max,
                                   bool is_variable) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSquantization_util_testDTcc mht_5(mht_5_v, 292, "", "./tensorflow/lite/delegates/gpu/common/quantization_util_test.cc", "CreateQuantizedTensor");

  TfLiteTensor result;
  result.type = kTfLiteUInt8;
  result.data.uint8 = const_cast<uint8_t*>(data);
  result.dims = dims;
  result.params = {ScaleFromMinMax<uint8_t>(min, max),
                   ZeroPointFromMinMax<uint8_t>(min, max)};
  result.allocation_type = kTfLiteMemNone;
  result.bytes = ElementCount(*dims) * sizeof(uint8_t);
  result.allocation = nullptr;
  result.name = name;
  result.is_variable = false;
  return result;
}

// TODO(b/158578883): this function is copied from the Micro codebase. Consider
// moving to a shared location.
TfLiteTensor CreateTensor(TfLiteIntArray* dims, const char* name,
                          bool is_variable) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSquantization_util_testDTcc mht_6(mht_6_v, 314, "", "./tensorflow/lite/delegates/gpu/common/quantization_util_test.cc", "CreateTensor");

  TfLiteTensor result;
  result.dims = dims;
  result.name = name;
  result.params = {};
  result.quantization = {kTfLiteNoQuantization, nullptr};
  result.is_variable = is_variable;
  result.allocation_type = kTfLiteMemNone;
  result.allocation = nullptr;
  return result;
}

// TODO(b/158578883): this function is copied from the Micro codebase. Consider
// moving to a shared location.
TfLiteTensor CreateFloatTensor(const float* data, TfLiteIntArray* dims,
                               const char* name, bool is_variable) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("name: \"" + (name == nullptr ? std::string("nullptr") : std::string((char*)name)) + "\"");
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPSquantization_util_testDTcc mht_7(mht_7_v, 333, "", "./tensorflow/lite/delegates/gpu/common/quantization_util_test.cc", "CreateFloatTensor");

  TfLiteTensor result = CreateTensor(dims, name, is_variable);
  result.type = kTfLiteFloat32;
  result.data.f = const_cast<float*>(data);
  result.bytes = ElementCount(*dims) * sizeof(float);
  return result;
}

TEST(DequantizeInputs, Int8) {
  TfLiteContext context;
  auto input_dims = BuildTfLiteIntArray({1, 3, 2, 1});
  std::vector<int8_t> data = {-3, -2, -1, 1, 2, 3};
  std::vector<float> dequantized_data(data.size());

  TfLiteTensor input = CreateQuantizedTensor(
      data.data(), input_dims.get(), "input",
      /*min=*/-12.8f, /*max=*/12.7f, /*is_variable=*/false);
  TfLiteTensor dequantized_input = CreateFloatTensor(
      dequantized_data.data(), input_dims.get(), "input_dequant",
      /*is_variable=*/true);

  std::vector<TfLiteTensor> tensors{input, dequantized_input};
  PopulateContext(tensors, context);

  std::vector<uint32_t> input_indices = {1};
  absl::flat_hash_map<int, int> quant_conversion_map = {{1, 0}};

  auto status = DequantizeInputs(&context, input_indices, quant_conversion_map);
  EXPECT_TRUE(status.ok());
  EXPECT_THAT(dequantized_data,
              Pointwise(FloatNear(1e-6), {-0.3, -0.2, -0.1, 0.1, 0.2, 0.3}));
}

TEST(DequantizeInputs, UInt8) {
  TfLiteContext context;
  auto input_dims = BuildTfLiteIntArray({1, 3, 2, 1});
  std::vector<uint8_t> data = {0, 1, 2, 3, 4, 5};
  std::vector<float> dequantized_data(data.size());

  TfLiteTensor input =
      CreateQuantizedTensor(data.data(), input_dims.get(), "input",
                            /*min=*/0.0f, /*max=*/25.5f, /*is_variable=*/false);
  TfLiteTensor dequantized_input = CreateFloatTensor(
      dequantized_data.data(), input_dims.get(), "input_dequant",
      /*is_variable=*/true);

  std::vector<TfLiteTensor> tensors{input, dequantized_input};
  PopulateContext(tensors, context);

  std::vector<int64_t> input_indices = {1};
  absl::flat_hash_map<int, int> quant_conversion_map = {{1, 0}};

  auto status = DequantizeInputs(&context, input_indices, quant_conversion_map);
  EXPECT_TRUE(status.ok());
  EXPECT_THAT(dequantized_data,
              Pointwise(FloatNear(1e-6), {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}));
}

TEST(QuantizeOutputs, Int8) {
  TfLiteContext context;
  auto input_dims = BuildTfLiteIntArray({1, 3, 2, 1});
  std::vector<float> data = {-0.3, -0.2, -0.1, 0.1, 0.2, 0.3};
  std::vector<int8_t> quantized_data(data.size());
  TfLiteTensor output = CreateFloatTensor(data.data(), input_dims.get(),
                                          "output", /*is_variable=*/false);
  TfLiteTensor quantized_output = CreateQuantizedTensor(
      quantized_data.data(), input_dims.get(), "output_quant",
      /*min=*/-12.8f, /*max=*/12.7f, /*is_variable=*/true);

  std::vector<TfLiteTensor> tensors{output, quantized_output};
  PopulateContext(tensors, context);

  std::vector<uint32_t> output_indices = {0};
  absl::flat_hash_map<int, int> quant_conversion_map = {{0, 1}};

  auto status = QuantizeOutputs(&context, output_indices, quant_conversion_map);
  EXPECT_TRUE(status.ok());
  EXPECT_THAT(quantized_data, Pointwise(Eq(), {-3, -2, -1, 1, 2, 3}));
}

TEST(QuantizeOutputs, UInt8) {
  TfLiteContext context;
  auto input_dims = BuildTfLiteIntArray({1, 3, 2, 1});
  std::vector<float> data = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5};
  std::vector<uint8_t> quantized_data(data.size());
  TfLiteTensor output = CreateFloatTensor(data.data(), input_dims.get(),
                                          "output", /*is_variable=*/false);
  TfLiteTensor quantized_output = CreateQuantizedTensor(
      quantized_data.data(), input_dims.get(), "output_quant",
      /*min=*/0.0f, /*max=*/25.5f, /*is_variable=*/true);

  std::vector<TfLiteTensor> tensors{output, quantized_output};
  PopulateContext(tensors, context);

  std::vector<int64_t> output_indices = {0};
  absl::flat_hash_map<int, int> quant_conversion_map = {{0, 1}};

  auto status = QuantizeOutputs(&context, output_indices, quant_conversion_map);
  EXPECT_TRUE(status.ok());
  EXPECT_THAT(quantized_data, Pointwise(Eq(), {0, 1, 2, 3, 4, 5}));
}

}  // namespace
}  // namespace gpu
}  // namespace tflite
