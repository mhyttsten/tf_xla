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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_transpose_conv_testerDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_transpose_conv_testerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_transpose_conv_testerDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/xnnpack/quantized_transpose_conv_tester.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <numeric>
#include <random>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include "fp16.h"  // from @FP16
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace xnnpack {

void QuantizedTransposeConvTester::Test(TfLiteDelegate* delegate) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_transpose_conv_testerDTcc mht_0(mht_0_v, 210, "", "./tensorflow/lite/delegates/xnnpack/quantized_transpose_conv_tester.cc", "QuantizedTransposeConvTester::Test");

  std::vector<char> buffer = CreateTfLiteModel();
  const Model* model = GetModel(buffer.data());

  std::unique_ptr<Interpreter> delegate_interpreter;
  ASSERT_EQ(
      InterpreterBuilder(
          model,
          ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
          &delegate_interpreter),
      kTfLiteOk);
  std::unique_ptr<Interpreter> default_interpreter;
  ASSERT_EQ(
      InterpreterBuilder(
          model,
          ::tflite::ops::builtin::BuiltinOpResolverWithoutDefaultDelegates())(
          &default_interpreter),
      kTfLiteOk);

  ASSERT_TRUE(delegate_interpreter);
  ASSERT_TRUE(default_interpreter);

  ASSERT_EQ(delegate_interpreter->inputs().size(), 1);
  ASSERT_EQ(default_interpreter->inputs().size(), 1);

  ASSERT_EQ(delegate_interpreter->outputs().size(), 1);
  ASSERT_EQ(default_interpreter->outputs().size(), 1);

  ASSERT_EQ(delegate_interpreter->AllocateTensors(), kTfLiteOk);
  ASSERT_EQ(default_interpreter->AllocateTensors(), kTfLiteOk);

  ASSERT_EQ(delegate_interpreter->ModifyGraphWithDelegate(delegate), kTfLiteOk);

  std::random_device random_device;
  auto rng = std::mt19937(random_device());

  const int input_data_size =
      BatchSize() * InputHeight() * InputWidth() * InputChannels();

  // std::uniform_int_distribution<T> is undefined behavior when T is not short,
  // int, long, long long, or their respective unsigned variants:
  // https://en.cppreference.com/w/cpp/numeric/random/uniform_int_distribution.
  auto uint8rng =
      std::bind(std::uniform_int_distribution<int32_t>(0, 255), rng);
  uint8_t* default_input_data = reinterpret_cast<uint8_t*>(
      default_interpreter->input_tensor(0)->data.data);
  std::generate(default_input_data, default_input_data + input_data_size,
                std::ref(uint8rng));

  uint8_t* xnnpack_input_data = reinterpret_cast<uint8_t*>(
      delegate_interpreter->input_tensor(0)->data.data);
  std::copy(default_input_data, default_input_data + input_data_size,
            xnnpack_input_data);

  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  if (Unsigned()) {
    EnsureOutputsClose<uint8_t>(default_interpreter.get(),
                                delegate_interpreter.get());
  } else {
    EnsureOutputsClose<int8_t>(default_interpreter.get(),
                               delegate_interpreter.get());
  }
}

template <typename WeightType>
void QuantizedTransposeConvTester::EnsureOutputsClose(
    const Interpreter* default_interpreter,
    const Interpreter* delegate_interpreter) const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_transpose_conv_testerDTcc mht_1(mht_1_v, 282, "", "./tensorflow/lite/delegates/xnnpack/quantized_transpose_conv_tester.cc", "QuantizedTransposeConvTester::EnsureOutputsClose");

  const WeightType* default_output_data =
      default_interpreter->typed_output_tensor<WeightType>(0);
  const WeightType* xnnpack_output_data =
      delegate_interpreter->typed_output_tensor<WeightType>(0);

  const size_t output_data_size =
      BatchSize() * OutputHeight() * OutputWidth() * OutputChannels();

  const int kQuantizationErrorTolerance = 1;

  for (size_t i = 0; i < output_data_size; i++) {
    const int diff = static_cast<int>(default_output_data[i]) -
                     static_cast<int>(xnnpack_output_data[i]);
    ASSERT_LE(std::abs(diff), kQuantizationErrorTolerance);
  }
}

std::vector<char> QuantizedTransposeConvTester::CreateTfLiteModel() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSquantized_transpose_conv_testerDTcc mht_2(mht_2_v, 303, "", "./tensorflow/lite/delegates/xnnpack/quantized_transpose_conv_tester.cc", "QuantizedTransposeConvTester::CreateTfLiteModel");

  std::random_device random_device;
  auto rng = std::mt19937(random_device());

  const std::vector<int32_t> input_shape = {BatchSize(), InputHeight(),
                                            InputWidth(), InputChannels()};
  const std::vector<int32_t> output_shape = {BatchSize(), OutputHeight(),
                                             OutputWidth(), OutputChannels()};
  const std::vector<int32_t> filter_shape = {OutputChannels(), KernelHeight(),
                                             KernelWidth(), InputChannels()};
  const std::vector<int32_t> bias_shape = {OutputChannels()};

  flatbuffers::FlatBufferBuilder builder;

  std::vector<flatbuffers::Offset<OperatorCode>> operator_codes;

  std::vector<flatbuffers::Offset<tflite::Operator>> operators;
  std::vector<flatbuffers::Offset<Tensor>> tensors;

  // Buffer 0 is a sentinel as required by the schema, means "no buffer".
  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers = {
      CreateBuffer(builder, builder.CreateVector({}))};
  const int kNoBuffer = 0;

  // Create a tensor containing the expected output shape.
  const int buffer_index_output_shape = buffers.size();
  buffers.emplace_back(CreateBuffer(
      builder, builder.CreateVector(
                   reinterpret_cast<const uint8_t*>(output_shape.data()),
                   sizeof(int32_t) * output_shape.size())));

  std::vector<int32_t> output_shape_tensor_shape = {4};
  const int tensor_index_output_shape = tensors.size();
  tensors.emplace_back(
      CreateTensorDirect(builder, &output_shape_tensor_shape, TensorType_INT32,
                         /*buffer=*/buffer_index_output_shape));

  flatbuffers::Offset<::tflite::QuantizationParameters>
      quantization_parameters = 0;

  std::vector<uint8_t> filter_data(OutputChannels() * KernelHeight() *
                                   KernelWidth() * InputChannels());

  auto uint8rng =
      std::bind(std::uniform_int_distribution<int32_t>(0, 255), rng);
  std::generate(filter_data.begin(), filter_data.end(), uint8rng);

  const int buffer_index_filter = buffers.size();
  buffers.emplace_back(CreateBuffer(
      builder,
      builder.CreateVector(reinterpret_cast<const uint8_t*>(filter_data.data()),
                           sizeof(uint8_t) * filter_data.size())));

  const ::tflite::TensorType input_tensor_type =
      Unsigned() ? ::tflite::TensorType_UINT8 : ::tflite::TensorType_INT8;

  auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);
  const float quantization_scale = f32rng();
  int64_t zero_point = 0;
  if (Unsigned()) {
    zero_point = std::accumulate(filter_data.begin(), filter_data.end(), 0) /
                 filter_data.size();
  }

  quantization_parameters = CreateQuantizationParameters(
      builder, /*min=*/0, /*max=*/0,
      builder.CreateVector<float>({quantization_scale}),
      builder.CreateVector<int64_t>({zero_point}));
  tensors.emplace_back(CreateTensorDirect(
      builder, &filter_shape, input_tensor_type, buffer_index_filter,
      /*name=*/nullptr, quantization_parameters));

  if (UseBias()) {
    const int32_t kMaxAbsBias = 10000;
    auto int32rng = std::bind(
        std::uniform_int_distribution<int32_t>(-kMaxAbsBias, kMaxAbsBias), rng);
    std::vector<int32_t> bias_data(OutputChannels());
    std::generate(bias_data.begin(), bias_data.end(), int32rng);

    const int buffer_index_bias = buffers.size();
    buffers.emplace_back(CreateBuffer(
        builder,
        builder.CreateVector(reinterpret_cast<const uint8_t*>(bias_data.data()),
                             sizeof(int32_t) * bias_data.size())));

    // TFLite checks that bias quantization scale is close to that of the
    // input and filter quantization scales multiplied.
    const float bias_quantization_scale =
        quantization_scale * quantization_scale;
    auto bias_quantization_parameters = CreateQuantizationParameters(
        builder, /*min=*/0, /*max=*/0,
        /*scale=*/builder.CreateVector<float>({bias_quantization_scale}),
        /*zero_point=*/builder.CreateVector<int64_t>({0}));

    tensors.emplace_back(
        CreateTensorDirect(builder, &bias_shape, TensorType_INT32,
                           /*buffer=*/buffer_index_bias,
                           /*name=*/nullptr, bias_quantization_parameters));
  }

  const int top_tensor = tensors.size() - 1;
  const int tensor_index_filter = UseBias() ? top_tensor - 1 : top_tensor;

  const int tensor_index_input = tensors.size();
  tensors.emplace_back(
      CreateTensorDirect(builder, &input_shape, input_tensor_type, kNoBuffer,
                         /*name=*/nullptr, quantization_parameters));

  std::vector<int32_t> op_inputs = {tensor_index_output_shape,
                                    tensor_index_filter, tensor_index_input};
  if (UseBias()) {
    const int tensor_index_bias = top_tensor;
    op_inputs.push_back(tensor_index_bias);
  }

  const int tensor_index_output = tensors.size();
  tensors.emplace_back(
      CreateTensorDirect(builder, &output_shape, input_tensor_type, kNoBuffer,
                         /*name=*/nullptr, quantization_parameters));

  const std::vector<int32_t> op_outputs = {tensor_index_output};

  const int opcode_index_transpose_conv = operator_codes.size();
  operator_codes.emplace_back(
      CreateOperatorCode(builder, BuiltinOperator_TRANSPOSE_CONV));

  flatbuffers::Offset<TransposeConvOptions> transpose_conv_options =
      CreateTransposeConvOptions(builder, Padding(), StrideWidth(),
                                 StrideHeight());
  operators.emplace_back(CreateOperatorDirect(
      builder, /*opcode_index=*/opcode_index_transpose_conv, &op_inputs,
      &op_outputs, BuiltinOptions_TransposeConvOptions,
      transpose_conv_options.Union()));

  const std::vector<int32_t> subgraph_inputs = {tensor_index_input};
  const std::vector<int32_t> subgraph_outputs = {tensor_index_output};
  flatbuffers::Offset<SubGraph> subgraph = CreateSubGraphDirect(
      builder, &tensors, &subgraph_inputs, &subgraph_outputs, &operators);

  flatbuffers::Offset<flatbuffers::String> description =
      builder.CreateString("Quantized TransposeConv model");

  flatbuffers::Offset<Model> model_buffer = CreateModel(
      builder, TFLITE_SCHEMA_VERSION,
      builder.CreateVector(operator_codes.data(), operator_codes.size()),
      builder.CreateVector(&subgraph, 1), description,
      builder.CreateVector(buffers.data(), buffers.size()));

  builder.Finish(model_buffer);

  return std::vector<char>(builder.GetBufferPointer(),
                           builder.GetBufferPointer() + builder.GetSize());
}

}  // namespace xnnpack
}  // namespace tflite
