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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTcc() {
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

#include "tensorflow/lite/delegates/xnnpack/transpose_conv_tester.h"

#include <cassert>
#include <cstdint>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "fp16.h"  // from @FP16
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/delegates/xnnpack/test_util.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace xnnpack {

void TransposeConvTester::Test(TfLiteDelegate* delegate) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTcc mht_0(mht_0_v, 209, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.cc", "TransposeConvTester::Test");

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);

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

  const int input_data_size =
      BatchSize() * InputHeight() * InputWidth() * InputChannels();
  float* default_input_data = default_interpreter->typed_input_tensor<float>(0);
  std::generate(default_input_data, default_input_data + input_data_size,
                std::ref(f32rng));

  float* xnnpack_input_data =
      delegate_interpreter->typed_input_tensor<float>(0);
  std::copy(default_input_data, default_input_data + input_data_size,
            xnnpack_input_data);

  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  float* default_output_data = default_interpreter->typed_tensor<float>(
      default_interpreter->outputs()[0]);
  float* xnnpack_output_data = delegate_interpreter->typed_tensor<float>(
      delegate_interpreter->outputs()[0]);

  const int output_data_size =
      BatchSize() * OutputHeight() * OutputWidth() * OutputChannels();
  for (size_t i = 0; i < output_data_size; i++) {
    ASSERT_NEAR(default_output_data[i], xnnpack_output_data[i],
                std::numeric_limits<float>::epsilon() *
                    std::max(std::abs(default_output_data[i]) * 25.0f, 1.0f));
  }
}

std::vector<char> TransposeConvTester::CreateTfLiteModel() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPStranspose_conv_testerDTcc mht_1(mht_1_v, 277, "", "./tensorflow/lite/delegates/xnnpack/transpose_conv_tester.cc", "TransposeConvTester::CreateTfLiteModel");

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);

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

  // The last one (two) tensor(s) will be the float32 kernel (and bias if used).
  if (FP16Weights()) {
    const int kOpCodeIndexDequantize = operator_codes.size();
    operator_codes.emplace_back(
        CreateOperatorCode(builder, BuiltinOperator_DEQUANTIZE));

    auto f16rng = std::bind(fp16_ieee_from_fp32_value, f32rng);

    std::vector<uint16_t> filter_data(OutputChannels() * KernelHeight() *
                                      KernelWidth() * InputChannels());

    std::generate(filter_data.begin(), filter_data.end(), f16rng);

    const int buffer_index_filter = buffers.size();
    buffers.emplace_back(CreateBuffer(
        builder, builder.CreateVector(
                     reinterpret_cast<const uint8_t*>(filter_data.data()),
                     sizeof(uint16_t) * filter_data.size())));

    const int tensor_index_float16_filter = tensors.size();
    tensors.emplace_back(CreateTensorDirect(builder, &filter_shape,
                                            TensorType_FLOAT16,
                                            /*buffer=*/buffer_index_filter));

    const int kInvalidIndex = -1;
    int tensor_index_float16_bias = kInvalidIndex;
    if (UseBias()) {
      std::vector<uint16_t> bias_data(OutputChannels());
      std::generate(bias_data.begin(), bias_data.end(), f16rng);

      const int buffer_index_bias = buffers.size();
      buffers.emplace_back(CreateBuffer(
          builder, builder.CreateVector(
                       reinterpret_cast<const uint8_t*>(bias_data.data()),
                       sizeof(uint16_t) * bias_data.size())));

      tensor_index_float16_bias = tensors.size();
      tensors.emplace_back(CreateTensorDirect(builder, &bias_shape,
                                              TensorType_FLOAT16,
                                              /*buffer=*/buffer_index_bias));
    }

    const int tensor_index_filter = tensors.size();
    tensors.emplace_back(CreateTensorDirect(
        builder, &filter_shape, TensorType_FLOAT32, /*buffer=*/kNoBuffer));

    const std::vector<int32_t> dequantize_filter_inputs = {
        tensor_index_float16_filter};
    const std::vector<int32_t> dequantize_filter_outputs{tensor_index_filter};
    operators.emplace_back(CreateOperatorDirect(
        builder, /*opcode_index=*/kOpCodeIndexDequantize,
        &dequantize_filter_inputs, &dequantize_filter_outputs));

    assert(tensor_index_filter + 1 == tensors.size());

    if (UseBias()) {
      const int tensor_index_bias = tensors.size();
      tensors.emplace_back(CreateTensorDirect(
          builder, &bias_shape, TensorType_FLOAT32, /*buffer=*/kNoBuffer));

      const std::vector<int32_t> dequantize_bias_inputs = {
          tensor_index_float16_bias};
      const std::vector<int32_t> dequantize_bias_outputs = {tensor_index_bias};
      operators.emplace_back(CreateOperatorDirect(
          builder, /*opcode_index=*/kOpCodeIndexDequantize,
          &dequantize_bias_inputs, &dequantize_bias_outputs));

      assert(tensor_index_bias + 1 == tensors.size());
    }
  } else if (INT8Weights() || INT8ChannelWiseWeights()) {
    const int kOpCodeIndexDequantize = operator_codes.size();
    operator_codes.emplace_back(
        CreateOperatorCode(builder, BuiltinOperator_DEQUANTIZE));

    std::vector<float> filter_data(OutputChannels() * KernelHeight() *
                                   KernelWidth() * InputChannels());
    std::generate(filter_data.begin(), filter_data.end(), f32rng);

    std::vector<float> filter_scales;
    std::vector<int64_t> filter_zero_points;
    int32_t filter_quantized_dimension = 0;
    std::vector<int8_t> quantized_filter_data(filter_data.size());
    if (INT8Weights()) {
      filter_scales.resize(1, GetInt8QuantizationScale(filter_data));
      filter_zero_points.resize(1, 0);
      std::transform(
          filter_data.begin(), filter_data.end(), quantized_filter_data.begin(),
          std::bind(QuantizeInt8, std::placeholders::_1, 0, filter_scales[0]));
    } else {
      filter_quantized_dimension =
          static_cast<int32_t>(filter_shape.size()) - 1;
      const int32_t num_scales = filter_shape[filter_quantized_dimension];
      filter_scales = GetInt8QuantizationScalePerChannel(
          filter_data.data(), filter_quantized_dimension, filter_shape);
      filter_zero_points.resize(num_scales, 0);
      QuantizeInt8PerChannel(filter_scales.data(), filter_zero_points.data(),
                             filter_quantized_dimension, filter_data.data(),
                             quantized_filter_data.data(), filter_shape);
    }
    const int buffer_index_filter = buffers.size();
    buffers.emplace_back(CreateBuffer(
        builder,
        builder.CreateVector(
            reinterpret_cast<const uint8_t*>(quantized_filter_data.data()),
            sizeof(int8_t) * quantized_filter_data.size())));

    const int tensor_index_int8_filter = tensors.size();
    tensors.emplace_back(CreateTensorDirect(
        builder, &filter_shape, TensorType_INT8,
        /*buffer=*/buffer_index_filter, /*name=*/nullptr,
        CreateQuantizationParameters(
            builder, /*min=*/0, /*max=*/0,
            builder.CreateVector<float>(filter_scales),
            builder.CreateVector<int64_t>(filter_zero_points),
            /*details_type=*/QuantizationDetails_NONE,
            /*details=*/0, filter_quantized_dimension)));

    const int tensor_index_filter = tensors.size();
    tensors.emplace_back(CreateTensorDirect(
        builder, &filter_shape, TensorType_FLOAT32, /*buffer=*/kNoBuffer));

    const std::vector<int32_t> dequantize_filter_inputs = {
        tensor_index_int8_filter};
    const std::vector<int32_t> dequantize_filter_outputs{tensor_index_filter};
    operators.emplace_back(CreateOperatorDirect(
        builder, /*opcode_index=*/kOpCodeIndexDequantize,
        &dequantize_filter_inputs, &dequantize_filter_outputs));

    assert(tensor_index_filter + 1 == tensors.size());

    if (UseBias()) {
      std::vector<float> bias_data(OutputChannels());
      std::generate(bias_data.begin(), bias_data.end(), f32rng);

      const int buffer_index_bias = buffers.size();
      buffers.emplace_back(CreateBuffer(
          builder, builder.CreateVector(
                       reinterpret_cast<const uint8_t*>(bias_data.data()),
                       sizeof(float) * bias_data.size())));

      tensors.emplace_back(CreateTensorDirect(builder, &bias_shape,
                                              TensorType_FLOAT32,
                                              /*buffer=*/buffer_index_bias));
    }
  } else {
    std::vector<float> filter_data(OutputChannels() * KernelHeight() *
                                   KernelWidth() * InputChannels());

    std::generate(filter_data.begin(), filter_data.end(), f32rng);

    const int buffer_index_filter = buffers.size();
    buffers.emplace_back(CreateBuffer(
        builder, builder.CreateVector(
                     reinterpret_cast<const uint8_t*>(filter_data.data()),
                     sizeof(float) * filter_data.size())));

    if (SparseWeights()) {
      const int dims_count = filter_shape.size();
      std::vector<flatbuffers::Offset<DimensionMetadata>> dim_metadata(
          dims_count);
      std::vector<int> traversal_order(dims_count);
      for (int dim = 0; dim < dims_count; dim++) {
        traversal_order[dim] = dim;
        dim_metadata[dim] = CreateDimensionMetadata(
            builder, DimensionType_DENSE, filter_shape[dim]);
      }
      flatbuffers::Offset<SparsityParameters> sparsity_parameters =
          CreateSparsityParameters(
              builder, builder.CreateVector(traversal_order),
              /*block_map=*/0, builder.CreateVector(dim_metadata));
      const int tensor_index_filter_sparse = tensors.size();
      tensors.emplace_back(CreateTensorDirect(
          builder, &filter_shape, TensorType_FLOAT32,
          /*buffer=*/buffer_index_filter, /*name=*/nullptr, /*quantization=*/0,
          /*is_variable=*/false, /*sparsity=*/sparsity_parameters));

      const int opcode_index_densify = operator_codes.size();
      operator_codes.emplace_back(
          CreateOperatorCode(builder, BuiltinOperator_DENSIFY));

      const int future_tensor_index_filter = tensors.size();
      const std::vector<int32_t> densify_filter_inputs = {
          tensor_index_filter_sparse};
      const std::vector<int32_t> densify_filter_outputs = {
          future_tensor_index_filter};
      operators.emplace_back(CreateOperatorDirect(
          builder, /*opcode_index=*/opcode_index_densify,
          &densify_filter_inputs, &densify_filter_outputs));

      // The dense filter tensor is just about to be added.
      assert(future_tensor_index_filter == tensors.size());
    }

    tensors.emplace_back(CreateTensorDirect(
        builder, &filter_shape, TensorType_FLOAT32,
        /*buffer=*/SparseWeights() ? kNoBuffer : buffer_index_filter));

    if (UseBias()) {
      std::vector<float> bias_data(OutputChannels());
      std::generate(bias_data.begin(), bias_data.end(), f32rng);

      const int buffer_index_bias = buffers.size();
      buffers.emplace_back(CreateBuffer(
          builder, builder.CreateVector(
                       reinterpret_cast<const uint8_t*>(bias_data.data()),
                       sizeof(float) * bias_data.size())));

      tensors.emplace_back(CreateTensorDirect(builder, &bias_shape,
                                              TensorType_FLOAT32,
                                              /*buffer=*/buffer_index_bias));
    }
  }

  const int top_tensor = tensors.size() - 1;
  const int tensor_index_filter = UseBias() ? top_tensor - 1 : top_tensor;

  const int tensor_index_input = tensors.size();
  tensors.emplace_back(
      CreateTensorDirect(builder, &input_shape, TensorType_FLOAT32));

  std::vector<int32_t> op_inputs = {tensor_index_output_shape,
                                    tensor_index_filter, tensor_index_input};
  if (UseBias()) {
    const int tensor_index_bias = top_tensor;
    op_inputs.push_back(tensor_index_bias);
  }

  const int tensor_index_output = tensors.size();
  tensors.emplace_back(
      CreateTensorDirect(builder, &output_shape, TensorType_FLOAT32));

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
      builder.CreateString("TransposeConv model");

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
