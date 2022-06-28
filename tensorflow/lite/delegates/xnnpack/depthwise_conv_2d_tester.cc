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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSdepthwise_conv_2d_testerDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSdepthwise_conv_2d_testerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSdepthwise_conv_2d_testerDTcc() {
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

#include "tensorflow/lite/delegates/xnnpack/depthwise_conv_2d_tester.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "fp16.h"  // from @FP16
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/delegates/xnnpack/test_util.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/schema/schema_conversion_utils.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace tflite {
namespace xnnpack {

void DepthwiseConv2DTester::Test(TfLiteDelegate* delegate) const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSdepthwise_conv_2d_testerDTcc mht_0(mht_0_v, 209, "", "./tensorflow/lite/delegates/xnnpack/depthwise_conv_2d_tester.cc", "DepthwiseConv2DTester::Test");

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
  auto input_rng =
      std::bind(std::uniform_real_distribution<float>(), std::ref(rng));
  float* default_input_data = default_interpreter->typed_input_tensor<float>(0);
  std::generate(default_input_data,
                default_input_data + BatchSize() * InputHeight() *
                                         InputWidth() * InputChannels(),
                input_rng);

  float* delegate_input_data =
      delegate_interpreter->typed_input_tensor<float>(0);
  std::copy(default_input_data,
            default_input_data +
                BatchSize() * InputHeight() * InputWidth() * InputChannels(),
            delegate_input_data);

  ASSERT_EQ(default_interpreter->Invoke(), kTfLiteOk);
  ASSERT_EQ(delegate_interpreter->Invoke(), kTfLiteOk);

  float* default_output_data =
      default_interpreter->typed_output_tensor<float>(0);
  float* delegate_output_data =
      delegate_interpreter->typed_output_tensor<float>(0);

  for (int32_t i = 0; i < BatchSize(); i++) {
    for (int32_t y = 0; y < OutputHeight(); y++) {
      for (int32_t x = 0; x < OutputWidth(); x++) {
        for (int32_t c = 0; c < OutputChannels(); c++) {
          const int32_t index = ((i * OutputHeight() + y) * OutputWidth() + x) *
                                    OutputChannels() +
                                c;
          ASSERT_NEAR(default_output_data[index], delegate_output_data[index],
                      std::abs(default_output_data[index]) * 3.0e-6f)
              << "batch " << i << " / " << BatchSize() << ", y position " << y
              << " / " << OutputHeight() << ", x position " << x << " / "
              << OutputWidth() << ", channel " << c << " / "
              << OutputChannels();
        }
      }
    }
  }
}

std::vector<char> DepthwiseConv2DTester::CreateTfLiteModel() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSxnnpackPSdepthwise_conv_2d_testerDTcc mht_1(mht_1_v, 289, "", "./tensorflow/lite/delegates/xnnpack/depthwise_conv_2d_tester.cc", "DepthwiseConv2DTester::CreateTfLiteModel");

  std::random_device random_device;
  auto rng = std::mt19937(random_device());
  auto range_rng = std::bind(
      std::uniform_real_distribution<float>(-25.0f, 25.0f), std::ref(rng));

  flatbuffers::FlatBufferBuilder builder;
  std::vector<flatbuffers::Offset<OperatorCode>> operator_codes{
      {CreateOperatorCode(builder, BuiltinOperator_DEPTHWISE_CONV_2D)}};
  std::vector<flatbuffers::Offset<tflite::Operator>> operators;
  std::vector<flatbuffers::Offset<tflite::Buffer>> buffers{
      {CreateBuffer(builder, builder.CreateVector({}))}};

  const std::vector<int32_t> filter_shape = {1, KernelHeight(), KernelWidth(),
                                             OutputChannels()};
  const std::vector<int32_t> bias_shape = {OutputChannels()};
  std::vector<float> filter_scales;
  std::vector<int64_t> filter_zero_points;
  int32_t filter_quantized_dimension = 0;
  if (FP16Weights()) {
    operator_codes.emplace_back(
        CreateOperatorCode(builder, BuiltinOperator_DEQUANTIZE));

    std::vector<uint16_t> filter_data(KernelHeight() * KernelWidth() *
                                      OutputChannels());
    std::vector<uint16_t> bias_data(OutputChannels());
    for (int32_t ic = 0; ic < InputChannels(); ic++) {
      // Use the same range of all-positive or all-negative values to generate
      // all pixels within the same batch index & channel, but different ranges
      // for different channels or batches. This ensures that no catastrophic
      // cancellation occur, but test covers both positive and negative inputs.
      const float range = range_rng();
      auto value_rng =
          std::bind(fp16_ieee_from_fp32_value,
                    std::bind(std::uniform_real_distribution<float>(
                                  std::min(range, 0.0f), std::max(range, 0.0f)),
                              std::ref(rng)));
      for (int32_t m = 0; m < DepthMultiplier(); m++) {
        const int32_t oc = ic * DepthMultiplier() + m;
        bias_data[oc] = value_rng();
        for (int32_t y = 0; y < KernelHeight(); y++) {
          for (int32_t x = 0; x < KernelWidth(); x++) {
            const int32_t index =
                (y * KernelWidth() + x) * OutputChannels() + oc;
            filter_data[index] = value_rng();
          }
        }
      }
    }

    buffers.emplace_back(CreateBuffer(
        builder, builder.CreateVector(
                     reinterpret_cast<const uint8_t*>(filter_data.data()),
                     sizeof(uint16_t) * filter_data.size())));
    buffers.emplace_back(CreateBuffer(
        builder,
        builder.CreateVector(reinterpret_cast<const uint8_t*>(bias_data.data()),
                             sizeof(uint16_t) * bias_data.size())));

    const std::array<int32_t, 1> dequantize_filter_inputs{{0}};
    const std::array<int32_t, 1> dequantize_filter_outputs{{3}};
    operators.emplace_back(CreateOperator(
        builder, /*opcode_index=*/1,
        builder.CreateVector<int32_t>(dequantize_filter_inputs.data(),
                                      dequantize_filter_inputs.size()),
        builder.CreateVector<int32_t>(dequantize_filter_outputs.data(),
                                      dequantize_filter_outputs.size())));
    const std::array<int32_t, 1> dequantize_bias_inputs{{1}};
    const std::array<int32_t, 1> dequantize_bias_outputs{{4}};
    operators.emplace_back(CreateOperator(
        builder, /*opcode_index=*/1,
        builder.CreateVector<int32_t>(dequantize_bias_inputs.data(),
                                      dequantize_bias_inputs.size()),
        builder.CreateVector<int32_t>(dequantize_bias_outputs.data(),
                                      dequantize_bias_outputs.size())));
  } else {
    std::vector<float> filter_data(KernelHeight() * KernelWidth() *
                                   OutputChannels());
    std::vector<float> bias_data(OutputChannels());
    for (int32_t ic = 0; ic < InputChannels(); ic++) {
      // Use the same range of all-positive or all-negative values to generate
      // all pixels within the same batch index & channel, but different ranges
      // for different channels or batches. This ensures that no catastrophic
      // cancellation occur, but test covers both positive and negative inputs.
      const float range = range_rng();
      auto value_rng =
          std::bind(std::uniform_real_distribution<float>(
                        std::min(range, 0.0f), std::max(range, 0.0f)),
                    std::ref(rng));
      for (int32_t m = 0; m < DepthMultiplier(); m++) {
        const int32_t oc = ic * DepthMultiplier() + m;
        bias_data[oc] = value_rng();
        for (int32_t y = 0; y < KernelHeight(); y++) {
          for (int32_t x = 0; x < KernelWidth(); x++) {
            const int32_t index =
                (y * KernelWidth() + x) * OutputChannels() + oc;
            filter_data[index] = value_rng();
          }
        }
      }
    }

    if (INT8Weights() || INT8ChannelWiseWeights()) {
      std::vector<int8_t> quantized_filter_data(filter_data.size());
      if (INT8Weights()) {
        filter_scales.resize(1, GetInt8QuantizationScale(filter_data));
        filter_zero_points.resize(1, 0);
        std::transform(filter_data.begin(), filter_data.end(),
                       quantized_filter_data.begin(),
                       std::bind(QuantizeInt8, std::placeholders::_1, 0,
                                 filter_scales[0]));
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
      buffers.emplace_back(CreateBuffer(
          builder,
          builder.CreateVector(
              reinterpret_cast<const uint8_t*>(quantized_filter_data.data()),
              sizeof(int8_t) * quantized_filter_data.size())));
      operator_codes.emplace_back(
          CreateOperatorCode(builder, BuiltinOperator_DEQUANTIZE));
      const std::array<int32_t, 1> dequantize_filter_inputs{{0}};
      const std::array<int32_t, 1> dequantize_filter_outputs{{2}};
      operators.emplace_back(CreateOperator(
          builder, /*opcode_index=*/1,
          builder.CreateVector<int32_t>(dequantize_filter_inputs.data(),
                                        dequantize_filter_inputs.size()),
          builder.CreateVector<int32_t>(dequantize_filter_outputs.data(),
                                        dequantize_filter_outputs.size())));
    } else {
      buffers.emplace_back(CreateBuffer(
          builder, builder.CreateVector(
                       reinterpret_cast<const uint8_t*>(filter_data.data()),
                       sizeof(float) * filter_data.size())));

      if (SparseWeights()) {
        operator_codes.emplace_back(
            CreateOperatorCode(builder, BuiltinOperator_DENSIFY));
        const std::array<int32_t, 1> densify_filter_inputs{{0}};
        const std::array<int32_t, 1> densify_filter_outputs{{2}};
        operators.emplace_back(CreateOperator(
            builder, /*opcode_index=*/1,
            builder.CreateVector<int32_t>(densify_filter_inputs.data(),
                                          densify_filter_inputs.size()),
            builder.CreateVector<int32_t>(densify_filter_outputs.data(),
                                          densify_filter_outputs.size())));
      }
    }

    // Bias is stored in FP32 even when filter is quantized to INT8
    buffers.emplace_back(CreateBuffer(
        builder,
        builder.CreateVector(reinterpret_cast<const uint8_t*>(bias_data.data()),
                             sizeof(float) * bias_data.size())));
  }

  const std::array<int32_t, 4> input_shape{
      {BatchSize(), InputHeight(), InputWidth(), InputChannels()}};
  const std::array<int32_t, 4> output_shape{
      {BatchSize(), OutputHeight(), OutputWidth(), OutputChannels()}};

  std::vector<flatbuffers::Offset<tflite::Tensor>> tensors;
  if (FP16Weights()) {
    tensors.emplace_back(CreateTensor(
        builder,
        builder.CreateVector<int32_t>(filter_shape.data(), filter_shape.size()),
        TensorType_FLOAT16, /*buffer=*/1));
    tensors.emplace_back(CreateTensor(
        builder,
        builder.CreateVector<int32_t>(bias_shape.data(), bias_shape.size()),
        TensorType_FLOAT16, /*buffer=*/2));
  } else if (INT8Weights() || INT8ChannelWiseWeights()) {
    tensors.emplace_back(CreateTensor(
        builder,
        builder.CreateVector<int32_t>(filter_shape.data(), filter_shape.size()),
        TensorType_INT8, /*buffer=*/1, /*name=*/0,
        CreateQuantizationParameters(
            builder, /*min=*/0, /*max=*/0,
            builder.CreateVector<float>(filter_scales),
            builder.CreateVector<int64_t>(filter_zero_points),
            /*details_type=*/QuantizationDetails_NONE,
            /*details=*/0, filter_quantized_dimension)));
  } else if (SparseWeights()) {
    // Sparse tensor in TFLite can be in different formats. Here we choose the
    // simplest configuration that
    //   1. all dimensions are dense,
    //   2. in-order traversal, and
    //   3. no block configuration.
    int dims_count = filter_shape.size();
    std::vector<flatbuffers::Offset<DimensionMetadata>> dim_metadata(
        dims_count);
    std::vector<int> traversal_order(dims_count);
    for (int i = 0; i < dims_count; i++) {
      traversal_order[i] = i;
      dim_metadata[i] = CreateDimensionMetadata(builder, DimensionType_DENSE,
                                                filter_shape[i]);
    }
    flatbuffers::Offset<SparsityParameters> sparsity_param =
        CreateSparsityParameters(builder, builder.CreateVector(traversal_order),
                                 0, builder.CreateVector(dim_metadata));
    tensors.emplace_back(CreateTensor(
        builder,
        builder.CreateVector<int32_t>(filter_shape.data(), filter_shape.size()),
        TensorType_FLOAT32, /*buffer=*/1, /*name=*/0, /*quantization=*/0,
        /*is_variable=*/false, /*sparsity=*/sparsity_param));
  }
  tensors.emplace_back(CreateTensor(
      builder,
      builder.CreateVector<int32_t>(input_shape.data(), input_shape.size()),
      TensorType_FLOAT32));
  tensors.emplace_back(CreateTensor(
      builder,
      builder.CreateVector<int32_t>(filter_shape.data(), filter_shape.size()),
      TensorType_FLOAT32,
      /*buffer=*/
      (FP16Weights() || INT8Weights() || INT8ChannelWiseWeights() ||
       SparseWeights())
          ? 0
          : 1));
  tensors.emplace_back(CreateTensor(
      builder,
      builder.CreateVector<int32_t>(bias_shape.data(), bias_shape.size()),
      TensorType_FLOAT32, /*buffer=*/FP16Weights() ? 0 : 2));
  tensors.emplace_back(CreateTensor(
      builder,
      builder.CreateVector<int32_t>(output_shape.data(), output_shape.size()),
      TensorType_FLOAT32));

  const std::array<int32_t, 3> op_inputs{
      {static_cast<int>(tensors.size()) - 4,
       static_cast<int>(tensors.size()) - 3,
       static_cast<int>(tensors.size()) - 2}};
  const std::array<int32_t, 1> op_outputs{
      {static_cast<int>(tensors.size()) - 1}};

  flatbuffers::Offset<DepthwiseConv2DOptions> depthwise_conv2d_options =
      CreateDepthwiseConv2DOptions(
          builder, Padding(), StrideWidth(), StrideHeight(), DepthMultiplier(),
          Activation(), DilationWidth(), DilationHeight());
  operators.emplace_back(CreateOperator(
      builder, /*opcode_index=*/0,
      builder.CreateVector<int32_t>(op_inputs.data(), op_inputs.size()),
      builder.CreateVector<int32_t>(op_outputs.data(), op_outputs.size()),
      BuiltinOptions_DepthwiseConv2DOptions, depthwise_conv2d_options.Union()));

  const std::array<int32_t, 1> subgraph_inputs{
      {static_cast<int>(tensors.size()) - 4}};
  const std::array<int32_t, 1> subgraph_outputs{
      {static_cast<int>(tensors.size()) - 1}};
  flatbuffers::Offset<SubGraph> subgraph = CreateSubGraph(
      builder, builder.CreateVector(tensors.data(), tensors.size()),
      builder.CreateVector<int32_t>(subgraph_inputs.data(),
                                    subgraph_inputs.size()),
      builder.CreateVector<int32_t>(subgraph_outputs.data(),
                                    subgraph_outputs.size()),
      builder.CreateVector(operators.data(), operators.size()));

  flatbuffers::Offset<flatbuffers::String> description =
      builder.CreateString("DepthwiseConv2D model");

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
