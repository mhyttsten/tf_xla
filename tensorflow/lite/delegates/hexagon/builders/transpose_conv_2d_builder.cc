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
class MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStranspose_conv_2d_builderDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStranspose_conv_2d_builderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStranspose_conv_2d_builderDTcc() {
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
#include "tensorflow/lite/delegates/hexagon/builders/transpose_conv_2d_builder.h"

#include <stdint.h>

#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/delegates/hexagon/hexagon_nn/hexagon_nn.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"

namespace tflite {
namespace delegates {
namespace hexagon {
namespace {

constexpr uint8_t k8BitSignFlipConstant = 0x80;
// 1/1024 ~ 0.0009766 is a restriction set by Hexagon's kernels.
// TODO(b/151103818): Figure out a way to retrieve this constant reliably.
constexpr float kHexagonMinRelativeScale = 0.0009766f;

}  // namespace

TfLiteStatus TransposeConv2dOpBuilder::PopulateSubGraph(
    const TfLiteIntArray* inputs, const TfLiteIntArray* outputs,
    TfLiteContext* context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStranspose_conv_2d_builderDTcc mht_0(mht_0_v, 210, "", "./tensorflow/lite/delegates/hexagon/builders/transpose_conv_2d_builder.cc", "TransposeConv2dOpBuilder::PopulateSubGraph");

  // DATA TENSOR.
  int tensor_id = inputs->data[2];
  const auto& data_tensor = context->tensors[tensor_id];
  AddInput(graph_builder_->GetHexagonTensorId(tensor_id));

  // WEIGHTS.
  tensor_id = inputs->data[1];
  const auto& weights_tensor = context->tensors[tensor_id];
  if (weights_tensor.allocation_type != kTfLiteMmapRo) {
    context->ReportError(
        context, "Weights tensor doesn't have correct allocation type: %s",
        weights_tensor.name);
    return kTfLiteError;
  }
  int filter_batch_size, filter_height_size, filter_width_size,
      filter_depth_size;
  GetDims(&filter_batch_size, &filter_height_size, &filter_width_size,
          &filter_depth_size, weights_tensor.dims);
  // Weights tensor could be int8 even for per-tensor quantization.
  // Therefore, we look at the number of scale values to check if it is
  // per-channel quantized.
  TfLiteAffineQuantization* weights_quant_params =
      reinterpret_cast<TfLiteAffineQuantization*>(
          weights_tensor.quantization.params);
  const bool is_per_channel_quant = weights_quant_params->scale->size > 1;
  AddInput(graph_builder_->GetHexagonTensorId(tensor_id));

  // Handle weights quantization.
  float weights_min = 0;
  float weights_max = 0;
  if (is_per_channel_quant) {
    ProcessPerChannelQuantizedWeights(weights_tensor, context, &weights_min,
                                      &weights_max, graph_builder_,
                                      &per_channel_quant_);
  } else {
    TF_LITE_ENSURE_STATUS(ComputeMinAndMaxQuantValues(
        weights_tensor, &weights_min, &weights_max));
  }
  auto* weights_min_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&weights_min), sizeof(weights_min));
  auto* weights_max_const = graph_builder_->AddConstNodeWithData(
      kScalarShape, reinterpret_cast<char*>(&weights_max), sizeof(weights_max));

  // Min/max inputs for data & weights tensors.
  TF_LITE_ENSURE_STATUS(ComputeAndAddMinAndMax(context, data_tensor));
  AddInput(TensorID(weights_min_const->GetID(), 0));
  AddInput(TensorID(weights_max_const->GetID(), 0));

  // Output dims are required to compute padding.
  int output_batch_size, output_height_size, output_width_size,
      output_depth_size;
  GetDims(&output_batch_size, &output_height_size, &output_width_size,
          &output_depth_size, context->tensors[outputs->data[0]].dims);

  // PADDING & STRIDE.
  // Hexagon TransposeConv requires an explicit padding tensor. So we compute
  // the same using stride, input & output info.
  const TfLiteTransposeConvParams* params =
      reinterpret_cast<const TfLiteTransposeConvParams*>(builtin_data_);
  int unused_output_height, unused_output_width;
  TfLitePaddingValues padding = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width, 1, 1, output_height_size,
      output_width_size, filter_height_size, filter_width_size, params->padding,
      &unused_output_height, &unused_output_width);
  std::vector<int> padding_tensor = {padding.height, padding.height,
                                     padding.width, padding.width};
  std::vector<int> padding_tensor_shape = {1, 1, 2, 2};
  auto* padding_const = graph_builder_->AddConstNodeWithData(
      padding_tensor_shape.data(),
      reinterpret_cast<char*>(padding_tensor.data()), (sizeof(int) * 4));
  AddInput(TensorID(padding_const->GetID(), 0));

  // Stride shape.
  int stride_height = params->stride_height;
  int stride_width = params->stride_width;
  static int dummy = 0;
  stride_shape_ = {1, stride_height, stride_width, 1};
  auto* stride_node = graph_builder_->AddConstNodeWithData(
      stride_shape_.data(), reinterpret_cast<char*>(&dummy), sizeof(dummy));
  AddInput(TensorID(stride_node->GetID(), 0));

  // BIAS.
  const bool has_bias = inputs->size == 4;
  OpBuilder* bias_const = nullptr;
  OpBuilder* bias_min_const = nullptr;
  OpBuilder* bias_max_const = nullptr;
  if (!has_bias) {
    // If the TFLite node does not have a bias, we simply feed in 0s.
    std::vector<int> bias_data(output_depth_size, 0);
    bias_shape_ = {1, 1, 1, output_depth_size};
    bias_const = graph_builder_->AddConstNodeWithData(
        bias_shape_.data(), reinterpret_cast<char*>(bias_data.data()),
        sizeof(bias_data[0]) * bias_data.size());
    float zero_bound = 0;
    bias_min_const = graph_builder_->AddConstNodeWithData(
        kScalarShape, reinterpret_cast<char*>(&zero_bound), sizeof(zero_bound));
    bias_max_const = graph_builder_->AddConstNodeWithData(
        kScalarShape, reinterpret_cast<char*>(&zero_bound), sizeof(zero_bound));
  } else {
    const auto& bias_tensor = context->tensors[inputs->data[3]];
    if (bias_tensor.allocation_type != kTfLiteMmapRo) {
      TF_LITE_KERNEL_LOG(context,
                         "Bias tensor doesn't have correct allocation type: %s",
                         bias_tensor.name);
      return kTfLiteError;
    }
    float bias_min = 0;
    float bias_max = 0;
    if (per_channel_quant_.channel_scales_node != nullptr) {
      ProcessPerChannelQuantizedBias(
          data_tensor, bias_tensor, inputs->data[3], context, &bias_min,
          &bias_max, graph_builder_, &per_channel_quant_, &bias_const);
    } else {
      bias_const =
          graph_builder_->AddConstNodeWithData(inputs->data[3], bias_tensor);
      TF_LITE_ENSURE_STATUS(
          ComputeMinAndMaxQuantValues(bias_tensor, &bias_min, &bias_max));
    }

    bias_min_const = graph_builder_->AddConstNodeWithData(
        kScalarShape, reinterpret_cast<char*>(&bias_min), sizeof(bias_min));
    bias_max_const = graph_builder_->AddConstNodeWithData(
        kScalarShape, reinterpret_cast<char*>(&bias_max), sizeof(bias_max));
  }
  AddInput(TensorID(bias_const->GetID(), 0));
  AddInput(TensorID(bias_min_const->GetID(), 0));
  AddInput(TensorID(bias_max_const->GetID(), 0));

  // Output quantization.
  TF_LITE_ENSURE_STATUS(
      ComputeAndAddMinAndMax(context, context->tensors[outputs->data[0]]));

  // Channel scales, if this op is per-channel quantized.
  if (per_channel_quant_.channel_scales_node != nullptr) {
    AddInput(TensorID(per_channel_quant_.channel_scales_node->GetID(), 0));
  }

  // Hexagon outputs for this node.
  node_output_ = AddOutput(sizeof(uint8_t), 4,
                           {output_batch_size, output_height_size,
                            output_width_size, output_depth_size});
  AddOutput(sizeof(float), 4, kScalarShape);
  AddOutput(sizeof(float), 4, kScalarShape);

  return kTfLiteOk;
}

TfLiteStatus TransposeConv2dOpBuilder::RegisterOutputs(
    const TfLiteIntArray* outputs, TfLiteContext* context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStranspose_conv_2d_builderDTcc mht_1(mht_1_v, 362, "", "./tensorflow/lite/delegates/hexagon/builders/transpose_conv_2d_builder.cc", "TransposeConv2dOpBuilder::RegisterOutputs");

  // Should be only 1 output.
  graph_builder_->AddTensorWithID(outputs->data[0], node_output_.first,
                                  node_output_.second);
  return kTfLiteOk;
}

TransposeConv2dOpBuilder::~TransposeConv2dOpBuilder() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStranspose_conv_2d_builderDTcc mht_2(mht_2_v, 372, "", "./tensorflow/lite/delegates/hexagon/builders/transpose_conv_2d_builder.cc", "TransposeConv2dOpBuilder::~TransposeConv2dOpBuilder");
}

OpBuilder* CreateTransposeConv2DBuilder(GraphBuilder* graph_builder,
                                        int op_type) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSbuildersPStranspose_conv_2d_builderDTcc mht_3(mht_3_v, 378, "", "./tensorflow/lite/delegates/hexagon/builders/transpose_conv_2d_builder.cc", "CreateTransposeConv2DBuilder");

  return new TransposeConv2dOpBuilder(graph_builder, op_type);
}

}  // namespace hexagon
}  // namespace delegates
}  // namespace tflite
