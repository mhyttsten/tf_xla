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
class MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSutilsDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSutilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSutilsDTcc() {
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
#include "tensorflow/lite/delegates/hexagon/utils.h"

#include <vector>

#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace {

bool IsActivationReluOrNone(TfLiteFusedActivation activation) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSutilsDTcc mht_0(mht_0_v, 196, "", "./tensorflow/lite/delegates/hexagon/utils.cc", "IsActivationReluOrNone");

  return (activation == kTfLiteActRelu || activation == kTfLiteActRelu6 ||
          activation == kTfLiteActReluN1To1 || activation == kTfLiteActNone);
}

bool TensorTypeMatch(int tensor_id, TfLiteContext* context,
                     TfLiteType tensor_type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSutilsDTcc mht_1(mht_1_v, 205, "", "./tensorflow/lite/delegates/hexagon/utils.cc", "TensorTypeMatch");

  const auto& tensor = context->tensors[tensor_id];
  return tensor.type == tensor_type;
}

// For each input tensor i, checks if the type matches one of the possibilities
// in per_input_possible_types[i].
bool InputsWithCorrectTypes(
    const TfLiteNode* node, TfLiteContext* context,
    const std::vector<std::vector<TfLiteType>>& per_input_possible_types) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSutilsDTcc mht_2(mht_2_v, 217, "", "./tensorflow/lite/delegates/hexagon/utils.cc", "InputsWithCorrectTypes");

  if (node->inputs->size != per_input_possible_types.size()) return false;
  for (int i = 0; i < per_input_possible_types.size(); ++i) {
    // Skip optional tensor.
    if (node->inputs->data[i] == -1) continue;
    bool type_found = false;
    for (auto possible_type : per_input_possible_types[i]) {
      if (TensorTypeMatch(node->inputs->data[i], context, possible_type)) {
        type_found = true;
        break;
      }
    }
    if (!type_found) return false;
  }
  return true;
}

}  // namespace

TfLiteStatus Get4DShape(unsigned int* batch_size, unsigned int* height_size,
                        unsigned int* width_size, unsigned int* depth_size,
                        TfLiteIntArray* dims) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSutilsDTcc mht_3(mht_3_v, 241, "", "./tensorflow/lite/delegates/hexagon/utils.cc", "Get4DShape");

  if (dims->size > 4) return kTfLiteError;
  unsigned int* dim[] = {batch_size, height_size, width_size, depth_size};
  for (int i = 0; i < 4; ++i) *(dim[i]) = 1;
  for (int i = 4 - dims->size; i < 4; ++i) {
    *dim[i] = dims->data[i - (4 - dims->size)];
  }
  return kTfLiteOk;
}

// We maintain an op-version allowlist here to ensure we don't accept unintended
// ops.
bool CheckOpVersion(const TfLiteRegistration* registration) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSutilsDTcc mht_4(mht_4_v, 256, "", "./tensorflow/lite/delegates/hexagon/utils.cc", "CheckOpVersion");

  switch (registration->builtin_code) {
    case kTfLiteBuiltinAdd:
    case kTfLiteBuiltinArgMax:
    case kTfLiteBuiltinArgMin:
    case kTfLiteBuiltinAveragePool2d:
    case kTfLiteBuiltinConcatenation:
    case kTfLiteBuiltinL2Normalization:
    case kTfLiteBuiltinLogistic:
    case kTfLiteBuiltinMaximum:
    case kTfLiteBuiltinMaxPool2d:
    case kTfLiteBuiltinMean:
    case kTfLiteBuiltinMinimum:
    case kTfLiteBuiltinMirrorPad:
    case kTfLiteBuiltinMul:
    case kTfLiteBuiltinPack:
    case kTfLiteBuiltinPad:
    case kTfLiteBuiltinQuantize:
    case kTfLiteBuiltinRelu6:
    case kTfLiteBuiltinSlice:
    case kTfLiteBuiltinSoftmax:
    case kTfLiteBuiltinSpaceToDepth:
    case kTfLiteBuiltinDepthToSpace:
    case kTfLiteBuiltinSplit:
    case kTfLiteBuiltinStridedSlice:
    case kTfLiteBuiltinSub:
    case kTfLiteBuiltinTanh:
    case kTfLiteBuiltinTranspose:
      return registration->version <= 2;
    case kTfLiteBuiltinSquaredDifference:
    case kTfLiteBuiltinRelu:
    case kTfLiteBuiltinRsqrt:
      return registration->version == 2;
    case kTfLiteBuiltinConv2d:
    case kTfLiteBuiltinDepthwiseConv2d:
    case kTfLiteBuiltinResizeBilinear:
    case kTfLiteBuiltinResizeNearestNeighbor:
    case kTfLiteBuiltinTransposeConv:
      return registration->version <= 3;
    case kTfLiteBuiltinFullyConnected:
      return registration->version <= 4;
    default:
      return registration->version == 1;
  }
}

bool IsNodeSupportedByHexagon(const TfLiteRegistration* registration,
                              const TfLiteNode* node, TfLiteContext* context) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPShexagonPSutilsDTcc mht_5(mht_5_v, 306, "", "./tensorflow/lite/delegates/hexagon/utils.cc", "IsNodeSupportedByHexagon");

  // Ensure all inputs & outputs have dim <= 4.
  int tensor_id;
  for (int i = 0; i < node->inputs->size; ++i) {
    tensor_id = node->inputs->data[i];
    // Skip optional tensors. Builders should handle optional tensors
    // not available.
    if (tensor_id == -1) continue;
    const auto& tensor = context->tensors[tensor_id];
    if (tensor.dims->size > 4) return false;
  }
  for (int i = 0; i < node->outputs->size; ++i) {
    tensor_id = node->outputs->data[i];
    const auto& tensor = context->tensors[tensor_id];
    if (tensor.dims->size > 4) return false;
  }

  if (!CheckOpVersion(registration)) return false;

  switch (registration->builtin_code) {
    case kTfLiteBuiltinAdd: {
      if (!InputsWithCorrectTypes(
              node, context,
              {{kTfLiteUInt8, kTfLiteInt8}, {kTfLiteUInt8, kTfLiteInt8}}))
        return false;
      const TfLiteAddParams* add_params =
          reinterpret_cast<const TfLiteAddParams*>(node->builtin_data);
      return IsActivationReluOrNone(add_params->activation);
    }
    case kTfLiteBuiltinMul: {
      if (!InputsWithCorrectTypes(
              node, context,
              {{kTfLiteUInt8, kTfLiteInt8}, {kTfLiteUInt8, kTfLiteInt8}}))
        return false;
      const TfLiteMulParams* mul_params =
          reinterpret_cast<const TfLiteMulParams*>(node->builtin_data);
      // TODO(b/129276536): Add support for activation on Mul node.
      return IsActivationReluOrNone(mul_params->activation);
    }
    case kTfLiteBuiltinSub: {
      if (!InputsWithCorrectTypes(
              node, context,
              {{kTfLiteUInt8, kTfLiteInt8}, {kTfLiteUInt8, kTfLiteInt8}}))
        return false;
      const TfLiteSubParams* sub_params =
          reinterpret_cast<const TfLiteSubParams*>(node->builtin_data);
      return IsActivationReluOrNone(sub_params->activation);
    }
    case kTfLiteBuiltinSum:
      // TODO(b/139277813): Enable these when they pass unit tests. These seem
      // to recompute the output min/max instead of taking them as inputs, which
      // causes an unexpected shift in dequantized values.
      return false;
    case kTfLiteBuiltinMean: {
      return InputsWithCorrectTypes(
                 node, context,
                 {{kTfLiteUInt8, kTfLiteInt8}, {kTfLiteInt32}}) &&
             IsConstantTensor(GetInput(context, node, 1));
    }
    case kTfLiteBuiltinMirrorPad: {
      if (!InputsWithCorrectTypes(
              node, context, {{kTfLiteUInt8, kTfLiteInt8}, {kTfLiteInt32}}) ||
          !IsConstantTensor(GetInput(context, node, 1)))
        return false;
      const TfLiteMirrorPaddingParams* params =
          reinterpret_cast<const TfLiteMirrorPaddingParams*>(
              node->builtin_data);
      return params->mode == kTfLiteMirrorPaddingReflect ||
             params->mode == kTfLiteMirrorPaddingSymmetric;
    }
    case kTfLiteBuiltinPad: {
      // TODO(b/139277813): Currently we only support padding with the default
      // of 0. Add support for user-defined constant if required.
      return (
          node->inputs->size == 2 &&
          InputsWithCorrectTypes(
              node, context, {{kTfLiteUInt8, kTfLiteInt8}, {kTfLiteInt32}}) &&
          IsConstantTensor(GetInput(context, node, 1)));
    }
    case kTfLiteBuiltinFullyConnected: {
      if (!InputsWithCorrectTypes(node, context,
                                  {{kTfLiteUInt8, kTfLiteInt8},
                                   {kTfLiteUInt8, kTfLiteInt8},
                                   {kTfLiteInt32, kTfLiteNoType}})) {
        return false;
      }

      bool bias_const_or_no_bias = true;
      if (node->inputs->data[2] != -1) {
        const auto& bias_tensor = context->tensors[node->inputs->data[2]];
        bias_const_or_no_bias = bias_tensor.allocation_type == kTfLiteMmapRo;
      }

      const TfLiteFullyConnectedParams* matmul_params =
          reinterpret_cast<const TfLiteFullyConnectedParams*>(
              node->builtin_data);
      return (bias_const_or_no_bias &&
              IsActivationReluOrNone(matmul_params->activation) &&
              matmul_params->keep_num_dims == false &&
              matmul_params->weights_format ==
                  kTfLiteFullyConnectedWeightsFormatDefault);
    }
    case kTfLiteBuiltinConcatenation: {
      // All concatenated tensors must be 8-bit.
      for (int i = 0; i < node->inputs->size; ++i) {
        if (!TensorTypeMatch(node->inputs->data[i], context, kTfLiteUInt8) &&
            !TensorTypeMatch(node->inputs->data[i], context, kTfLiteInt8))
          return false;
      }
      return true;
    }
    case kTfLiteBuiltinMaxPool2d: {
      if (!InputsWithCorrectTypes(node, context, {{kTfLiteUInt8, kTfLiteInt8}}))
        return false;
      // TODO(b/129276536): Add support for activation here.
      const TfLitePoolParams* pool_params =
          reinterpret_cast<const TfLitePoolParams*>(node->builtin_data);
      // Disable max pool on delegate with activation SAME when filter is > 12.
      if (pool_params->padding == kTfLitePaddingSame &&
          (pool_params->filter_height >= 13 ||
           pool_params->filter_width >= 13)) {
        return false;
      }
      return pool_params->activation == kTfLiteActNone;
    }
    case kTfLiteBuiltinAveragePool2d: {
      if (!InputsWithCorrectTypes(node, context, {{kTfLiteUInt8, kTfLiteInt8}}))
        return false;
      const TfLitePoolParams* pool_params =
          reinterpret_cast<const TfLitePoolParams*>(node->builtin_data);
      return (node->inputs->size == 1 &&
              pool_params->activation == kTfLiteActNone);
    }
    case kTfLiteBuiltinTransposeConv: {
      if (NumInputs(node) == 3) {
        if (!InputsWithCorrectTypes(node, context,
                                    {{kTfLiteInt32},
                                     {kTfLiteUInt8, kTfLiteInt8},
                                     {kTfLiteUInt8, kTfLiteInt8}}))
          return false;
      } else if (NumInputs(node) == 4) {
        if (!InputsWithCorrectTypes(node, context,
                                    {{kTfLiteInt32},
                                     {kTfLiteUInt8, kTfLiteInt8},
                                     {kTfLiteUInt8, kTfLiteInt8},
                                     {kTfLiteInt32}}))
          return false;
      } else {
        return false;
      }
      const TfLiteTransposeConvParams* params =
          reinterpret_cast<const TfLiteTransposeConvParams*>(
              node->builtin_data);
      return (params->stride_height <= 3 && params->stride_width <= 3 &&
              (params->padding == kTfLitePaddingSame ||
               params->padding == kTfLitePaddingValid));
    }
    case kTfLiteBuiltinConv2d: {
      if (!InputsWithCorrectTypes(node, context,
                                  {{kTfLiteUInt8, kTfLiteInt8},
                                   {kTfLiteUInt8, kTfLiteInt8},
                                   {kTfLiteInt32}}))
        return false;
      const TfLiteConvParams* conv_params =
          reinterpret_cast<const TfLiteConvParams*>(node->builtin_data);
      return (IsActivationReluOrNone(conv_params->activation) &&
              conv_params->stride_height <= 3 &&
              conv_params->stride_width <= 3 &&
              conv_params->dilation_height_factor == 1 &&
              conv_params->dilation_width_factor == 1);
    }
    case kTfLiteBuiltinDepthwiseConv2d: {
      if (!InputsWithCorrectTypes(node, context,
                                  {{kTfLiteUInt8, kTfLiteInt8},
                                   {kTfLiteUInt8, kTfLiteInt8},
                                   {kTfLiteInt32}}))
        return false;

      // Check dilation.
      const TfLiteDepthwiseConvParams* conv_params =
          reinterpret_cast<const TfLiteDepthwiseConvParams*>(
              node->builtin_data);
      const bool dilation = conv_params->dilation_height_factor != 1 ||
                            conv_params->dilation_width_factor != 1;
      if (dilation) {
        // We only support dilations when stride == 1.
        if (conv_params->stride_height != 1 || conv_params->stride_width != 1)
          return false;
      }

      // We currently only support depth_multiplier > 1 when:
      // 1. dilation_factor == 1 AND
      // 2. input_depth == 1
      // TODO(b/143759564): Add support for general case.
      const auto& input = context->tensors[node->inputs->data[0]];
      const bool supported_depth_multiplier =
          conv_params->depth_multiplier == 1 ||
          (!dilation && input.dims->size == 4 && input.dims->data[3] == 1);

      return (IsActivationReluOrNone(conv_params->activation) &&
              conv_params->stride_height <= 3 &&
              conv_params->stride_width <= 3 && supported_depth_multiplier);
    }
    case kTfLiteBuiltinReshape: {
      if (node->inputs->size > 2 ||
          (!TensorTypeMatch(node->inputs->data[0], context, kTfLiteUInt8) &&
           !TensorTypeMatch(node->inputs->data[0], context, kTfLiteInt8)))
        return false;
      return true;
    }
    case kTfLiteBuiltinSoftmax: {
      return (
          InputsWithCorrectTypes(node, context, {{kTfLiteUInt8, kTfLiteInt8}}));
    }
    case kTfLiteBuiltinHardSwish:
    case kTfLiteBuiltinRelu:
    case kTfLiteBuiltinRelu6:
    case kTfLiteBuiltinTanh:
    case kTfLiteBuiltinLogistic: {
      return InputsWithCorrectTypes(node, context,
                                    {{kTfLiteUInt8, kTfLiteInt8}});
    }
    case kTfLiteBuiltinResizeNearestNeighbor: {
      return InputsWithCorrectTypes(
                 node, context,
                 {{kTfLiteUInt8, kTfLiteInt8}, {kTfLiteInt32}}) &&
             IsConstantTensor(GetInput(context, node, 1));
    }
    case kTfLiteBuiltinL2Normalization: {
      if (!InputsWithCorrectTypes(node, context, {{kTfLiteUInt8, kTfLiteInt8}}))
        return false;
      const TfLiteL2NormParams* norm_params =
          reinterpret_cast<const TfLiteL2NormParams*>(node->builtin_data);
      return (norm_params->activation == kTfLiteActNone);
    }
    case kTfLiteBuiltinArgMax:
    case kTfLiteBuiltinArgMin:
      return InputsWithCorrectTypes(
          node, context, {{kTfLiteUInt8, kTfLiteInt8}, {kTfLiteInt32}});
    case kTfLiteBuiltinSplit: {
      if (!InputsWithCorrectTypes(
              node, context, {{kTfLiteInt32}, {kTfLiteUInt8, kTfLiteInt8}}))
        return false;
      const auto& input_tensor = context->tensors[node->inputs->data[1]];
      const bool is_four_dim_or_less = input_tensor.dims->size < 5;
      // We need splitting axis to be constant, so Hexagon knows output
      // shapes.
      return is_four_dim_or_less &&
             IsConstantTensor(GetInput(context, node, 0));
    }
    case kTfLiteBuiltinResizeBilinear: {
      if (!InputsWithCorrectTypes(
              node, context, {{kTfLiteUInt8, kTfLiteInt8}, {kTfLiteInt32}}) ||
          !IsConstantTensor(GetInput(context, node, 1))) {
        return false;
      }
      const auto& size_tensor = context->tensors[node->inputs->data[1]];
      return NumElements(&size_tensor) == 2;
    }
    case kTfLiteBuiltinNeg: {
      return InputsWithCorrectTypes(node, context,
                                    {{kTfLiteUInt8, kTfLiteInt8}});
    }
    case kTfLiteBuiltinTranspose: {
      return InputsWithCorrectTypes(
          node, context, {{kTfLiteUInt8, kTfLiteInt8}, {kTfLiteInt32}});
    }
    case kTfLiteBuiltinSpaceToDepth:
    case kTfLiteBuiltinDepthToSpace: {
      return InputsWithCorrectTypes(node, context,
                                    {{kTfLiteUInt8, kTfLiteInt8}});
    }
    case kTfLiteBuiltinQuantize: {
      return InputsWithCorrectTypes(node, context,
                                    {{kTfLiteUInt8, kTfLiteInt8}});
    }
    case kTfLiteBuiltinMinimum: {
      return InputsWithCorrectTypes(
          node, context,
          {{kTfLiteUInt8, kTfLiteInt8}, {kTfLiteUInt8, kTfLiteInt8}});
    }
    case kTfLiteBuiltinMaximum: {
      return InputsWithCorrectTypes(
          node, context,
          {{kTfLiteUInt8, kTfLiteInt8}, {kTfLiteUInt8, kTfLiteInt8}});
    }
    case kTfLiteBuiltinSlice: {
      const auto& begins_tensor = context->tensors[node->inputs->data[1]];
      const auto& sizes_tensor = context->tensors[node->inputs->data[2]];
      if (!IsConstantTensor(&begins_tensor) || !IsConstantTensor(&sizes_tensor))
        return false;
      return InputsWithCorrectTypes(node, context,
                                    {{kTfLiteUInt8, kTfLiteInt8},
                                     {kTfLiteInt32, kTfLiteInt64},
                                     {kTfLiteInt32, kTfLiteInt64}});
    }
    case kTfLiteBuiltinPack: {
      // All tensors must be 8-bit.
      for (int i = 0; i < node->inputs->size; ++i) {
        if (!TensorTypeMatch(node->inputs->data[i], context, kTfLiteUInt8) &&
            !TensorTypeMatch(node->inputs->data[i], context, kTfLiteInt8))
          return false;
      }
      return true;
    }
    case kTfLiteBuiltinStridedSlice: {
      if (!InputsWithCorrectTypes(node, context,
                                  {{kTfLiteUInt8, kTfLiteInt8},
                                   {kTfLiteInt32},
                                   {kTfLiteInt32},
                                   {kTfLiteInt32}}))
        return false;
      const auto& begins_tensor = context->tensors[node->inputs->data[1]];
      const auto& ends_tensor = context->tensors[node->inputs->data[2]];
      const auto& step_tensor = context->tensors[node->inputs->data[3]];
      if (!IsConstantTensor(&begins_tensor) ||
          !IsConstantTensor(&ends_tensor) || !IsConstantTensor(&step_tensor))
        return false;
      const TfLiteStridedSliceParams* params =
          reinterpret_cast<const TfLiteStridedSliceParams*>(node->builtin_data);
      // Hexagon doesn't support ellipsis/new-axis masks.
      return (params->ellipsis_mask == 0 && params->new_axis_mask == 0);
    }
    case kTfLiteBuiltinSquaredDifference: {
      return InputsWithCorrectTypes(node, context,
                                    {{kTfLiteInt8}, {kTfLiteInt8}});
    }
    case kTfLiteBuiltinRsqrt: {
      return InputsWithCorrectTypes(node, context, {{kTfLiteInt8}});
    }
    default:
      return false;
  }
  return false;
}

}  // namespace tflite
