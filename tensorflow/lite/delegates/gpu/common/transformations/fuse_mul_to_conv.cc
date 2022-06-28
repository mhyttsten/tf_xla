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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSfuse_mul_to_convDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSfuse_mul_to_convDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSfuse_mul_to_convDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/transformations/fuse_mul_to_conv.h"

#include <any>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {
namespace {

class MergeConvolutionWithMul : public SequenceTransformation {
 public:
  int ExpectedSequenceLength() const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSfuse_mul_to_convDTcc mht_0(mht_0_v, 208, "", "./tensorflow/lite/delegates/gpu/common/transformations/fuse_mul_to_conv.cc", "ExpectedSequenceLength");
 return 2; }

  TransformResult ApplyToNodesSequence(const std::vector<Node*>& sequence,
                                       GraphFloat32* graph) final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSfuse_mul_to_convDTcc mht_1(mht_1_v, 214, "", "./tensorflow/lite/delegates/gpu/common/transformations/fuse_mul_to_conv.cc", "ApplyToNodesSequence");

    auto& conv_node = *sequence[0];
    if (graph->FindInputs(conv_node.id).size() != 1) {
      return {TransformStatus::DECLINED,
              "This fusion is only applicable to ops with one runtime input."};
    }

    auto& mul_node = *sequence[1];
    if (mul_node.operation.type != ToString(OperationType::MUL) ||
        !mul_node.operation.attributes.has_value()) {
      return {TransformStatus::SKIPPED, ""};
    }

    ElementwiseAttributes mul_attr =
        absl::any_cast<ElementwiseAttributes>(mul_node.operation.attributes);
    if (!absl::holds_alternative<Tensor<Linear, DataType::FLOAT32>>(
            mul_attr.param) &&
        !absl::holds_alternative<float>(mul_attr.param)) {
      return {
          TransformStatus::DECLINED,
          "This fuse applicable only for broadcast or scalar multiplication."};
    }

    if (conv_node.operation.type == ToString(OperationType::CONVOLUTION_2D)) {
      Convolution2DAttributes* conv_attr =
          absl::any_cast<Convolution2DAttributes>(
              &conv_node.operation.attributes);
      FuseConvolution2DWithMultiply(mul_attr, conv_attr);
    } else if (conv_node.operation.type ==
               ToString(OperationType::CONVOLUTION_TRANSPOSED)) {
      ConvolutionTransposedAttributes* conv_attr =
          absl::any_cast<ConvolutionTransposedAttributes>(
              &conv_node.operation.attributes);
      FuseConvolutionTransposedWithMultiply(mul_attr, conv_attr);
    } else if (conv_node.operation.type ==
               ToString(OperationType::DEPTHWISE_CONVOLUTION)) {
      DepthwiseConvolution2DAttributes* conv_attr =
          absl::any_cast<DepthwiseConvolution2DAttributes>(
              &conv_node.operation.attributes);
      FuseDepthwiseConvolution2DWithMultiply(mul_attr, conv_attr);
    } else if (conv_node.operation.type ==
               ToString(OperationType::FULLY_CONNECTED)) {
      FullyConnectedAttributes* conv_attr =
          absl::any_cast<FullyConnectedAttributes>(
              &conv_node.operation.attributes);
      FuseFullyConnectedWithMultiply(mul_attr, conv_attr);
    } else {
      return {TransformStatus::SKIPPED, ""};
    }

    absl::Status status = RemoveFollowingNode(graph, &mul_node, &conv_node);
    if (!status.ok()) {
      return {TransformStatus::INVALID,
              "Unable to remove mul node after convolution: " +
                  std::string(status.message())};
    }
    return {TransformStatus::APPLIED, ""};
  }
};

class MergeMulWithConvolution : public SequenceTransformation {
 public:
  int ExpectedSequenceLength() const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSfuse_mul_to_convDTcc mht_2(mht_2_v, 279, "", "./tensorflow/lite/delegates/gpu/common/transformations/fuse_mul_to_conv.cc", "ExpectedSequenceLength");
 return 2; }

  TransformResult ApplyToNodesSequence(const std::vector<Node*>& sequence,
                                       GraphFloat32* graph) final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSfuse_mul_to_convDTcc mht_3(mht_3_v, 285, "", "./tensorflow/lite/delegates/gpu/common/transformations/fuse_mul_to_conv.cc", "ApplyToNodesSequence");

    auto& conv_node = *sequence[1];
    if (graph->FindInputs(conv_node.id).size() != 1) {
      return {TransformStatus::DECLINED,
              "This fusion is only applicable to ops with one runtime input."};
    }
    auto& mul_node = *sequence[0];
    if (mul_node.operation.type != ToString(OperationType::MUL) ||
        !mul_node.operation.attributes.has_value()) {
      return {TransformStatus::SKIPPED, ""};
    }

    ElementwiseAttributes mul_attr =
        absl::any_cast<ElementwiseAttributes>(mul_node.operation.attributes);
    if (!absl::holds_alternative<Tensor<Linear, DataType::FLOAT32>>(
            mul_attr.param) &&
        !absl::holds_alternative<float>(mul_attr.param)) {
      return {
          TransformStatus::DECLINED,
          "This fuse applicable only for broadcast or scalar multiplication."};
    }

    if (conv_node.operation.type == ToString(OperationType::CONVOLUTION_2D)) {
      Convolution2DAttributes* conv_attr =
          absl::any_cast<Convolution2DAttributes>(
              &conv_node.operation.attributes);
      FuseMultiplyWithConvolution2D(mul_attr, conv_attr);
    } else if (conv_node.operation.type ==
               ToString(OperationType::CONVOLUTION_TRANSPOSED)) {
      ConvolutionTransposedAttributes* conv_attr =
          absl::any_cast<ConvolutionTransposedAttributes>(
              &conv_node.operation.attributes);
      FuseMultiplyWithConvolutionTransposed(mul_attr, conv_attr);
    } else if (conv_node.operation.type ==
               ToString(OperationType::DEPTHWISE_CONVOLUTION)) {
      DepthwiseConvolution2DAttributes* conv_attr =
          absl::any_cast<DepthwiseConvolution2DAttributes>(
              &conv_node.operation.attributes);
      FuseMultiplyWithDepthwiseConvolution2D(mul_attr, conv_attr);
    } else if (conv_node.operation.type ==
               ToString(OperationType::FULLY_CONNECTED)) {
      FullyConnectedAttributes* conv_attr =
          absl::any_cast<FullyConnectedAttributes>(
              &conv_node.operation.attributes);
      FuseMultiplyWithFullyConnected(mul_attr, conv_attr);
    } else {
      return {TransformStatus::SKIPPED, ""};
    }

    absl::Status status = RemovePrecedingNode(graph, &mul_node, &conv_node);
    if (!status.ok()) {
      return {TransformStatus::INVALID,
              "Unable to remove mul node after convolution: " +
                  std::string(status.message())};
    }
    return {TransformStatus::APPLIED, ""};
  }
};

}  // namespace

std::unique_ptr<SequenceTransformation> NewMergeConvolutionWithMul() {
  return absl::make_unique<MergeConvolutionWithMul>();
}

std::unique_ptr<SequenceTransformation> NewMergeMulWithConvolution() {
  return absl::make_unique<MergeMulWithConvolution>();
}

void FuseConvolution2DWithMultiply(const ElementwiseAttributes& mul_attr,
                                   Convolution2DAttributes* attr) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSfuse_mul_to_convDTcc mht_4(mht_4_v, 358, "", "./tensorflow/lite/delegates/gpu/common/transformations/fuse_mul_to_conv.cc", "FuseConvolution2DWithMultiply");

  auto mul = absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&mul_attr.param);
  auto mul_scalar = absl::get_if<float>(&mul_attr.param);
  for (int d = 0; d < attr->weights.shape.o; ++d) {
    const float multiplier = mul ? mul->data[d] : *mul_scalar;
    for (int s = 0; s < attr->weights.shape.i; ++s) {
      for (int k_y = 0; k_y < attr->weights.shape.h; ++k_y) {
        for (int k_x = 0; k_x < attr->weights.shape.w; ++k_x) {
          const int index = attr->weights.shape.LinearIndex({{d, k_y, k_x, s}});
          attr->weights.data[index] *= multiplier;
        }
      }
    }
    if (!attr->bias.data.empty()) {
      attr->bias.data[d] *= multiplier;
    }
  }
}

void FuseDepthwiseConvolution2DWithMultiply(
    const ElementwiseAttributes& mul_attr,
    DepthwiseConvolution2DAttributes* attr) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSfuse_mul_to_convDTcc mht_5(mht_5_v, 382, "", "./tensorflow/lite/delegates/gpu/common/transformations/fuse_mul_to_conv.cc", "FuseDepthwiseConvolution2DWithMultiply");

  auto mul = absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&mul_attr.param);
  auto mul_scalar = absl::get_if<float>(&mul_attr.param);
  for (int g = 0; g < attr->weights.shape.o; ++g) {
    for (int s = 0; s < attr->weights.shape.i; ++s) {
      const int d = s * attr->weights.shape.o + g;
      const float multiplier = mul ? mul->data[d] : *mul_scalar;
      for (int k_y = 0; k_y < attr->weights.shape.h; ++k_y) {
        for (int k_x = 0; k_x < attr->weights.shape.w; ++k_x) {
          const int index = attr->weights.shape.LinearIndex({{g, k_y, k_x, s}});
          attr->weights.data[index] *= multiplier;
        }
      }
      if (!attr->bias.data.empty()) {
        attr->bias.data[d] *= multiplier;
      }
    }
  }
}

void FuseConvolutionTransposedWithMultiply(
    const ElementwiseAttributes& mul_attr,
    ConvolutionTransposedAttributes* attr) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSfuse_mul_to_convDTcc mht_6(mht_6_v, 407, "", "./tensorflow/lite/delegates/gpu/common/transformations/fuse_mul_to_conv.cc", "FuseConvolutionTransposedWithMultiply");

  auto mul = absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&mul_attr.param);
  auto mul_scalar = absl::get_if<float>(&mul_attr.param);
  for (int d = 0; d < attr->weights.shape.o; ++d) {
    const float multiplier = mul ? mul->data[d] : *mul_scalar;
    for (int s = 0; s < attr->weights.shape.i; ++s) {
      for (int k_y = 0; k_y < attr->weights.shape.h; ++k_y) {
        for (int k_x = 0; k_x < attr->weights.shape.w; ++k_x) {
          const int index = attr->weights.shape.LinearIndex({{d, k_y, k_x, s}});
          attr->weights.data[index] *= multiplier;
        }
      }
    }
    if (!attr->bias.data.empty()) {
      attr->bias.data[d] *= multiplier;
    }
  }
}

void FuseFullyConnectedWithMultiply(const ElementwiseAttributes& mul_attr,
                                    FullyConnectedAttributes* attr) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSfuse_mul_to_convDTcc mht_7(mht_7_v, 430, "", "./tensorflow/lite/delegates/gpu/common/transformations/fuse_mul_to_conv.cc", "FuseFullyConnectedWithMultiply");

  auto mul = absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&mul_attr.param);
  auto mul_scalar = absl::get_if<float>(&mul_attr.param);
  for (int d = 0; d < attr->weights.shape.o; ++d) {
    const float multiplier = mul ? mul->data[d] : *mul_scalar;
    for (int s = 0; s < attr->weights.shape.i; ++s) {
      const int index = attr->weights.shape.LinearIndex({{d, 0, 0, s}});
      attr->weights.data[index] *= multiplier;
    }
    if (!attr->bias.data.empty()) {
      attr->bias.data[d] *= multiplier;
    }
  }
}

void FuseMultiplyWithConvolution2D(const ElementwiseAttributes& mul_attr,
                                   Convolution2DAttributes* attr) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSfuse_mul_to_convDTcc mht_8(mht_8_v, 449, "", "./tensorflow/lite/delegates/gpu/common/transformations/fuse_mul_to_conv.cc", "FuseMultiplyWithConvolution2D");

  auto mul = absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&mul_attr.param);
  auto mul_scalar = absl::get_if<float>(&mul_attr.param);
  for (int s = 0; s < attr->weights.shape.i; ++s) {
    const float multiplier = mul ? mul->data[s] : *mul_scalar;
    for (int d = 0; d < attr->weights.shape.o; ++d) {
      for (int k_y = 0; k_y < attr->weights.shape.h; ++k_y) {
        for (int k_x = 0; k_x < attr->weights.shape.w; ++k_x) {
          const int index = attr->weights.shape.LinearIndex({{d, k_y, k_x, s}});
          attr->weights.data[index] *= multiplier;
        }
      }
    }
  }
}

void FuseMultiplyWithDepthwiseConvolution2D(
    const ElementwiseAttributes& mul_attr,
    DepthwiseConvolution2DAttributes* attr) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSfuse_mul_to_convDTcc mht_9(mht_9_v, 470, "", "./tensorflow/lite/delegates/gpu/common/transformations/fuse_mul_to_conv.cc", "FuseMultiplyWithDepthwiseConvolution2D");

  auto mul = absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&mul_attr.param);
  auto mul_scalar = absl::get_if<float>(&mul_attr.param);
  for (int s = 0; s < attr->weights.shape.i; ++s) {
    const float multiplier = mul ? mul->data[s] : *mul_scalar;
    for (int g = 0; g < attr->weights.shape.o; ++g) {
      for (int k_y = 0; k_y < attr->weights.shape.h; ++k_y) {
        for (int k_x = 0; k_x < attr->weights.shape.w; ++k_x) {
          const int index = attr->weights.shape.LinearIndex({{g, k_y, k_x, s}});
          attr->weights.data[index] *= multiplier;
        }
      }
    }
  }
}

void FuseMultiplyWithConvolutionTransposed(
    const ElementwiseAttributes& mul_attr,
    ConvolutionTransposedAttributes* attr) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSfuse_mul_to_convDTcc mht_10(mht_10_v, 491, "", "./tensorflow/lite/delegates/gpu/common/transformations/fuse_mul_to_conv.cc", "FuseMultiplyWithConvolutionTransposed");

  auto mul = absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&mul_attr.param);
  auto mul_scalar = absl::get_if<float>(&mul_attr.param);
  for (int s = 0; s < attr->weights.shape.i; ++s) {
    const float multiplier = mul ? mul->data[s] : *mul_scalar;
    for (int d = 0; d < attr->weights.shape.o; ++d) {
      for (int k_y = 0; k_y < attr->weights.shape.h; ++k_y) {
        for (int k_x = 0; k_x < attr->weights.shape.w; ++k_x) {
          const int index = attr->weights.shape.LinearIndex({{d, k_y, k_x, s}});
          attr->weights.data[index] *= multiplier;
        }
      }
    }
  }
}

void FuseMultiplyWithFullyConnected(const ElementwiseAttributes& mul_attr,
                                    FullyConnectedAttributes* attr) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSfuse_mul_to_convDTcc mht_11(mht_11_v, 511, "", "./tensorflow/lite/delegates/gpu/common/transformations/fuse_mul_to_conv.cc", "FuseMultiplyWithFullyConnected");

  auto mul = absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&mul_attr.param);
  auto mul_scalar = absl::get_if<float>(&mul_attr.param);
  for (int s = 0; s < attr->weights.shape.i; ++s) {
    const float multiplier = mul ? mul->data[s] : *mul_scalar;
    for (int d = 0; d < attr->weights.shape.o; ++d) {
      const int index = attr->weights.shape.LinearIndex({{d, 0, 0, s}});
      attr->weights.data[index] *= multiplier;
    }
  }
}

}  // namespace gpu
}  // namespace tflite
