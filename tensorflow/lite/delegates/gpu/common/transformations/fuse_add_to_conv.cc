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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSfuse_add_to_convDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSfuse_add_to_convDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSfuse_add_to_convDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/transformations/fuse_add_to_conv.h"

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

void FuseBiasWithAddAttributes(const ElementwiseAttributes& add_attr,
                               const int channels,
                               Tensor<Linear, DataType::FLOAT32>* bias) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSfuse_add_to_convDTcc mht_0(mht_0_v, 208, "", "./tensorflow/lite/delegates/gpu/common/transformations/fuse_add_to_conv.cc", "FuseBiasWithAddAttributes");

  auto add = absl::get_if<Tensor<Linear, DataType::FLOAT32>>(&add_attr.param);
  auto add_scalar = absl::get_if<float>(&add_attr.param);
  if (bias->data.empty()) {
    *bias = MakeZeroTensor<Linear, DataType::FLOAT32>(Linear(channels));
  }
  for (int d = 0; d < channels; ++d) {
    bias->data[d] += add ? add->data[d] : *add_scalar;
  }
}

class MergeConvolutionWithAdd : public SequenceTransformation {
 public:
  int ExpectedSequenceLength() const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSfuse_add_to_convDTcc mht_1(mht_1_v, 224, "", "./tensorflow/lite/delegates/gpu/common/transformations/fuse_add_to_conv.cc", "ExpectedSequenceLength");
 return 2; }

  TransformResult ApplyToNodesSequence(const std::vector<Node*>& sequence,
                                       GraphFloat32* graph) final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSfuse_add_to_convDTcc mht_2(mht_2_v, 230, "", "./tensorflow/lite/delegates/gpu/common/transformations/fuse_add_to_conv.cc", "ApplyToNodesSequence");

    auto& conv_node = *sequence[0];
    if (graph->FindInputs(conv_node.id).size() != 1) {
      return {TransformStatus::DECLINED,
              "This fusion is only applicable to ops with one runtime input."};
    }
    auto& add_node = *sequence[1];
    if (add_node.operation.type != ToString(OperationType::ADD)) {
      return {TransformStatus::SKIPPED, ""};
    }
    ElementwiseAttributes add_attr =
        absl::any_cast<ElementwiseAttributes>(add_node.operation.attributes);
    if (!absl::holds_alternative<Tensor<Linear, DataType::FLOAT32>>(
            add_attr.param) &&
        !absl::holds_alternative<float>(add_attr.param)) {
      return {TransformStatus::DECLINED,
              "This fuse applicable only for broadcast or scalar addition."};
    }

    if (conv_node.operation.type == ToString(OperationType::CONVOLUTION_2D)) {
      Convolution2DAttributes* conv_attr =
          absl::any_cast<Convolution2DAttributes>(
              &conv_node.operation.attributes);
      FuseConvolution2DWithAdd(add_attr, conv_attr);
    } else if (conv_node.operation.type ==
               ToString(OperationType::CONVOLUTION_TRANSPOSED)) {
      ConvolutionTransposedAttributes* conv_attr =
          absl::any_cast<ConvolutionTransposedAttributes>(
              &conv_node.operation.attributes);
      FuseConvolutionTransposedWithAdd(add_attr, conv_attr);
    } else if (conv_node.operation.type ==
               ToString(OperationType::DEPTHWISE_CONVOLUTION)) {
      DepthwiseConvolution2DAttributes* conv_attr =
          absl::any_cast<DepthwiseConvolution2DAttributes>(
              &conv_node.operation.attributes);
      FuseDepthwiseConvolution2DWithAdd(add_attr, conv_attr);
    } else if (conv_node.operation.type ==
               ToString(OperationType::FULLY_CONNECTED)) {
      FullyConnectedAttributes* conv_attr =
          absl::any_cast<FullyConnectedAttributes>(
              &conv_node.operation.attributes);
      FuseFullyConnectedWithAdd(add_attr, conv_attr);
    } else {
      return {TransformStatus::SKIPPED, ""};
    }

    absl::Status status = RemoveFollowingNode(graph, &add_node, &conv_node);
    if (!status.ok()) {
      return {TransformStatus::INVALID,
              "Unable to remove add node after convolution: " +
                  std::string(status.message())};
    }
    return {TransformStatus::APPLIED, ""};
  }
};

}  // namespace

std::unique_ptr<SequenceTransformation> NewMergeConvolutionWithAdd() {
  return absl::make_unique<MergeConvolutionWithAdd>();
}

void FuseConvolution2DWithAdd(const ElementwiseAttributes& add_attr,
                              Convolution2DAttributes* attr) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSfuse_add_to_convDTcc mht_3(mht_3_v, 296, "", "./tensorflow/lite/delegates/gpu/common/transformations/fuse_add_to_conv.cc", "FuseConvolution2DWithAdd");

  FuseBiasWithAddAttributes(add_attr, attr->weights.shape.o, &attr->bias);
}

void FuseDepthwiseConvolution2DWithAdd(const ElementwiseAttributes& add_attr,
                                       DepthwiseConvolution2DAttributes* attr) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSfuse_add_to_convDTcc mht_4(mht_4_v, 304, "", "./tensorflow/lite/delegates/gpu/common/transformations/fuse_add_to_conv.cc", "FuseDepthwiseConvolution2DWithAdd");

  FuseBiasWithAddAttributes(
      add_attr, attr->weights.shape.o * attr->weights.shape.i, &attr->bias);
}

void FuseConvolutionTransposedWithAdd(const ElementwiseAttributes& add_attr,
                                      ConvolutionTransposedAttributes* attr) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSfuse_add_to_convDTcc mht_5(mht_5_v, 313, "", "./tensorflow/lite/delegates/gpu/common/transformations/fuse_add_to_conv.cc", "FuseConvolutionTransposedWithAdd");

  FuseBiasWithAddAttributes(add_attr, attr->weights.shape.o, &attr->bias);
}

void FuseFullyConnectedWithAdd(const ElementwiseAttributes& add_attr,
                               FullyConnectedAttributes* attr) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSfuse_add_to_convDTcc mht_6(mht_6_v, 321, "", "./tensorflow/lite/delegates/gpu/common/transformations/fuse_add_to_conv.cc", "FuseFullyConnectedWithAdd");

  FuseBiasWithAddAttributes(add_attr, attr->weights.shape.o, &attr->bias);
}

}  // namespace gpu
}  // namespace tflite
