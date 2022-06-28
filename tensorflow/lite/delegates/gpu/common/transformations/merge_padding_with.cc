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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSmerge_padding_withDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSmerge_padding_withDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSmerge_padding_withDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/transformations/merge_padding_with.h"

#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/data_type.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/shape.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"
#include "tensorflow/lite/delegates/gpu/common/transformations/matching.h"

namespace tflite {
namespace gpu {
namespace {

template <typename Attr>
class MergePaddingWith2DOperation : public SequenceTransformation {
 public:
  explicit MergePaddingWith2DOperation(OperationType operation_type)
      : operations_to_match_(
            {ToString(OperationType::PAD), ToString(operation_type)}) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSmerge_padding_withDTcc mht_0(mht_0_v, 214, "", "./tensorflow/lite/delegates/gpu/common/transformations/merge_padding_with.cc", "MergePaddingWith2DOperation");
}

  int ExpectedSequenceLength() const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSmerge_padding_withDTcc mht_1(mht_1_v, 219, "", "./tensorflow/lite/delegates/gpu/common/transformations/merge_padding_with.cc", "ExpectedSequenceLength");
 return 2; }

  TransformResult ApplyToNodesSequence(const std::vector<Node*>& sequence,
                                       GraphFloat32* graph) final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSmerge_padding_withDTcc mht_2(mht_2_v, 225, "", "./tensorflow/lite/delegates/gpu/common/transformations/merge_padding_with.cc", "ApplyToNodesSequence");

    if (!MatchesByOperationType(sequence, operations_to_match_)) {
      return {TransformStatus::SKIPPED, ""};
    }

    Node* pad_node = sequence.front();
    Node* op_node = sequence.back();

    PadAttributes pad_attr =
        absl::any_cast<PadAttributes>(pad_node->operation.attributes);

    if (pad_attr.type != PaddingContentType::ZEROS) {
      return {TransformStatus::DECLINED, "Only Zero padding is supported."};
    }
    if (pad_attr.appended.c != 0 || pad_attr.prepended.c != 0 ||
        pad_attr.appended.b != 0 || pad_attr.prepended.b != 0) {
      return {TransformStatus::DECLINED,
              "Pad has non-zero padding on non HW axis."};
    }

    Attr* node_attr = absl::any_cast<Attr>(&op_node->operation.attributes);
    absl::Status status = RemovePrecedingNode(graph, pad_node, op_node);
    if (!status.ok()) {
      return {TransformStatus::INVALID,
              "Unable to remove Pad node with Operation node: " +
                  std::string(status.message())};
    }

    node_attr->padding.appended.h += pad_attr.appended.h;
    node_attr->padding.appended.w += pad_attr.appended.w;
    node_attr->padding.prepended.h += pad_attr.prepended.h;
    node_attr->padding.prepended.w += pad_attr.prepended.w;
    return {
        TransformStatus::APPLIED,
        absl::StrCat("Added padding: prepended = {h = ", pad_attr.prepended.h,
                     ", w = ", pad_attr.prepended.w, "}, appended = { h = ",
                     pad_attr.appended.h, ", w = ", pad_attr.appended.w, "}")};
  }

 private:
  const std::vector<std::string> operations_to_match_;
};

}  // namespace

std::unique_ptr<SequenceTransformation> NewMergePaddingWithPooling() {
  return absl::make_unique<MergePaddingWith2DOperation<Pooling2DAttributes>>(
      OperationType::POOLING_2D);
}

std::unique_ptr<SequenceTransformation> NewMergePaddingWithConvolution2D() {
  return absl::make_unique<
      MergePaddingWith2DOperation<Convolution2DAttributes>>(
      OperationType::CONVOLUTION_2D);
}

std::unique_ptr<SequenceTransformation>
NewMergePaddingWithDepthwiseConvolution() {
  return absl::make_unique<
      MergePaddingWith2DOperation<DepthwiseConvolution2DAttributes>>(
      OperationType::DEPTHWISE_CONVOLUTION);
}

class MergePaddingWithAddOperation : public NodeTransformation {
 public:
  TransformResult ApplyToNode(Node* node, GraphFloat32* graph) final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSmerge_padding_withDTcc mht_3(mht_3_v, 293, "", "./tensorflow/lite/delegates/gpu/common/transformations/merge_padding_with.cc", "ApplyToNode");

    if (node->operation.type != ToString(OperationType::PAD)) {
      return {TransformStatus::SKIPPED, ""};
    }
    auto inputs = graph->FindInputs(node->id);
    if (inputs.size() != 1) {
      return {TransformStatus::SKIPPED, ""};
    }

    const auto& input_shape = graph->FindInputs(node->id)[0]->tensor.shape;
    if (input_shape.c % 4 != 0) {
      return {TransformStatus::DECLINED,
              "Pad with input where src_channels % 4 != 0"};
    }

    PadAttributes pad_attr =
        absl::any_cast<PadAttributes>(node->operation.attributes);

    if (pad_attr.type != PaddingContentType::ZEROS) {
      return {TransformStatus::DECLINED, "Only Zero padding is supported."};
    }
    if (pad_attr.prepended != BHWC(0, 0, 0, 0) || pad_attr.appended.h != 0 ||
        pad_attr.appended.w != 0 || pad_attr.appended.b != 0) {
      return {TransformStatus::DECLINED,
              "Pad has padding not only in appended channels axis."};
    }

    auto pad_output = graph->FindOutputs(node->id)[0];
    auto consumer_nodes = graph->FindConsumers(pad_output->id);
    if (consumer_nodes.size() != 1) {
      return {TransformStatus::SKIPPED, ""};
    }
    auto add_node = consumer_nodes[0];
    auto consumer_type = OperationTypeFromString(add_node->operation.type);
    if (consumer_type != OperationType::ADD) {
      return {TransformStatus::SKIPPED, ""};
    }

    ElementwiseAttributes add_attr =
        absl::any_cast<ElementwiseAttributes>(add_node->operation.attributes);
    const bool is_add_hwc =
        absl::holds_alternative<Tensor<HWC, DataType::FLOAT32>>(add_attr.param);
    const bool is_add_linear =
        absl::holds_alternative<Tensor<Linear, DataType::FLOAT32>>(
            add_attr.param);
    const bool is_add_scalar = absl::holds_alternative<float>(add_attr.param);
    if (is_add_hwc || is_add_linear || is_add_scalar) {
      return {TransformStatus::SKIPPED,
              "Cannot remove padding when ADD has constant argument."};
    }

    absl::Status status = RemovePrecedingNode(graph, node, add_node);
    if (!status.ok()) {
      return {TransformStatus::INVALID,
              "Unable to remove Pad node " + std::string(status.message())};
    }

    return {TransformStatus::APPLIED,
            "Removed padding with zeroes in appended channels dimension"};
  }
};

std::unique_ptr<NodeTransformation> NewMergePaddingWithAdd() {
  return absl::make_unique<MergePaddingWithAddOperation>();
}

}  // namespace gpu
}  // namespace tflite
