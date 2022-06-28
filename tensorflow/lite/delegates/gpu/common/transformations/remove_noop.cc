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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSremove_noopDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSremove_noopDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSremove_noopDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/transformations/remove_noop.h"

#include <algorithm>
#include <any>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/memory/memory.h"
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

using ShouldRemoveOperation = std::function<bool(GraphFloat32* graph, Node*)>;

class RemoveOperation : public SequenceTransformation {
 public:
  explicit RemoveOperation(ShouldRemoveOperation remove_predicate)
      : remove_predicate_(std::move(remove_predicate)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSremove_noopDTcc mht_0(mht_0_v, 216, "", "./tensorflow/lite/delegates/gpu/common/transformations/remove_noop.cc", "RemoveOperation");
}

  int ExpectedSequenceLength() const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSremove_noopDTcc mht_1(mht_1_v, 221, "", "./tensorflow/lite/delegates/gpu/common/transformations/remove_noop.cc", "ExpectedSequenceLength");
 return 2; }

  TransformResult ApplyToNodesSequence(const std::vector<Node*>& sequence,
                                       GraphFloat32* graph) final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSremove_noopDTcc mht_2(mht_2_v, 227, "", "./tensorflow/lite/delegates/gpu/common/transformations/remove_noop.cc", "ApplyToNodesSequence");

    Node* prev_op_node = sequence.front();
    Node* op_node = sequence.back();
    if (!remove_predicate_(graph, op_node)) {
      return {TransformStatus::SKIPPED, ""};
    }
    absl::Status status = RemoveFollowingNode(graph, op_node, prev_op_node);
    if (!status.ok()) {
      return {TransformStatus::INVALID,
              "Unable to remove a node: " + std::string(status.message())};
    }
    return {TransformStatus::APPLIED, ""};
  }

 private:
  ShouldRemoveOperation remove_predicate_;
};

}  // namespace

std::unique_ptr<SequenceTransformation> NewRemoveSingleInputConcat() {
  // Using SequenceTransformation implies that CONCAT has a single input.
  auto type = ToString(OperationType::CONCAT);
  return absl::make_unique<RemoveOperation>(
      [type](GraphFloat32* graph, Node* node) {
        return type == node->operation.type;
      });
}

std::unique_ptr<SequenceTransformation> NewRemoveSingleInputAdd() {
  // Using SequenceTransformation implies that ADD has a single input.
  auto type = ToString(OperationType::ADD);
  return absl::make_unique<RemoveOperation>(
      [type](GraphFloat32* graph, Node* node) {
        if (node->operation.type != type) {
          return false;
        }
        auto& attr = absl::any_cast<const ElementwiseAttributes&>(
            node->operation.attributes);
        return !absl::holds_alternative<Tensor<HWC, DataType::FLOAT32>>(
                   attr.param) &&
               !absl::holds_alternative<Tensor<Linear, DataType::FLOAT32>>(
                   attr.param) &&
               !absl::holds_alternative<float>(attr.param);
      });
}

std::unique_ptr<SequenceTransformation> NewRemoveDegenerateUpsampling() {
  auto type = ToString(OperationType::RESIZE);
  return absl::make_unique<RemoveOperation>(
      [type](GraphFloat32* graph, Node* node) {
        if (node->operation.type != type) {
          return false;
        }
        auto inputs = graph->FindInputs(node->id);
        auto outputs = graph->FindOutputs(node->id);
        return inputs.size() == 1 && outputs.size() == 1 &&
               inputs[0]->tensor.shape == outputs[0]->tensor.shape;
      });
}

class RemoveIdentityReshape : public NodeTransformation {
 public:
  TransformResult ApplyToNode(Node* node, GraphFloat32* graph) final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSremove_noopDTcc mht_3(mht_3_v, 293, "", "./tensorflow/lite/delegates/gpu/common/transformations/remove_noop.cc", "ApplyToNode");

    if (node->operation.type != ToString(OperationType::RESHAPE)) {
      return {TransformStatus::SKIPPED, ""};
    }
    auto input_shape = graph->FindInputs(node->id)[0]->tensor.shape;
    const auto& reshape_attr =
        absl::any_cast<const ReshapeAttributes&>(node->operation.attributes);
    if (input_shape != reshape_attr.new_shape) {
      return {TransformStatus::SKIPPED, ""};
    }
    auto output = graph->FindOutputs(node->id)[0];
    const auto& graph_outputs = graph->outputs();
    if (std::find(graph_outputs.begin(), graph_outputs.end(), output) !=
        graph_outputs.end()) {
      return {TransformStatus::SKIPPED,
              "Can not apply transformation when node output is graph output"};
    }
    absl::Status status = RemoveSimpleNodeKeepInput(graph, node);
    if (!status.ok()) {
      return {TransformStatus::INVALID,
              "Unable to remove a node: " + std::string(status.message())};
    }
    return {TransformStatus::APPLIED,
            "Removed reshape with input_shape == output_shape."};
  }
};

std::unique_ptr<NodeTransformation> NewRemoveIdentityReshape() {
  return absl::make_unique<RemoveIdentityReshape>();
}

class RemoveIdentityStridedSlice : public NodeTransformation {
 public:
  TransformResult ApplyToNode(Node* node, GraphFloat32* graph) final {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSremove_noopDTcc mht_4(mht_4_v, 329, "", "./tensorflow/lite/delegates/gpu/common/transformations/remove_noop.cc", "ApplyToNode");

    if (node->operation.type != ToString(OperationType::SLICE)) {
      return {TransformStatus::SKIPPED, ""};
    }
    auto input = graph->FindInputs(node->id)[0];
    auto output = graph->FindOutputs(node->id)[0];
    const auto& slice_attr =
        absl::any_cast<const SliceAttributes&>(node->operation.attributes);
    if (input->tensor.shape != output->tensor.shape) {
      return {TransformStatus::SKIPPED, ""};
    }
    if (slice_attr.starts != BHWC(0, 0, 0, 0)) {
      return {TransformStatus::SKIPPED, ""};
    }
    if (slice_attr.strides != BHWC(1, 1, 1, 1)) {
      return {TransformStatus::SKIPPED, ""};
    }
    if (slice_attr.ends != output->tensor.shape) {
      return {TransformStatus::SKIPPED, ""};
    }
    const auto& graph_outputs = graph->outputs();
    const auto& graph_inputs = graph->inputs();
    const bool input_is_graph_input =
        std::find(graph_inputs.begin(), graph_inputs.end(), input) !=
        graph_inputs.end();
    const bool output_is_graph_output =
        std::find(graph_outputs.begin(), graph_outputs.end(), output) !=
        graph_outputs.end();
    if (input_is_graph_input && output_is_graph_output) {
      return {TransformStatus::SKIPPED,
              "Can not apply transformation when node input is graph input and "
              "node output is graph output"};
    }
    if (output_is_graph_output) {
      if (graph->FindConsumers(input->id).size() != 1) {
        return {TransformStatus::SKIPPED,
                "Can not apply transformation when node output is graph output "
                "and input consumed by other nodes."};
      }
      absl::Status status = RemoveSimpleNodeKeepOutput(graph, node);
      if (!status.ok()) {
        return {TransformStatus::INVALID,
                "Unable to remove a node: " + std::string(status.message())};
      }
      return {TransformStatus::APPLIED, "Removed identity strided slice."};
    }
    absl::Status status = RemoveSimpleNodeKeepInput(graph, node);
    if (!status.ok()) {
      return {TransformStatus::INVALID,
              "Unable to remove a node: " + std::string(status.message())};
    }
    return {TransformStatus::APPLIED, "Removed identity strided slice."};
  }
};

std::unique_ptr<NodeTransformation> NewRemoveIdentityStridedSlice() {
  return absl::make_unique<RemoveIdentityStridedSlice>();
}

}  // namespace gpu
}  // namespace tflite
