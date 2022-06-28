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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSadd_quant_adjustmentsDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSadd_quant_adjustmentsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSadd_quant_adjustmentsDTcc() {
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

#include "tensorflow/lite/delegates/gpu/common/transformations/add_quant_adjustments.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/types/any.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/model_transformer.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/status.h"
#include "tensorflow/lite/delegates/gpu/common/tensor.h"

namespace tflite {
namespace gpu {

class AddQuantAdjustments : public NodeTransformation {
 public:
  TransformResult ApplyToNode(Node* node, GraphFloat32* graph) final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPScommonPStransformationsPSadd_quant_adjustmentsDTcc mht_0(mht_0_v, 206, "", "./tensorflow/lite/delegates/gpu/common/transformations/add_quant_adjustments.cc", "ApplyToNode");

    if (node->operation.type ==
        ToString(OperationType::QUANTIZE_AND_DEQUANTIZE)) {
      return {TransformStatus::SKIPPED, ""};
    }

    bool transform_applied = false;
    auto node_outputs = graph->FindOutputs(node->id);
    for (auto output_value : node_outputs) {
      // Skip if quantization doesn't apply.
      if (!output_value->quant_params) continue;
      auto consumers = graph->FindConsumers(output_value->id);
      // No need to do anything if this isn't consumed by another node.
      if (consumers.empty()) {
        continue;
      }

      // Add a new QuantizeAndDequantize node.
      Node* quant_and_dequant_node;
      absl::Status status =
          graph->InsertNodeAfter(node->id, &quant_and_dequant_node);
      if (!status.ok()) {
        return {TransformStatus::INVALID, "Could not insert new node."};
      }
      quant_and_dequant_node->operation.type =
          ToString(OperationType::QUANTIZE_AND_DEQUANTIZE);
      QuantizeAndDequantizeAttributes attr;
      attr.min = output_value->quant_params.value().min;
      attr.max = output_value->quant_params.value().max;
      attr.scale = output_value->quant_params.value().scale;
      quant_and_dequant_node->operation.attributes = attr;

      // Add one output Value for the new node.
      // The tensor information should rename the same.
      Value* adjusted_value = graph->NewValue();
      adjusted_value->tensor = output_value->tensor;
      status =
          graph->SetProducer(quant_and_dequant_node->id, adjusted_value->id);
      if (!status.ok()) {
        return {TransformStatus::INVALID,
                "Could not create QuantizeAndDequantize node."};
      }

      // Replace output_value with adjusted_value on all consumers.
      for (auto& consumer : consumers) {
        status = graph->ReplaceInput(consumer->id, output_value->id,
                                     adjusted_value->id);
        if (!status.ok()) {
          return {TransformStatus::INVALID,
                  absl::StrCat(
                      "Failed to associate quant-adjusted value for consumer: ",
                      status.message())};
        }
      }

      // Add QuantizeAndDequantize node as a consumer of output_value.
      status = graph->AddConsumer(quant_and_dequant_node->id, output_value->id);
      if (!status.ok()) {
        return {TransformStatus::INVALID,
                absl::StrCat(
                    "Could not associate output to QuantizeAndDequantize: ",
                    status.message())};
      }

      // Remove quant params on output_value, to make the transformation
      // idempotent.
      output_value->quant_params.reset();
      transform_applied = true;
    }

    if (transform_applied) {
      return {TransformStatus::APPLIED, ""};
    }
    return {TransformStatus::SKIPPED, ""};
  }
};

std::unique_ptr<NodeTransformation> NewAddQuantAdjustments() {
  return absl::make_unique<AddQuantAdjustments>();
}

}  // namespace gpu
}  // namespace tflite
