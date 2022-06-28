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
class MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSfuse_auto_inputDTcc {
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
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSfuse_auto_inputDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSfuse_auto_inputDTcc() {
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

#include "tensorflow/lite/delegates/gpu/gl/compiler/fuse_auto_input.h"

#include <string>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/types/any.h"
#include "absl/types/variant.h"
#include "tensorflow/lite/delegates/gpu/common/model.h"
#include "tensorflow/lite/delegates/gpu/common/operations.h"
#include "tensorflow/lite/delegates/gpu/common/types.h"
#include "tensorflow/lite/delegates/gpu/gl/compiler/compiled_node.h"

namespace tflite {
namespace gpu {
namespace gl {
namespace {

std::pair<std::string, std::string> MakeValueReplacement(int n, int k) {
  return {absl::StrCat("value_", n), absl::StrCat("value_", k)};
}

std::pair<std::string, std::string> MakeDataReplacement(int n, int k) {
  return {absl::StrCat("input_data_", n), absl::StrCat("input_data_", k)};
}

}  // namespace

TransformResult FuseAutoInput::ApplyToNode(Node* node, GraphFloat32* graph) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSdelegatesPSgpuPSglPScompilerPSfuse_auto_inputDTcc mht_0(mht_0_v, 215, "", "./tensorflow/lite/delegates/gpu/gl/compiler/fuse_auto_input.cc", "FuseAutoInput::ApplyToNode");

  auto& node_attr =
      absl::any_cast<CompiledNodeAttributes&>(node->operation.attributes);
  auto& node_code = node_attr.code;

  if (node_code.input != IOStructure::AUTO) {
    return {TransformStatus::SKIPPED, ""};
  }
  uint3 workgroup = node_code.workgroup;

  auto node_outputs = graph->FindOutputs(node->id);

  // Check which inputs could be fused into the current node.
  std::vector<std::pair<Node*, int>> nodes_to_fuse;
  std::vector<std::pair<ValueId, int>> input_values;
  int input_num = -1;
  for (auto input_value : graph->FindInputs(node->id)) {
    input_num++;
    const ValueId input_id = input_value->id;
    input_values.push_back({input_id, input_num});

    if (graph->FindConsumers(input_id).size() > 1) {
      continue;  // input is consumed by >1 nodes
    }
    Node* input_producer = graph->FindProducer(input_id);
    if (input_producer == nullptr) {
      continue;  // graph's input
    }
    if (graph->FindOutputs(input_producer->id).size() != 1) {
      continue;  // input node has more than one output
    }
    auto& input_producer_attr = absl::any_cast<const CompiledNodeAttributes&>(
        input_producer->operation.attributes);
    if (input_producer_attr.code.output != IOStructure::AUTO) {
      continue;
    }
    if (input_producer_attr.code.workload != node_code.workload &&
        uint3() != input_producer_attr.code.workload) {
      continue;
    }
    if (input_producer_attr.code.workgroup != uint3()) {
      // New fused node should fuse only a single shader that has pre-defined
      // workgroup. Such shader is considered "heavy". Do not fuse two heavy
      // shaders into one.
      // TODO(eignasheva): make sure it still works.
      if (workgroup != uint3()) {
        continue;
      }
      workgroup = input_producer_attr.code.workgroup;
    }
    nodes_to_fuse.push_back({input_producer, input_num});
    input_values.pop_back();  // this value will not be used as input.
  }
  if (nodes_to_fuse.empty()) {
    return {TransformStatus::SKIPPED, ""};
  }

  // Skip fusions which will result in duplicate inputs, e.g. diamond shapes.
  {
    absl::flat_hash_set<ValueId> all_inputs;
    for (const auto& node_to_fuse : nodes_to_fuse) {
      for (const auto& input : graph->FindInputs(node_to_fuse.first->id)) {
        if (all_inputs.find(input->id) != all_inputs.end()) {
          return {TransformStatus::SKIPPED, ""};
        }
        all_inputs.insert(input->id);
      }
    }
    for (const auto& input : graph->FindInputs(node->id)) {
      if (all_inputs.find(input->id) != all_inputs.end()) {
        return {TransformStatus::SKIPPED, ""};
      }
      all_inputs.insert(input->id);
    }
  }

  // Break connections between current node and its inputs.
  for (auto value : graph->FindInputs(node->id)) {
    if (!graph->RemoveConsumer(node->id, value->id).ok()) {
      return {TransformStatus::INVALID, ""};
    }
  }

  std::string operation_type;
  std::string source_code;
  std::string values;

  // Node source code need to be appended later to the end.
  std::swap(source_code, node_code.source_code);

  // Indicates value_k that is beyond originally declared [0..n] values,
  // therefore, it can be used by newly added dependencies.
  int extra_input_num = input_num;
  input_num = 0;

  // Fuse all nodes into one.
  for (auto input_and_num : nodes_to_fuse) {
    auto& input = input_and_num.first;
    auto& attr =
        absl::any_cast<CompiledNodeAttributes&>(input->operation.attributes);
    auto super_inputs = graph->FindInputs(input->id);

    // Replace all internal references in the input source code. For example:
    // source code "value_0 = max(0, value_0);" will be rewritten into
    // "value_2 = max(0, value_2);"
    std::vector<std::pair<std::string, std::string>> replacements;
    for (int i = 0; i < super_inputs.size(); ++i) {
      // Node source code uses value_N to access output value from the fused
      // node. Use correct reference.
      //
      // Here value_N does not correspond to input_N anymore. Instead it tracks
      // value_n and input_m independently. Value_index uses an index needed
      // for the "final" shader, while input_num preserves the order of inputs.
      // For example:
      //    Shader A: input_0, input_1
      //    value_0 = value_0 > value_1 ? value_0 : value_1;
      //
      //    Shader B:  input_0
      //    value_0 = max(0, value_0);
      //
      //    AddShader: input_0, input_1
      //    value_0 = value_0 + value_1;
      //
      //    Fused shader is going to have 3 inputs: input_0 (A), input_1 (A),
      //    input_2 (B). But Shader B need to store result in value_1, because
      //    AddShader refers to it as 'value_1'. So, fused shader will look as
      //    follows:
      //
      //    // Shader A
      //    vec4 value_0 = input_data_0.data[gid.x, gid.y, gid.z];
      //    vec4 value_2 = input_data_1.data[gid.x, gid.y, gid.z];
      //    value_0 = value_0 > value_2 ? value_0 : value_2;
      //
      //    // Shader B
      //    vec4 value_1 = input_data_2.data[gid.x, gid.y, gid.z];
      //    value_1 = max(0, value_1);
      //
      //    // AddShader
      //    value_0 = value_0 + value_1;
      //
      //    output_data_0.data[gid.x, gid.y, gid.z] = value_0;
      int value_index = i == 0 ? input_and_num.second : ++extra_input_num;
      replacements.push_back(MakeValueReplacement(i, value_index));
      replacements.push_back(MakeDataReplacement(i, input_num));

      // Declare input values based on the input structure of the merged node.
      // This code copies what shader_codegen would do automatically.
      if (attr.code.input == IOStructure::AUTO) {
        absl::StrAppend(&values, "  value_", value_index, " = $input_data_",
                        input_num, "[gid.x, gid.y, gid.z]$;\n");
      }

      if (!graph->AddConsumer(node->id, super_inputs[i]->id).ok()) {
        return {TransformStatus::INVALID, ""};
      }
      input_num++;
    }

    // Also rename all _h and _w parameters to the new names.
    for (auto& param : attr.code.parameters) {
      param.name = absl::StrReplaceAll(param.name, replacements);
    }
    attr.code.source_code =
        absl::StrReplaceAll(attr.code.source_code, replacements);

    // Merge all objects, parameters and source code.
    if (!MergeCode(&attr, &node_attr).ok()) {
      return {TransformStatus::INVALID, "Unable to merge the code"};
    }
    absl::StrAppend(&node_attr.code.source_code, "{\n", attr.code.source_code,
                    "\n}");

    if (!operation_type.empty()) {
      operation_type += ",";
    }
    operation_type += input->operation.type;

    if (!graph->DeleteNode(input->id).ok()) {
      return {TransformStatus::INVALID, ""};
    }
  }

  // Add back all inputs that are used directly by the fused node.
  for (int i = 0; i < input_values.size(); i++) {
    if (node_code.input == IOStructure::AUTO) {
      absl::StrAppend(&values, "  value_", input_values[i].second,
                      " = $input_data_", input_num,
                      "[gid.x, gid.y, gid.z]$;\n");
    }
    if (!graph->AddConsumer(node->id, input_values[i].first).ok()) {
      return {TransformStatus::INVALID, ""};
    }
    input_num++;
  }

  node_code.input = IOStructure::ONLY_DEFINITIONS;

  absl::StrAppend(&node->operation.type, "(", operation_type, ")");
  node_code.source_code =
      absl::StrCat(values, node_code.source_code, "{//FUSED",
                   node->operation.type, "\n", source_code, "\n}");

  return {TransformStatus::APPLIED, ""};
}

}  // namespace gl
}  // namespace gpu
}  // namespace tflite
