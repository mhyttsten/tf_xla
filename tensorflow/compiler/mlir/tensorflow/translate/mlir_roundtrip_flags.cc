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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSmlir_roundtrip_flagsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSmlir_roundtrip_flagsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSmlir_roundtrip_flagsDTcc() {
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

#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"

#include <ostream>
#include <sstream>
#include <type_traits>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "llvm/ADT/Optional.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

std::string GraphImportConfig::str() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSmlir_roundtrip_flagsDTcc mht_0(mht_0_v, 208, "", "./tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.cc", "GraphImportConfig::str");

  std::ostringstream ss;

  ss << "graph_func_name: " << graph_func_name;
  InputArrays inputs;
  ss << "\ninputs: ";
  for (auto& it : inputs) {
    ss << "\n\t" << it.first << " -> "
       << DataTypeString(it.second.imported_dtype) << " "
       << it.second.shape.DebugString();
  }
  ss << "\noutputs:";
  for (auto& output : outputs) ss << " " << output;
  ss << "\ncontrol_outputs:";
  for (auto& output : control_outputs) ss << " " << output;
  ss << "\nprune_unused_nodes: " << prune_unused_nodes;
  ss << "\nconvert_legacy_fed_inputs: " << convert_legacy_fed_inputs;
  ss << "\ngraph_as_function: " << graph_as_function;
  ss << "\nupgrade_legacy: " << upgrade_legacy;
  ss << "\nrestrict_functionalization_to_tpu_nodes: "
     << restrict_functionalization_to_tpu_nodes;
  ss << "\nenable_shape_inference: " << enable_shape_inference;
  ss << "\nunconditionally_use_set_output_shapes: "
     << unconditionally_use_set_output_shapes;

  return ss.str();
}

Status ParseOutputArrayInfo(absl::string_view array_names,
                            std::vector<string>* outputs) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("array_names: \"" + std::string(array_names.data(), array_names.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSmlir_roundtrip_flagsDTcc mht_1(mht_1_v, 241, "", "./tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.cc", "ParseOutputArrayInfo");

  TF_RETURN_IF_ERROR(ParseNodeNames(array_names, *outputs));
  return Status::OK();
}

Status ParseOutputArrayInfo(const std::vector<string>& output_names,
                            std::vector<string>* outputs) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSmlir_roundtrip_flagsDTcc mht_2(mht_2_v, 250, "", "./tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.cc", "ParseOutputArrayInfo");

  for (auto& output_name : output_names) {
    if (output_name.empty()) continue;
    outputs->push_back(output_name);
  }
  return Status::OK();
}

Status ParseInputArrayInfo(absl::string_view array_names,
                           absl::string_view data_types,
                           absl::string_view shapes,
                           GraphImportConfig::InputArrays* inputs) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("array_names: \"" + std::string(array_names.data(), array_names.size()) + "\"");
   mht_3_v.push_back("data_types: \"" + std::string(data_types.data(), data_types.size()) + "\"");
   mht_3_v.push_back("shapes: \"" + std::string(shapes.data(), shapes.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSmlir_roundtrip_flagsDTcc mht_3(mht_3_v, 267, "", "./tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.cc", "ParseInputArrayInfo");

  std::vector<string> node_names;
  std::vector<string> node_dtypes;
  std::vector<llvm::Optional<std::vector<int>>> node_shapes;
  TF_RETURN_IF_ERROR(ParseNodeNames(array_names, node_names));
  TF_RETURN_IF_ERROR(ParseNodeDataTypes(data_types, node_dtypes));
  TF_RETURN_IF_ERROR(ParseNodeShapes(shapes, node_shapes));
  return ParseInputArrayInfo(node_names, node_dtypes, node_shapes, inputs);
}

Status ParseInputArrayInfo(
    const std::vector<string>& node_names,
    const std::vector<string>& node_dtypes,
    const std::vector<llvm::Optional<std::vector<int>>>& node_shapes,
    GraphImportConfig::InputArrays* inputs) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSmlir_roundtrip_flagsDTcc mht_4(mht_4_v, 284, "", "./tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.cc", "ParseInputArrayInfo");

  std::vector<std::string> used_node_dtypes;
  if (node_dtypes.empty()) {
    // Mark all the node dtypes Invalid, so the importer can handle them by
    // using the type from the graph.
    used_node_dtypes.resize(node_names.size(), DataType_Name(DT_INVALID));
  } else if (node_names.size() == node_dtypes.size()) {
    for (const auto& dtype : node_dtypes) {
      if (dtype.empty()) {
        used_node_dtypes.push_back(DataType_Name(DT_INVALID));
      } else if (dtype != DataType_Name(DT_INVALID)) {
        used_node_dtypes.push_back(dtype);
      } else {
        return errors::FailedPrecondition(
            "Use '' if want to use the type from graph.");
      }
    }
  } else {
    return errors::FailedPrecondition(absl::StrCat(
        "Unmatched node array and data type numbers (#arrays ",
        node_names.size(), ", #data_types ", node_dtypes.size(), ")"));
  }

  if (!node_shapes.empty() && node_names.size() != node_shapes.size()) {
    return errors::FailedPrecondition(absl::StrCat(
        "Unmatched node array and shape numbers (#arrays ", node_names.size(),
        ", #input_shapes ", node_shapes.size(), ")"));
  }

  // StringMap doesn't support reserve else reserve input map size here.
  for (int i = 0, end = node_names.size(); i < end; i++) {
    auto& name = node_names[i];
    if (name.empty()) continue;

    auto it_inserted_pair = inputs->insert({name, {}});
    if (!it_inserted_pair.second)
      return errors::FailedPrecondition(
          absl::StrCat("tensor ", name, " is repeated in the arrays flag"));

    ArrayInfo& info = it_inserted_pair.first->second;
    if (!DataType_Parse(used_node_dtypes[i], &info.imported_dtype)) {
      return errors::FailedPrecondition(
          absl::StrCat("Invalid node type '", node_dtypes[i], "'"));
    }

    if (!node_shapes.empty()) {
      if (!node_shapes[i].hasValue()) {
        info.shape.set_unknown_rank(true);
        continue;
      }
      for (auto& dim : node_shapes[i].getValue()) {
        info.shape.add_dim()->set_size(dim);
      }
    }
  }
  return Status::OK();
}

Status ParseNodeShapes(
    absl::string_view shapes_str,
    std::vector<llvm::Optional<std::vector<int>>>& shapes_vector) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("shapes_str: \"" + std::string(shapes_str.data(), shapes_str.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSmlir_roundtrip_flagsDTcc mht_5(mht_5_v, 348, "", "./tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.cc", "ParseNodeShapes");

  shapes_vector.clear();
  if (!shapes_str.empty()) {
    std::vector<string> node_shapes_str = absl::StrSplit(shapes_str, ':');
    for (int i = 0; i < node_shapes_str.size(); i++) {
      if (node_shapes_str[i] == "*") {
        shapes_vector.push_back(llvm::None);
        continue;
      }
      std::vector<int> dims;
      for (const absl::string_view dim_str :
           absl::StrSplit(node_shapes_str[i], ',')) {
        // Treats empty input shape as scalar
        if (dim_str.empty()) continue;
        if (dim_str == "?") {
          dims.push_back(-1);
          continue;
        }
        int size;
        TF_RET_CHECK(absl::SimpleAtoi(dim_str, &size));
        dims.push_back(size);
      }
      shapes_vector.push_back(dims);
    }
  }
  return Status::OK();
}

Status ParseNodeNames(absl::string_view names_str,
                      std::vector<std::string>& names_vector) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("names_str: \"" + std::string(names_str.data(), names_str.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSmlir_roundtrip_flagsDTcc mht_6(mht_6_v, 381, "", "./tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.cc", "ParseNodeNames");

  names_vector = absl::StrSplit(names_str, ',', absl::SkipEmpty());
  return Status::OK();
}

Status ParseNodeDataTypes(absl::string_view data_types_str,
                          std::vector<std::string>& data_type_vector) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("data_types_str: \"" + std::string(data_types_str.data(), data_types_str.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStranslatePSmlir_roundtrip_flagsDTcc mht_7(mht_7_v, 391, "", "./tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.cc", "ParseNodeDataTypes");

  data_type_vector.clear();
  if (!data_types_str.empty()) {
    data_type_vector = absl::StrSplit(data_types_str, ',');
  }
  return Status::OK();
}

}  // namespace tensorflow
