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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_configDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_configDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_configDTcc() {
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

#include "tensorflow/compiler/mlir/lite/quantization/quantization_config.h"

#include <algorithm>
#include <ios>
#include <sstream>
#include <string>
#include <vector>

#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "tensorflow/core/framework/types.pb.h"

// Is this dtype a quantization type from TensorFlow.
static bool IsQuantizationType(tensorflow::DataType dtype) {
  switch (dtype) {
    case tensorflow::DT_QINT8:
    case tensorflow::DT_QUINT8:
    case tensorflow::DT_QINT16:
    case tensorflow::DT_QUINT16:
    case tensorflow::DT_QINT32:
      return true;
    default:
      return false;
  }
}

namespace mlir {
namespace quant {
namespace {
bool GetBooleanSpecs(const std::string& bool_val) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("bool_val: \"" + bool_val + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_configDTcc mht_0(mht_0_v, 218, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_config.cc", "GetBooleanSpecs");

  bool result;
  std::stringstream iss(bool_val);
  iss >> std::boolalpha >> result;
  return result;
}
}  // namespace

void ParseCustomOpSpecs(absl::string_view node_names,
                        const CustomOpUpdateOptions& update_option,
                        CustomOpMap& custom_op_map) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("node_names: \"" + std::string(node_names.data(), node_names.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_configDTcc mht_1(mht_1_v, 232, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_config.cc", "ParseCustomOpSpecs");

  if (node_names.empty()) return;

  std::vector<std::string> custom_nodes = absl::StrSplit(node_names, ',');

  for (auto& cur_node : custom_nodes) {
    std::vector<std::string> node_infos = absl::StrSplit(cur_node, '=');
    std::string node_name = node_infos[0];
    auto node_specification = node_infos[1];
    CustomOpInfo new_node_info;
    switch (update_option) {
      case CustomOpUpdateOptions::kINputIndices: {
        std::vector<std::string> indices =
            absl::StrSplit(node_specification, '-');
        for (auto& cur_index : indices) {
          custom_op_map[node_name].quantizable_input_indices.push_back(
              std::stoi(cur_index));
        }
        break;
      }
      case CustomOpUpdateOptions::kWeightOnly:
        custom_op_map[node_name].is_weight_only =
            GetBooleanSpecs(node_specification);
        break;
      case CustomOpUpdateOptions::kNoSideEffect:
        custom_op_map[node_name].no_side_effect =
            GetBooleanSpecs(node_specification);
        break;
    }
  }
}

bool ParseInputNodeQuantSpecs(absl::string_view node_names,
                              absl::string_view min_values,
                              absl::string_view max_values,
                              absl::string_view inference_type,
                              QuantizationSpecs* quant_specs) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("node_names: \"" + std::string(node_names.data(), node_names.size()) + "\"");
   mht_2_v.push_back("min_values: \"" + std::string(min_values.data(), min_values.size()) + "\"");
   mht_2_v.push_back("max_values: \"" + std::string(max_values.data(), max_values.size()) + "\"");
   mht_2_v.push_back("inference_type: \"" + std::string(inference_type.data(), inference_type.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_configDTcc mht_2(mht_2_v, 275, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_config.cc", "ParseInputNodeQuantSpecs");

  std::vector<std::string> input_nodes = absl::StrSplit(node_names, ',');
  std::vector<llvm::Optional<double>> node_mins;
  if (!min_values.empty()) {
    std::vector<std::string> node_mins_str = absl::StrSplit(min_values, ',');
    for (int i = 0, e = node_mins_str.size(); i < e; i++) {
      double value;
      if (!absl::SimpleAtod(node_mins_str[i], &value)) {
        return true;
      }
      node_mins.push_back(value);
    }
  }

  std::vector<llvm::Optional<double>> node_maxs;
  if (!max_values.empty()) {
    std::vector<std::string> node_maxs_str = absl::StrSplit(max_values, ',');
    for (int i = 0, e = node_maxs_str.size(); i < e; i++) {
      double value;
      if (!absl::SimpleAtod(node_maxs_str[i], &value)) {
        llvm::errs() << "Unexpected mins: " << node_maxs_str[i] << "\n";
        return true;
      }
      node_maxs.push_back(value);
    }
  }

  tensorflow::DataType final_type = tensorflow::DT_FLOAT;
  if (!inference_type.empty() &&
      !DataType_Parse(std::string(inference_type), &final_type)) {
    return true;
  }
  return GetInputNodeQuantSpecs(input_nodes, node_mins, node_maxs, final_type,
                                quant_specs);
}

bool GetInputNodeQuantSpecs(
    const std::vector<std::string>& node_names,
    const std::vector<llvm::Optional<double>>& node_mins,
    const std::vector<llvm::Optional<double>>& node_maxs,
    tensorflow::DataType inference_type, QuantizationSpecs* quant_specs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePSquantizationPSquantization_configDTcc mht_3(mht_3_v, 318, "", "./tensorflow/compiler/mlir/lite/quantization/quantization_config.cc", "GetInputNodeQuantSpecs");

  quant_specs->inference_type = inference_type;

  // If min/max are not specified, just return;
  if (node_mins.empty() || node_maxs.empty()) return false;

  // Otherwise make sure min/max has the same size as inputs.
  if (IsQuantizationType(inference_type)) {
    // min/max should have same size as inputs, or shouldn't be specified.
    if (node_names.size() != node_mins.size() ||
        node_names.size() != node_maxs.size()) {
      return true;
    }
    for (int i = 0, e = node_names.size(); i != e; ++i) {
      quant_specs->input_ranges.push_back({node_mins[i], node_maxs[i]});
    }
    return false;
  }
  if (!node_mins.empty()) {
    llvm::dbgs() << "Ignored input_min_values.";
  }
  if (!node_maxs.empty()) {
    llvm::dbgs() << "Ignored input_max_values.";
  }
  return false;
}

}  // namespace quant
}  // namespace mlir
