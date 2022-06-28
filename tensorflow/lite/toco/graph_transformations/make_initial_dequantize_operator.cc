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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSmake_initial_dequantize_operatorDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSmake_initial_dequantize_operatorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSmake_initial_dequantize_operatorDTcc() {
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

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/graph_transformations/quantization_util.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/model_flags.pb.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

// This inserts an operator whose output is a float array (name:
// flags.input_array()).  It has to wait for any existing operators that
// generate this output to be removed by graph transformations.  Note that there
// may be more than one operator that takes the input_array as their input, and
// that some of these may be removed by graph transformations.
bool AddDequantizeOperatorToInput(const std::string& input_name,
                                  const Operator* op,
                                  GraphTransformation* transformation,
                                  Model* model) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("input_name: \"" + input_name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSmake_initial_dequantize_operatorDTcc mht_0(mht_0_v, 207, "", "./tensorflow/lite/toco/graph_transformations/make_initial_dequantize_operator.cc", "AddDequantizeOperatorToInput");

  // An operator with the required output may be a dequantize operator already
  // created.  Alternatively it may be an operator that needs to be removed
  // because it is unused, in which case we wait for RemoveUnusedOp to do its
  // work.
  if (GetOpWithOutput(*model, input_name)) {
    return false;
  }

  // We only apply for the first operator if there is more than one.  This is
  // not strictly necessary for ordering correctness, since we insert the
  // dequant operator at the beginning of the op sequence, but it makes the
  // insertion more predictable (eg forward vs backwards operator sweep).
  if (CountOpsWithInput(*model, input_name) > 1) {
    if (op != GetFirstOpWithInput(*model, input_name)) {
      return false;
    }
  }

  auto& input_array = model->GetArray(input_name);
  if (input_array.data_type != ArrayDataType::kFloat) {
    return false;
  }

  if (input_array.final_data_type == input_array.data_type ||
      input_array.final_data_type == ArrayDataType::kNone) {
    return false;
  }

  const auto& dequantized_input_name =
      AvailableArrayName(*model, input_name + "_dequantized");
  for (auto& other_op : model->operators) {
    for (std::string& other_op_input : other_op->inputs) {
      if (other_op_input == input_name) {
        other_op_input = dequantized_input_name;
      }
    }
  }

  auto& dequantized_input_array =
      model->GetOrCreateArray(dequantized_input_name);
  auto* image_input_op = new DequantizeOperator;
  image_input_op->inputs = {input_name};
  image_input_op->outputs = {dequantized_input_name};
  model->operators.emplace(model->operators.begin(), image_input_op);

  dequantized_input_array.data_type = ArrayDataType::kFloat;
  const auto& input_minmax = input_array.GetMinMax();
  auto& dequantized_input_minmax = dequantized_input_array.GetOrCreateMinMax();
  dequantized_input_minmax = input_minmax;
  auto& input_qparams = input_array.GetOrCreateQuantizationParams();
  input_array.data_type = input_array.final_data_type;
  ChooseQuantizationParamsForArrayAndQuantizedDataType(
      input_array, input_array.data_type, &input_qparams);

  transformation->AddMessageF(
      "Created %s"
      " to handle quantized input image data, taking over existing"
      " mean_value and std_value flags. Cleared those flags.",
      LogName(*image_input_op));

  return true;
}

::tensorflow::Status MakeInitialDequantizeOperator::Run(Model* model,
                                                        std::size_t op_index,
                                                        bool* modified) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSmake_initial_dequantize_operatorDTcc mht_1(mht_1_v, 276, "", "./tensorflow/lite/toco/graph_transformations/make_initial_dequantize_operator.cc", "MakeInitialDequantizeOperator::Run");

  *modified = false;
  // This is effectively a transformation applied to edges.  We iterate over the
  // specified node (op) and proceed for input edges.
  const auto it = model->operators.begin() + op_index;
  const auto* op = it->get();
  bool change_made = false;
  for (auto& input : op->inputs) {
    for (auto& input_array : *model->flags.mutable_input_arrays()) {
      if (input_array.name() == input) {
        if (AddDequantizeOperatorToInput(input_array.name(), op, this, model)) {
          change_made = true;
          input_array.clear_mean_value();
          input_array.clear_std_value();
        }
      }
    }
  }
  *modified = change_made;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
