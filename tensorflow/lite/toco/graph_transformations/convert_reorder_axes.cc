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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSconvert_reorder_axesDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSconvert_reorder_axesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSconvert_reorder_axesDTcc() {
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

#include "absl/strings/str_cat.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

// Creates a Reshape operator from ReorderAxes operator.
TensorFlowReshapeOperator* CreateReshapeFromReorderAxes(
    Model* model, ReorderAxesOperator* reorder_op, const Shape& input_shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSconvert_reorder_axesDTcc mht_0(mht_0_v, 199, "", "./tensorflow/lite/toco/graph_transformations/convert_reorder_axes.cc", "CreateReshapeFromReorderAxes");

  auto* reshape_op = new TensorFlowReshapeOperator;

  // Copy inputs and outputs to Reshape.
  reshape_op->inputs.push_back(reorder_op->inputs[0]);
  reshape_op->outputs = reorder_op->outputs;

  // Create reshape dimensions based on input shape. Conversion from
  // ReorderAxes to Reshape requires a 4D input shape.
  CHECK_EQ(input_shape.dimensions_count(), 4);
  std::vector<int> reshape_dims = {1, input_shape.dims(0), input_shape.dims(1),
                                   input_shape.dims(3) * input_shape.dims(2)};

  // Create a new input array for Reshape.
  std::string reshape_array_name =
      AvailableArrayName(*model, reshape_op->outputs[0]);
  reshape_op->inputs.push_back(reshape_array_name);

  Array& reshape_array = model->GetOrCreateArray(reshape_array_name);
  *(reshape_array.mutable_shape()->mutable_dims()) = {
      1, static_cast<int>(reshape_dims.size())};
  reshape_array.data_type = ArrayDataType::kInt32;
  auto& reshape_buffer =
      reshape_array.GetMutableBuffer<ArrayDataType::kInt32>();
  reshape_buffer.data = reshape_dims;

  return reshape_op;
}

// Creates a Transpose operator from ReorderAxes operator.
TransposeOperator* CreateTransposeFromReorderAxes(
    Model* model, ReorderAxesOperator* reorder_op, const Shape& input_shape,
    const AxesOrder& input_axes_order, const AxesOrder& output_axes_order) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSconvert_reorder_axesDTcc mht_1(mht_1_v, 234, "", "./tensorflow/lite/toco/graph_transformations/convert_reorder_axes.cc", "CreateTransposeFromReorderAxes");

  auto* transpose_op = new TransposeOperator;

  // Copy inputs and outputs to Transpose.
  transpose_op->inputs.push_back(reorder_op->inputs[0]);
  transpose_op->outputs = reorder_op->outputs;

  // Create permutations data based on input and output axes order.
  std::vector<int> permutations_data;
  GetShuffleShape(input_axes_order, output_axes_order, &permutations_data);

  // Create a new input permutations array for Transpose.
  std::string perm_array_name =
      AvailableArrayName(*model, transpose_op->outputs[0]);
  transpose_op->inputs.push_back(perm_array_name);

  Array& perm_array = model->GetOrCreateArray(perm_array_name);
  *(perm_array.mutable_shape()->mutable_dims()) = {
      static_cast<int>(permutations_data.size())};
  perm_array.data_type = ArrayDataType::kInt32;
  auto& perm_buffer = perm_array.GetMutableBuffer<ArrayDataType::kInt32>();
  perm_buffer.data = permutations_data;

  return transpose_op;
}

// Converts ReorderAxes into Transpose and Reshape which are compatible with the
// TFLite interpreter.
::tensorflow::Status ConvertReorderAxes::Run(Model* model, std::size_t op_index,
                                             bool* modified) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSconvert_reorder_axesDTcc mht_2(mht_2_v, 266, "", "./tensorflow/lite/toco/graph_transformations/convert_reorder_axes.cc", "ConvertReorderAxes::Run");

  *modified = false;
  auto reorder_it = model->operators.begin() + op_index;
  if (reorder_it->get()->type != OperatorType::kReorderAxes)
    return ::tensorflow::Status::OK();

  auto* reorder_op = static_cast<ReorderAxesOperator*>(reorder_it->get());
  CHECK_EQ(reorder_op->inputs.size(), 1);
  CHECK_EQ(reorder_op->outputs.size(), 1);

  const auto& input_array_name = reorder_op->inputs[0];
  const auto& output_array_name = reorder_op->outputs[0];
  auto& input_array = model->GetArray(input_array_name);
  auto& output_array = model->GetArray(output_array_name);

  // Get input array. If kFakeQuant is the input into ReorderAxes, get the input
  // array passed into kFakeQuant. kFakeQuant op is dropped when possible.
  std::string constant_input_array_name = input_array_name;
  if (!input_array.buffer) {
    const auto* op_producing_input = GetOpWithOutput(*model, input_array_name);
    if (op_producing_input &&
        op_producing_input->type == OperatorType::kFakeQuant) {
      constant_input_array_name = op_producing_input->inputs[0];
    }
  }

  // Yield if input array contains constants or if output array size has not
  // been adjusted to reflect the permutations in ReorderAxes. ReorderAxes will
  // be merged into a constant array when possible.
  if (IsConstantParameterArray(*model, constant_input_array_name))
    return ::tensorflow::Status::OK();
  if (!output_array.has_shape()) return ::tensorflow::Status::OK();

  const auto input_axes_order = reorder_op->input_axes_order;
  const auto output_axes_order = reorder_op->output_axes_order;
  const Shape input_shape = input_array.shape();

  // Creates a Reshape or Transpose operator depending on the conversion.
  if (input_axes_order == AxesOrder::kHWIM &&
      output_axes_order == AxesOrder::k1HWO) {
    // Add Reshape operator into the graph. This special case is not just a
    // permutation. The input dimensions get merged into 3 dimensions while the
    // order of the elements does not change.
    auto* reshape_op =
        CreateReshapeFromReorderAxes(model, reorder_op, input_shape);
    model->operators.emplace(reorder_it, reshape_op);
  } else {
    // Add Transpose operator into the graph.
    auto* transpose_op = CreateTransposeFromReorderAxes(
        model, reorder_op, input_shape, input_axes_order, output_axes_order);
    model->operators.emplace(reorder_it, transpose_op);
  }

  // Remove ReorderAxes operator from the graph.
  DeleteOpAndArrays(model, reorder_op);

  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
