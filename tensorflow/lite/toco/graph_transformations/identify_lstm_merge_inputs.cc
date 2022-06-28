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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSidentify_lstm_merge_inputsDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSidentify_lstm_merge_inputsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSidentify_lstm_merge_inputsDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include <iostream>
#include <string>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/graph_transformations/lstm_utils.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

::tensorflow::Status MergeLstmCellInputs::Run(Model* model,
                                              std::size_t op_index,
                                              bool* modified) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSidentify_lstm_merge_inputsDTcc mht_0(mht_0_v, 199, "", "./tensorflow/lite/toco/graph_transformations/identify_lstm_merge_inputs.cc", "MergeLstmCellInputs::Run");

  *modified = false;
  // Find lstm cell.
  auto op_it = model->operators.begin() + op_index;
  auto src_op = op_it->get();
  if (src_op->type != OperatorType::kLstmCell) {
    return ::tensorflow::Status::OK();
  }

  // Already a compact LstmCell. Do not need to merge cell inputs.
  const auto* src_lstm_op = static_cast<LstmCellOperator*>(src_op);
  if (src_lstm_op->kernel_type != LstmCellOperator::KERNEL_FULL ||
      src_lstm_op->inputs.size() != kExtendedLstmInputCount) {
    return ::tensorflow::Status::OK();
  }

  // Identify prev_activ_input, prev_state_input as required Op inputs,
  // using the rnn_states in the model flag.
  std::string prev_activ_input;
  if (!GetMatchingRnnArray(model, src_op->outputs[kOutputTensor],
                           &prev_activ_input)) {
    return ::tensorflow::Status::OK();
  }
  std::string prev_state_input;
  if (!GetMatchingRnnArray(model, src_op->outputs[kCellStateTensor],
                           &prev_state_input)) {
    return ::tensorflow::Status::OK();
  }

  // Get LstmCell's cell, input, output size.
  int num_cell = model->GetArray(src_op->inputs[kInputToInputWeightsTensor])
                     .shape()
                     .dims(0);
  int num_input = model->GetArray(src_op->inputs[kInputToInputWeightsTensor])
                      .shape()
                      .dims(1);
  int num_output =
      model->GetArray(src_op->inputs[kRecurrentToInputWeightsTensor])
          .shape()
          .dims(1);

  // Make sure n_cell and n_output are equal as there is no projection.
  CHECK_EQ(num_cell, num_output);

  // Create tensorflow_graphdef style's one big weight tensor.
  const std::string base_name(FindLongestCommonPrefix(
      src_op->outputs[kOutputTensor], src_op->outputs[kCellStateTensor]));
  std::string merged_weights =
      AvailableArrayName(*model, base_name + "weights");
  auto& array = model->GetOrCreateArray(merged_weights);
  array.data_type = ArrayDataType::kFloat;
  int weights_dim1 = 4 * num_cell;
  int weights_dim2 = num_input + num_output;
  Shape shape = Shape({weights_dim1, weights_dim2});
  array.copy_shape(shape);
  auto& buffer = array.GetMutableBuffer<ArrayDataType::kFloat>();
  buffer.data.resize(weights_dim1 * weights_dim2);

  // Merge 8 small weight tensors to 1 weight tensor.
  CopyArrayToSubArray(
      buffer, weights_dim2,
      model->GetArray(src_op->inputs[kInputToInputWeightsTensor]), 0, 0);
  CopyArrayToSubArray(
      buffer, weights_dim2,
      model->GetArray(src_op->inputs[kInputToCellWeightsTensor]), num_cell, 0);
  CopyArrayToSubArray(
      buffer, weights_dim2,
      model->GetArray(src_op->inputs[kInputToForgetWeightsTensor]),
      num_cell * 2, 0);
  CopyArrayToSubArray(
      buffer, weights_dim2,
      model->GetArray(src_op->inputs[kInputToOutputWeightsTensor]),
      num_cell * 3, 0);
  CopyArrayToSubArray(
      buffer, weights_dim2,
      model->GetArray(src_op->inputs[kRecurrentToInputWeightsTensor]), 0,
      num_input);
  CopyArrayToSubArray(
      buffer, weights_dim2,
      model->GetArray(src_op->inputs[kRecurrentToCellWeightsTensor]), num_cell,
      num_input);
  CopyArrayToSubArray(
      buffer, weights_dim2,
      model->GetArray(src_op->inputs[kRecurrentToForgetWeightsTensor]),
      num_cell * 2, num_input);
  CopyArrayToSubArray(
      buffer, weights_dim2,
      model->GetArray(src_op->inputs[kRecurrentToOutputWeightsTensor]),
      num_cell * 3, num_input);

  // Create tensorflow_graphdef style's one big bias tensor.
  std::string merged_biases = AvailableArrayName(*model, base_name + "biases");
  auto& bias_array = model->GetOrCreateArray(merged_biases);
  bias_array.data_type = ArrayDataType::kFloat;
  bias_array.copy_shape(Shape({weights_dim1}));
  auto& bias_buffer = bias_array.GetMutableBuffer<ArrayDataType::kFloat>();
  bias_buffer.data.resize(weights_dim1);

  // Merge 4 small bias tensors into a big one.
  CopyArrayToSubArray(bias_buffer, weights_dim2,
                      model->GetArray(src_op->inputs[kInputGateBiasTensor]), 0,
                      0);
  CopyArrayToSubArray(bias_buffer, weights_dim2,
                      model->GetArray(src_op->inputs[kCellGateBiasTensor]),
                      num_cell, 0);
  CopyArrayToSubArray(bias_buffer, weights_dim2,
                      model->GetArray(src_op->inputs[kForgetGateBiasTensor]),
                      num_cell * 2, 0);
  CopyArrayToSubArray(bias_buffer, weights_dim2,
                      model->GetArray(src_op->inputs[kOutputGateBiasTensor]),
                      num_cell * 3, 0);

  // Emplace a new LSTM cell operator (use basic 5 inputs kernel).
  auto lstm_cell_op = absl::make_unique<LstmCellOperator>();
  lstm_cell_op->kernel_type = LstmCellOperator::KERNEL_BASIC;

  // Compact LstmCell's 5 inputs.
  lstm_cell_op->inputs.resize(LstmCellOperator::NUM_INPUTS);
  lstm_cell_op->inputs[LstmCellOperator::DATA_INPUT] =
      src_op->inputs[kInputTensor];
  lstm_cell_op->inputs[LstmCellOperator::WEIGHTS_INPUT] = merged_weights;
  lstm_cell_op->inputs[LstmCellOperator::BIASES_INPUT] = merged_biases;
  lstm_cell_op->inputs[LstmCellOperator::PREV_ACTIV_INPUT] = prev_activ_input;
  lstm_cell_op->inputs[LstmCellOperator::PREV_STATE_INPUT] = prev_state_input;

  // Reorder LstmCell's 3 outputs.
  lstm_cell_op->outputs.resize(LstmCellOperator::NUM_OUTPUTS);
  lstm_cell_op->outputs[LstmCellOperator::ACTIV_OUTPUT] =
      src_op->outputs[kOutputTensor];
  lstm_cell_op->outputs[LstmCellOperator::STATE_OUTPUT] =
      src_op->outputs[kCellStateTensor];
  lstm_cell_op->outputs[LstmCellOperator::ACTIV_TEMP] =
      src_op->outputs[kOutputStateTensor];
  // Create a new temp array for the fourth output.
  const std::string& concat_temp_array_name =
      AvailableArrayName(*model, base_name + "concat_temp");
  model->GetOrCreateArray(concat_temp_array_name);
  lstm_cell_op->outputs[LstmCellOperator::CONCAT_TEMP] = concat_temp_array_name;

  // Add the op into model.
  model->operators.emplace(op_it, std::move(lstm_cell_op));
  AddMessageF("Creating compact LstmCell replacing previous lstm cell");

  DeleteOpAndArrays(model, src_op);

  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
