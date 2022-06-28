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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSidentify_lstm_split_inputsDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSidentify_lstm_split_inputsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSidentify_lstm_split_inputsDTcc() {
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

::tensorflow::Status SplitLstmCellInputs::Run(Model* model,
                                              std::size_t op_index,
                                              bool* modified) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSidentify_lstm_split_inputsDTcc mht_0(mht_0_v, 199, "", "./tensorflow/lite/toco/graph_transformations/identify_lstm_split_inputs.cc", "SplitLstmCellInputs::Run");

  *modified = false;
  // Find lstm cell.
  auto op_it = model->operators.begin() + op_index;
  auto curr_op = op_it->get();
  if (curr_op->type != OperatorType::kLstmCell) {
    return ::tensorflow::Status::OK();
  }

  const auto* curr_lstm_op = static_cast<LstmCellOperator*>(curr_op);
  // Already an extended LstmCell. Do not need to split cell inputs.
  if (curr_lstm_op->kernel_type != LstmCellOperator::KERNEL_BASIC ||
      curr_lstm_op->inputs.size() != LstmCellOperator::NUM_INPUTS) {
    return ::tensorflow::Status::OK();
  }

  // Make sure the WEIGHTS_INPUT and BIASES_INPUT are constant arrays,
  // that are able to be split into smaller weight and bias tensors.
  if (!IsConstantParameterArray(
          *model, curr_op->inputs[LstmCellOperator::WEIGHTS_INPUT]) ||
      !IsConstantParameterArray(
          *model, curr_op->inputs[LstmCellOperator::BIASES_INPUT])) {
    return ::tensorflow::Status::OK();
  }

  // Make sure propagate_fixed_sizes has defined the size of the output.
  if (!model->GetArray(curr_op->outputs[LstmCellOperator::ACTIV_OUTPUT])
           .has_shape()) {
    return ::tensorflow::Status::OK();
  }

  // Emplace a new LstmCell operator with extended inputs (kernel/lstm.cc).
  auto lstm_cell_op = absl::make_unique<LstmCellOperator>();
  lstm_cell_op->kernel_type = LstmCellOperator::KERNEL_FULL;
  lstm_cell_op->inputs.resize(kExtendedLstmInputCount);
  int num_input = model->GetArray(curr_op->inputs[LstmCellOperator::DATA_INPUT])
                      .shape()
                      .dims(1);

  // n_cell and n_output have the same size when there is no projection.
  int num_cell =
      model->GetArray(curr_op->outputs[LstmCellOperator::ACTIV_OUTPUT])
          .shape()
          .dims(1);
  int num_output = num_cell;

  // Data input.
  lstm_cell_op->inputs[kInputTensor] =
      curr_op->inputs[LstmCellOperator::ACTIV_OUTPUT];

  // Previous states.
  lstm_cell_op->inputs[kInputActivationStateTensor] =
      curr_op->inputs[LstmCellOperator::PREV_ACTIV_INPUT];
  lstm_cell_op->inputs[kInputCellStateTensor] =
      curr_op->inputs[LstmCellOperator::PREV_STATE_INPUT];

  // Get original weight tensor and decompose 1 tensor to 8 sub tensors.
  Array& kernel =
      model->GetArray(curr_op->inputs[LstmCellOperator::WEIGHTS_INPUT]);
  const std::string base_name(FindLongestCommonPrefix(
      curr_op->outputs[LstmCellOperator::ACTIV_OUTPUT],
      curr_op->outputs[LstmCellOperator::STATE_OUTPUT]));

  // Input weight tensors of size {n_cell, n_input}.
  CopySubArrayToArray(
      model, &(lstm_cell_op->inputs[kInputToInputWeightsTensor]),
      base_name + "weight_i_i", num_cell, num_input, kernel, 0, 0);
  CopySubArrayToArray(model, &(lstm_cell_op->inputs[kInputToCellWeightsTensor]),
                      base_name + "weight_c_i", num_cell, num_input, kernel,
                      num_cell, 0);
  CopySubArrayToArray(
      model, &(lstm_cell_op->inputs[kInputToForgetWeightsTensor]),
      base_name + "weight_f_i", num_cell, num_input, kernel, num_cell * 2, 0);
  CopySubArrayToArray(
      model, &(lstm_cell_op->inputs[kInputToOutputWeightsTensor]),
      base_name + "weight_o_i", num_cell, num_input, kernel, num_cell * 3, 0);

  // Recurrent weight tensors of size {n_cell, n_output}.
  CopySubArrayToArray(
      model, &(lstm_cell_op->inputs[kRecurrentToInputWeightsTensor]),
      base_name + "weight_i_r", num_cell, num_output, kernel, 0, num_input);
  CopySubArrayToArray(model,
                      &(lstm_cell_op->inputs[kRecurrentToCellWeightsTensor]),
                      base_name + "weight_c_r", num_cell, num_output, kernel,
                      num_cell, num_input);
  CopySubArrayToArray(model,
                      &(lstm_cell_op->inputs[kRecurrentToForgetWeightsTensor]),
                      base_name + "weight_f_r", num_cell, num_output, kernel,
                      num_cell * 2, num_input);
  CopySubArrayToArray(model,
                      &(lstm_cell_op->inputs[kRecurrentToOutputWeightsTensor]),
                      base_name + "weight_o_r", num_cell, num_output, kernel,
                      num_cell * 3, num_input);

  // Peephole (optional).
  CreateOptionalArray(model, &(lstm_cell_op->inputs[kCellToInputWeightsTensor]),
                      base_name + "peephole_c_i");
  CreateOptionalArray(model,
                      &(lstm_cell_op->inputs[kCellToForgetWeightsTensor]),
                      base_name + "peephole_c_f");
  CreateOptionalArray(model,
                      &(lstm_cell_op->inputs[kCellToOutputWeightsTensor]),
                      base_name + "peephole_c_o");

  // Get original bias tensor and decompose 1 tensor to 4 sub tensors
  Array& bias =
      model->GetArray(curr_op->inputs[LstmCellOperator::BIASES_INPUT]);
  CopySubArrayToArray(model, &(lstm_cell_op->inputs[kInputGateBiasTensor]),
                      base_name + "bias_i", num_cell, 1, bias, 0, 0);
  CopySubArrayToArray(model, &(lstm_cell_op->inputs[kCellGateBiasTensor]),
                      base_name + "bias_c", num_cell, 1, bias, num_cell, 0);
  CopySubArrayToArray(model, &(lstm_cell_op->inputs[kForgetGateBiasTensor]),
                      base_name + "bias_f", num_cell, 1, bias, num_cell * 2, 0);
  CopySubArrayToArray(model, &(lstm_cell_op->inputs[kOutputGateBiasTensor]),
                      base_name + "bias_o", num_cell, 1, bias, num_cell * 3, 0);

  // Projection (optional).
  CreateOptionalArray(model, &(lstm_cell_op->inputs[kProjectionWeightsTensor]),
                      base_name + "proj_weight");
  CreateOptionalArray(model, &(lstm_cell_op->inputs[kProjectionBiasTensor]),
                      base_name + "proj_bias");

  // Reorder and resize LstmCell's outputs.
  lstm_cell_op->outputs.resize(
      ExtendedLstmCellOutputs::kExtendedLstmOutputCount);
  lstm_cell_op->outputs[kOutputStateTensor] =
      curr_op->outputs[LstmCellOperator::ACTIV_TEMP];
  lstm_cell_op->outputs[kCellStateTensor] =
      curr_op->outputs[LstmCellOperator::STATE_OUTPUT];
  lstm_cell_op->outputs[kOutputTensor] =
      curr_op->outputs[LstmCellOperator::ACTIV_OUTPUT];

  // Add the op into model.
  model->operators.emplace(op_it, std::move(lstm_cell_op));
  AddMessageF("Creating extended LstmCell replacing previous lstm cell");

  DeleteOpAndArrays(model, curr_op);

  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
