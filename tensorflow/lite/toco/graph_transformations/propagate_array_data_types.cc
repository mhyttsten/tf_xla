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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSpropagate_array_data_typesDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSpropagate_array_data_typesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSpropagate_array_data_typesDTcc() {
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
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {
void SetDataTypeForAllOutputs(Model* model, Operator* op,
                              ArrayDataType data_type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSpropagate_array_data_typesDTcc mht_0(mht_0_v, 197, "", "./tensorflow/lite/toco/graph_transformations/propagate_array_data_types.cc", "SetDataTypeForAllOutputs");

  for (const auto& output : op->outputs) {
    model->GetArray(output).data_type = data_type;
  }
}
}  // namespace

::tensorflow::Status PropagateArrayDataTypes::Run(Model* model,
                                                  std::size_t op_index,
                                                  bool* modified) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSpropagate_array_data_typesDTcc mht_1(mht_1_v, 209, "", "./tensorflow/lite/toco/graph_transformations/propagate_array_data_types.cc", "PropagateArrayDataTypes::Run");

  *modified = false;
  auto it = model->operators.begin() + op_index;
  auto* op = it->get();

  // If the data type of some input is unknown, we need to yield.
  for (const auto& input : op->inputs) {
    if (!model->IsOptionalArray(input) &&
        model->GetArray(input).data_type == ArrayDataType::kNone) {
      return ::tensorflow::Status::OK();
    }
  }
  // Record data types of output before processing, so we can see at the
  // end if we changed anything, and return the correct boolean value.
  std::unordered_map<std::string, ArrayDataType> old_output_data_types;
  for (const auto& output : op->outputs) {
    old_output_data_types[output] = model->GetArray(output).data_type;
  }
  // Do the actual output data types propagation.
  switch (op->type) {
    case OperatorType::kDequantize:
      // These operators unconditionally produce float outputs
      SetDataTypeForAllOutputs(model, op, ArrayDataType::kFloat);
      break;
    case OperatorType::kLess:
    case OperatorType::kLessEqual:
    case OperatorType::kGreater:
    case OperatorType::kGreaterEqual:
    case OperatorType::kEqual:
    case OperatorType::kNotEqual:
    case OperatorType::kAny:
    case OperatorType::kLogicalAnd:
    case OperatorType::kLogicalNot:
    case OperatorType::kLogicalOr:
      // These operators unconditionally produce bool outputs
      SetDataTypeForAllOutputs(model, op, ArrayDataType::kBool);
      break;
    case OperatorType::kRank:
      // These operators only produce int32 outputs.
      SetDataTypeForAllOutputs(model, op, ArrayDataType::kInt32);
      break;
    case OperatorType::kShape: {
      // Shape op could produce int32 or int64 result. Set the output type
      // based on the `output_data_type` field.
      auto* shape_op = static_cast<TensorFlowShapeOperator*>(op);
      SetDataTypeForAllOutputs(model, op, shape_op->output_data_type);
      break;
    }
    case OperatorType::kSplit:
    case OperatorType::kConcat:
    case OperatorType::kFill: {
      // These operators produce an output with the same type as their 2nd input
      CHECK_GE(op->inputs.size(), 2);
      const ArrayDataType data_type = model->GetArray(op->inputs[1]).data_type;
      SetDataTypeForAllOutputs(model, op, data_type);
      break;
    }
    case OperatorType::kSplitV: {
      // These operators produce output with the same type as its 1st input
      CHECK_GE(op->inputs.size(), 3);
      const ArrayDataType data_type = model->GetArray(op->inputs[0]).data_type;
      SetDataTypeForAllOutputs(model, op, data_type);
      break;
    }
    case OperatorType::kTransposeConv: {
      // These operators produce an output with the same type as their 3rd input
      CHECK_GE(op->inputs.size(), 3);
      const ArrayDataType data_type = model->GetArray(op->inputs[2]).data_type;
      SetDataTypeForAllOutputs(model, op, data_type);
      break;
    }
    case OperatorType::kCast: {
      // Data type of the Cast op is specified.
      CHECK_EQ(op->outputs.size(), 1);
      auto* cast_op = static_cast<CastOperator*>(op);
      model->GetArray(op->outputs[0]).data_type = cast_op->dst_data_type;
      break;
    }
    case OperatorType::kArgMax: {
      // Data type of the ArgMax op is specified.
      CHECK_EQ(op->outputs.size(), 1);
      auto* argmax_op = static_cast<ArgMaxOperator*>(op);
      model->GetArray(op->outputs[0]).data_type = argmax_op->output_data_type;
      break;
    }
    case OperatorType::kArgMin: {
      // Data type of the ArgMin op is specified.
      CHECK_EQ(op->outputs.size(), 1);
      auto* argmin_op = static_cast<ArgMinOperator*>(op);
      model->GetArray(op->outputs[0]).data_type = argmin_op->output_data_type;
      break;
    }
    case OperatorType::kRange: {
      auto* range_op = static_cast<RangeOperator*>(op);
      // Output type of the Range op can be set via an attribute
      ArrayDataType data_type;
      if (range_op->dtype != ArrayDataType::kNone) {
        // Use the type if specified
        data_type = range_op->dtype;
      } else {
        // Otherwise use the first input
        CHECK_GE(op->inputs.size(), 1);
        data_type = model->GetArray(op->inputs[0]).data_type;
      }
      CHECK_EQ(op->outputs.size(), 1);
      SetDataTypeForAllOutputs(model, op, data_type);
      break;
    }
    case OperatorType::kRandomUniform: {
      auto* rand_op = static_cast<RandomUniformOperator*>(op);
      // The output type of RandomUniform is specified with an attribute
      if (rand_op->dtype == ArrayDataType::kNone) {
        return ::tensorflow::Status::OK();
      }
      CHECK_EQ(op->outputs.size(), 1);
      SetDataTypeForAllOutputs(model, op, rand_op->dtype);
      break;
    }
    case OperatorType::kTopK_V2: {
      // topk(values: T, k: int32) -> values: T, indices: int32
      CHECK_EQ(op->inputs.size(), 2);
      CHECK_EQ(op->outputs.size(), 2);
      CHECK(model->GetArray(op->inputs[1]).data_type == ArrayDataType::kInt32);
      model->GetArray(op->outputs[0]).data_type =
          model->GetArray(op->inputs[0]).data_type;
      model->GetArray(op->outputs[1]).data_type = ArrayDataType ::kInt32;
      break;
    }
    case OperatorType::kUnsupported: {
      auto* unsupported_op = static_cast<TensorFlowUnsupportedOperator*>(op);
      // Some output tensors from the op could be eliminated by optimization.
      // This can make unsupported_op->output_data_types have more elements than
      // op->outputs.
      if (unsupported_op->output_data_types.size() < op->outputs.size()) {
        return ::tensorflow::Status::OK();
      }
      for (size_t i = 0; i < op->outputs.size(); ++i) {
        const std::string& output = op->outputs[i];
        const ArrayDataType data_type = unsupported_op->output_data_types[i];
        model->GetArray(output).data_type = data_type;
      }
      break;
    }
    case OperatorType::kExpandDims: {
      // Yield on ExpandDim until it is converted to Reshape
      return ::tensorflow::Status::OK();
    }
    case OperatorType::kSelect: {
      // Select produces outputs with the same type as their 2nd input
      CHECK_EQ(op->inputs.size(), 3);
      const ArrayDataType data_type_x =
          model->GetArray(op->inputs[1]).data_type;
      const ArrayDataType data_type_y =
          model->GetArray(op->inputs[2]).data_type;
      CHECK(data_type_x == data_type_y);
      SetDataTypeForAllOutputs(model, op, data_type_x);
      break;
    }
    case OperatorType::kSparseToDense: {
      // Select produces outputs with the same type as their 3rd input
      CHECK_EQ(op->inputs.size(), 4);
      const ArrayDataType data_type = model->GetArray(op->inputs[2]).data_type;
      const ArrayDataType data_type_default =
          model->GetArray(op->inputs[3]).data_type;
      CHECK(data_type == data_type_default);
      SetDataTypeForAllOutputs(model, op, data_type);
      break;
    }
    case OperatorType::kPow: {
      CHECK_EQ(op->inputs.size(), 2);
      CHECK(model->GetArray(op->inputs[0]).data_type ==
            model->GetArray(op->inputs[1]).data_type);
      const ArrayDataType data_type = model->GetArray(op->inputs[0]).data_type;
      SetDataTypeForAllOutputs(model, op, data_type);
      break;
    }
    case OperatorType::kPack: {
      const ArrayDataType data_type = model->GetArray(op->inputs[0]).data_type;
      for (const auto& input : op->inputs) {
        CHECK(data_type == model->GetArray(input).data_type);
      }
      SetDataTypeForAllOutputs(model, op, data_type);
      break;
    }
    case OperatorType::kOneHot: {
      CHECK_EQ(op->inputs.size(), 4);
      CHECK_EQ(op->outputs.size(), 1);
      const ArrayDataType on_value_type =
          model->GetArray(op->inputs[OneHotOperator::ON_VALUE_INPUT]).data_type;
      const ArrayDataType off_value_type =
          model->GetArray(op->inputs[OneHotOperator::OFF_VALUE_INPUT])
              .data_type;
      CHECK(on_value_type == off_value_type);
      model->GetArray(op->outputs[0]).data_type = on_value_type;
      break;
    }
    case OperatorType::kCTCBeamSearchDecoder: {
      CHECK_EQ(op->inputs.size(), 2);
      // All outputs (sparse tensors) are int32s (although tf uses int64s)
      // except the last one (log probabilities) is float.
      const int output_size = op->outputs.size();
      for (int i = 0; i < output_size - 1; ++i) {
        model->GetArray(op->outputs[i]).data_type = ArrayDataType::kInt32;
      }
      model->GetArray(op->outputs[output_size - 1]).data_type =
          ArrayDataType::kFloat;
      break;
    }
    case OperatorType::kUnpack: {
      CHECK_EQ(op->inputs.size(), 1);
      const int output_size = op->outputs.size();
      for (int i = 0; i < output_size; ++i) {
        model->GetArray(op->outputs[i]).data_type =
            model->GetArray(op->inputs[0]).data_type;
      }
      break;
    }
    case OperatorType::kUnidirectionalSequenceLstm: {
      const ArrayDataType data_type = model->GetArray(op->inputs[0]).data_type;
      if (data_type != ArrayDataType::kFloat) return ::tensorflow::Status::OK();
      SetDataTypeForAllOutputs(model, op, data_type);
      break;
    }
    case OperatorType::kUnidirectionalSequenceRnn: {
      const ArrayDataType data_type = model->GetArray(op->inputs[0]).data_type;
      if (data_type != ArrayDataType::kFloat) return ::tensorflow::Status::OK();
      SetDataTypeForAllOutputs(model, op, data_type);
      break;
    }
    case OperatorType::kUnique: {
      CHECK_EQ(op->outputs.size(), 2);
      const UniqueOperator* unique_op = static_cast<UniqueOperator*>(op);
      const ArrayDataType data_type = model->GetArray(op->inputs[0]).data_type;
      model->GetArray(op->outputs[0]).data_type = data_type;
      model->GetArray(op->outputs[1]).data_type = unique_op->idx_out_type;
      break;
    }
    case OperatorType::kBidirectionalSequenceLstm: {
      const ArrayDataType data_type = model->GetArray(op->inputs[0]).data_type;
      if (data_type != ArrayDataType::kFloat) return ::tensorflow::Status::OK();
      SetDataTypeForAllOutputs(model, op, data_type);
      break;
    }
    case OperatorType::kBidirectionalSequenceRnn: {
      const ArrayDataType data_type = model->GetArray(op->inputs[0]).data_type;
      if (data_type != ArrayDataType::kFloat) return ::tensorflow::Status::OK();
      SetDataTypeForAllOutputs(model, op, data_type);
      break;
    }
    case OperatorType::kLstmCell: {
      // It's tricky to propagate data types through a LstmCell, as that has
      // multiple inputs and outputs, and there are quantized cases with
      // mixed (8bit vs 16bit) cases. Fortunately, that should never be needed,
      // as the data formats, such as TFLITE, that have LstmCell nodes, also
      // have data type fields for all their arrays.
      break;
    }
    case OperatorType::kMatrixDiag: {
      CHECK_EQ(op->inputs.size(), 1);
      CHECK_EQ(op->outputs.size(), 1);
      const ArrayDataType data_type = model->GetArray(op->inputs[0]).data_type;
      SetDataTypeForAllOutputs(model, op, data_type);
      break;
    }
    case OperatorType::kMatrixSetDiag: {
      CHECK_EQ(op->inputs.size(), 2);
      CHECK_EQ(op->outputs.size(), 1);
      const ArrayDataType data_type = model->GetArray(op->inputs[0]).data_type;
      SetDataTypeForAllOutputs(model, op, data_type);
      break;
    }
    default: {
      // These operators produce outputs with the same type as their 1st input
      CHECK_GT(op->inputs.size(), 0);
      const ArrayDataType data_type = model->GetArray(op->inputs[0]).data_type;
      SetDataTypeForAllOutputs(model, op, data_type);
      break;
    }
  }

  // Return true if any output data type changed, false if none changed.
  for (const auto& output : op->outputs) {
    if (old_output_data_types[output] != model->GetArray(output).data_type) {
      *modified = true;
      return ::tensorflow::Status::OK();
    }
  }
  return ::tensorflow::Status::OK();
}

}  // namespace toco
