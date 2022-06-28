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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_multiply_by_zeroDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_multiply_by_zeroDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_multiply_by_zeroDTcc() {
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
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"

namespace toco {

namespace {

template <typename T>
bool AreAllBufferElementsZero(const std::vector<T>& buffer_data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_multiply_by_zeroDTcc mht_0(mht_0_v, 199, "", "./tensorflow/lite/toco/graph_transformations/resolve_multiply_by_zero.cc", "AreAllBufferElementsZero");

  for (auto x : buffer_data) {
    if (x != T()) {
      return false;
    }
  }
  return true;
}

template <ArrayDataType Type>
void FillArrayWithZeros(Array* array) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_multiply_by_zeroDTcc mht_1(mht_1_v, 212, "", "./tensorflow/lite/toco/graph_transformations/resolve_multiply_by_zero.cc", "FillArrayWithZeros");

  CHECK(array->data_type == Type);
  std::vector<DataType<Type>>& data = array->GetMutableBuffer<Type>().data;
  data.resize(RequiredBufferSizeForShape(array->shape()));
  for (size_t i = 0; i < data.size(); i++) {
    data[i] = DataType<Type>();
  }
}

}  // namespace

// Removes a multiplication by array of constant zeros by making the output
// array to an array of constant zeros and removing the input arrays if they
// are no longer needed.
::tensorflow::Status ResolveMultiplyByZero::Run(Model* model,
                                                std::size_t op_index,
                                                bool* modified) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_multiply_by_zeroDTcc mht_2(mht_2_v, 231, "", "./tensorflow/lite/toco/graph_transformations/resolve_multiply_by_zero.cc", "ResolveMultiplyByZero::Run");

  *modified = false;
  const auto mul_it = model->operators.begin() + op_index;
  auto* mul_op = mul_it->get();
  if (mul_op->type != OperatorType::kMul) {
    return ::tensorflow::Status::OK();
  }
  const auto& output_array_name = mul_op->outputs[0];
  auto& output_array = model->GetArray(output_array_name);

  if (!IsDiscardableArray(*model, output_array_name)) {
    return ::tensorflow::Status::OK();
  }

  if (output_array.data_type == ArrayDataType::kNone) {
    // Yield until the output type has been set by PropagateArrayDataTypes
    return ::tensorflow::Status::OK();
  }

  // Yield if the output shape is not known yet.
  if (!output_array.has_shape()) {
    return ::tensorflow::Status::OK();
  }

  // This transformation only handles the case where one operand is all 0's and
  // the other is non-constant. Other cases are handled by constant propagation
  // or the trivial binary removal pass.
  const bool is_input_constant[2] = {
      IsConstantParameterArray(*model, mul_op->inputs[0]),
      IsConstantParameterArray(*model, mul_op->inputs[1]),
  };
  if (!is_input_constant[0] && !is_input_constant[1]) {
    // Neither input is constant, so nothing we can resolve here.
    return ::tensorflow::Status::OK();
  }
  if (is_input_constant[0] && is_input_constant[1]) {
    // Both inputs are constants. That's a job for constants propagation, not
    // for us to handle here.
    return ::tensorflow::Status::OK();
  }
  const int index_of_constant_input = is_input_constant[0] ? 0 : 1;
  const int index_of_variable_input = is_input_constant[0] ? 1 : 0;
  CHECK(is_input_constant[index_of_constant_input]);
  CHECK(!is_input_constant[index_of_variable_input]);

  const auto& constant_input_array =
      model->GetArray(mul_op->inputs[index_of_constant_input]);

  CHECK(constant_input_array.data_type == output_array.data_type);
  switch (output_array.data_type) {
    case ArrayDataType::kFloat: {
      const auto& constant_input_data =
          constant_input_array.GetBuffer<ArrayDataType::kFloat>().data;
      if (!AreAllBufferElementsZero<DataType<ArrayDataType::kFloat>>(
              constant_input_data)) {
        return ::tensorflow::Status::OK();
      }
      FillArrayWithZeros<ArrayDataType::kFloat>(&output_array);
    } break;
    case ArrayDataType::kUint8: {
      const auto& constant_input_data =
          constant_input_array.GetBuffer<ArrayDataType::kUint8>().data;
      if (!AreAllBufferElementsZero<DataType<ArrayDataType::kUint8>>(
              constant_input_data)) {
        return ::tensorflow::Status::OK();
      }
      FillArrayWithZeros<ArrayDataType::kUint8>(&output_array);
    } break;
    case ArrayDataType::kInt32: {
      const auto& constant_input_data =
          constant_input_array.GetBuffer<ArrayDataType::kInt32>().data;
      if (!AreAllBufferElementsZero<DataType<ArrayDataType::kInt32>>(
              constant_input_data)) {
        return ::tensorflow::Status::OK();
      }
      FillArrayWithZeros<ArrayDataType::kInt32>(&output_array);
    } break;
    case ArrayDataType::kInt64: {
      const auto& constant_input_data =
          constant_input_array.GetBuffer<ArrayDataType::kInt64>().data;
      if (!AreAllBufferElementsZero<DataType<ArrayDataType::kInt64>>(
              constant_input_data)) {
        return ::tensorflow::Status::OK();
      }
      FillArrayWithZeros<ArrayDataType::kInt64>(&output_array);
    } break;
    case ArrayDataType::kComplex64: {
      const auto& constant_input_data =
          constant_input_array.GetBuffer<ArrayDataType::kComplex64>().data;
      if (!AreAllBufferElementsZero<DataType<ArrayDataType::kComplex64>>(
              constant_input_data)) {
        return ::tensorflow::Status::OK();
      }
      FillArrayWithZeros<ArrayDataType::kComplex64>(&output_array);
    } break;
    default:
      AddMessageF(
          "Cannot resolve multiply by 0 because of unsupported data type\n");
      return ::tensorflow::Status::OK();
  }

  DeleteOpAndArrays(model, mul_op);
  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
