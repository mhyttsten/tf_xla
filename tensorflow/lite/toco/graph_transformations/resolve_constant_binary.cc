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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_binaryDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_binaryDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_binaryDTcc() {
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
#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/runtime/types.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

std::vector<bool> VectorGreaterThan(const std::vector<int>& a,
                                    const std::vector<int>& b) {
  DCHECK_EQ(a.size(), b.size());
  const int size = a.size();
  std::vector<bool> result(size);
  for (int i = 0; i < size; i++) {
    result[i] = a[i] > b[i];
  }
  return result;
}

void PairwiseVectorSelect(const std::vector<bool>& selector,
                          const std::vector<int>& input_a,
                          const std::vector<int>& input_b,
                          std::vector<int>* output_a,
                          std::vector<int>* output_b) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_binaryDTcc mht_0(mht_0_v, 216, "", "./tensorflow/lite/toco/graph_transformations/resolve_constant_binary.cc", "PairwiseVectorSelect");

  DCHECK_EQ(input_a.size(), input_b.size());
  DCHECK_EQ(output_a->size(), output_b->size());
  DCHECK_EQ(input_a.size(), output_a->size());
  DCHECK_EQ(selector.size(), input_a.size());
  const int size = input_a.size();
  for (int i = 0; i < size; i++) {
    if (selector[i]) {
      (*output_a)[i] = input_a[i];
      (*output_b)[i] = input_b[i];
    } else {
      (*output_a)[i] = input_b[i];
      (*output_b)[i] = input_a[i];
    }
  }
}

template <ArrayDataType InputsDataType, ArrayDataType OutputDataType>
void EvaluateBinaryOperatorOnConstantInputs(Model* model,
                                            const Operator* binary_op) {
  CHECK(IsConstantParameterArray(*model, binary_op->inputs[0]));
  CHECK(IsConstantParameterArray(*model, binary_op->inputs[1]));
  CHECK(binary_op->fused_activation_function ==
        FusedActivationFunctionType::kNone);
  const auto& input0_array = model->GetArray(binary_op->inputs[0]);
  const auto& input1_array = model->GetArray(binary_op->inputs[1]);
  const auto& output_name = binary_op->outputs[0];
  auto& output_array = model->GetArray(output_name);
  CHECK(input0_array.data_type == InputsDataType);
  CHECK(input1_array.data_type == InputsDataType);
  CHECK(output_array.data_type == OutputDataType);

  // We have already tested above for existence of input buffers
  // (synonymous to being a constant param).
  CHECK(input0_array.buffer);
  CHECK(input1_array.buffer);
  // On the other hand, the output should not already have a buffer.
  CHECK(!output_array.buffer);

  const auto& input0_data = input0_array.GetBuffer<InputsDataType>().data;
  const auto& input1_data = input1_array.GetBuffer<InputsDataType>().data;
  // Create the buffer on the output array, effectively turning it into
  // a constant parameter

  const Shape& output_shape = output_array.shape();
  auto& output_data = output_array.GetMutableBuffer<OutputDataType>().data;
  const int output_buffer_size = RequiredBufferSizeForShape(output_shape);
  output_data.resize(output_buffer_size);
  const int dims_count = output_shape.dimensions_count();

  // It will be convenient here to have copies of the operands shapes
  // extended to match the number of dimensions of the output shape.
  Shape input0_shape = input0_array.shape();
  Shape input1_shape = input1_array.shape();
  ExtendShape(&input0_shape, dims_count);
  ExtendShape(&input1_shape, dims_count);
  // Now we may still have operands of different sizes, which would indicate
  // that we have to "broadcast" the smaller dimension.  We do this using a
  // a vector of Booleans indicating which input is the larger in each
  // dimension.
  CHECK_EQ(input0_shape.dimensions_count(), input1_shape.dimensions_count());
  CHECK_EQ(input0_shape.dimensions_count(), dims_count);
  const std::vector<bool> input0_larger =
      VectorGreaterThan(input0_shape.dims(), input1_shape.dims());

  std::vector<int> big_sizes(dims_count);
  std::vector<int> small_sizes(dims_count);
  PairwiseVectorSelect(input0_larger, input0_shape.dims(), input1_shape.dims(),
                       &big_sizes, &small_sizes);

  // The output should already be correctly sized to match the big dimensions.
  for (int i = 0; i < dims_count; i++) {
    CHECK_EQ(output_shape.dims(i), big_sizes[i]);
  }

  std::vector<int> input0_indices(dims_count);
  std::vector<int> input1_indices(dims_count);
  std::vector<int> modulo_indices(dims_count);

  for (int k = 0; k < output_buffer_size; k++) {
    const std::vector<int> output_indices = ReverseOffset(output_shape, k);
    for (int i = 0; i < dims_count; i++) {
      modulo_indices[i] = output_indices[i] % small_sizes[i];
    }
    PairwiseVectorSelect(input0_larger, output_indices, modulo_indices,
                         &input0_indices, &input1_indices);
    const auto val0 = input0_data[Offset(input0_shape, input0_indices)];
    const auto val1 = input1_data[Offset(input1_shape, input1_indices)];

    DataType<OutputDataType> outval;
    if (binary_op->type == OperatorType::kAdd) {
      outval = val0 + val1;
    } else if (binary_op->type == OperatorType::kMul) {
      outval = val0 * val1;
    } else if (binary_op->type == OperatorType::kSub) {
      outval = val0 - val1;
    } else if (binary_op->type == OperatorType::kDiv) {
      outval = val0 / val1;
    } else if (binary_op->type == OperatorType::kFloorDiv) {
      outval = std::floor(val0 / val1);
    } else if (binary_op->type == OperatorType::kFloorMod) {
      outval = val0 - (std::floor(val0 / val1) * val1);
    } else if (binary_op->type == OperatorType::kMinimum) {
      outval = std::min(val0, val1);
    } else if (binary_op->type == OperatorType::kMaximum) {
      outval = std::max(val0, val1);
    } else if (binary_op->type == OperatorType::kLess) {
      outval = val0 < val1;
    } else if (binary_op->type == OperatorType::kLessEqual) {
      outval = val0 <= val1;
    } else if (binary_op->type == OperatorType::kGreater) {
      outval = val0 > val1;
    } else if (binary_op->type == OperatorType::kGreaterEqual) {
      outval = val0 >= val1;
    } else {
      LOG(FATAL) << "should not get here";
    }
    output_data[Offset(output_shape, output_indices)] = outval;
  }
}

bool EvaluateBinaryOperatorOnConstantInputs(Model* model,
                                            const Operator* binary_op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_binaryDTcc mht_1(mht_1_v, 341, "", "./tensorflow/lite/toco/graph_transformations/resolve_constant_binary.cc", "EvaluateBinaryOperatorOnConstantInputs");

  const auto inputs_data_type = model->GetArray(binary_op->inputs[0]).data_type;
  const auto output_data_type =
      model->GetArray(binary_op->outputs[0]).data_type;
#define TOCO_HANDLE_CASE(InputsDataType, OutputDataType)                    \
  if (inputs_data_type == InputsDataType &&                                 \
      output_data_type == OutputDataType) {                                 \
    EvaluateBinaryOperatorOnConstantInputs<InputsDataType, OutputDataType>( \
        model, binary_op);                                                  \
    return true;                                                            \
  }
  TOCO_HANDLE_CASE(ArrayDataType::kFloat, ArrayDataType::kFloat)
  TOCO_HANDLE_CASE(ArrayDataType::kFloat, ArrayDataType::kBool)
  TOCO_HANDLE_CASE(ArrayDataType::kInt32, ArrayDataType::kInt32)
  TOCO_HANDLE_CASE(ArrayDataType::kInt32, ArrayDataType::kBool)
  TOCO_HANDLE_CASE(ArrayDataType::kInt64, ArrayDataType::kInt64)
  TOCO_HANDLE_CASE(ArrayDataType::kInt64, ArrayDataType::kBool)
  return false;
#undef TOCO_HANDLE_CASE
}
}  // namespace

::tensorflow::Status ResolveConstantBinaryOperator::Run(Model* model,
                                                        std::size_t op_index,
                                                        bool* modified) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_binaryDTcc mht_2(mht_2_v, 368, "", "./tensorflow/lite/toco/graph_transformations/resolve_constant_binary.cc", "ResolveConstantBinaryOperator::Run");

  *modified = false;
  const auto binary_it = model->operators.begin() + op_index;
  const auto* binary_op = binary_it->get();
  // Test for binary ops of types that we know how to resolve
  if (binary_op->type != OperatorType::kAdd &&
      binary_op->type != OperatorType::kMul &&
      binary_op->type != OperatorType::kSub &&
      binary_op->type != OperatorType::kDiv &&
      binary_op->type != OperatorType::kFloorDiv &&
      binary_op->type != OperatorType::kFloorMod &&
      binary_op->type != OperatorType::kMinimum &&
      binary_op->type != OperatorType::kMaximum &&
      binary_op->type != OperatorType::kLess &&
      binary_op->type != OperatorType::kLessEqual &&
      binary_op->type != OperatorType::kGreater &&
      binary_op->type != OperatorType::kGreaterEqual) {
    return ::tensorflow::Status::OK();
  }
  CHECK_EQ(binary_op->inputs.size(), 2);

  const auto& input0_array = model->GetArray(binary_op->inputs[0]);
  const auto& input1_array = model->GetArray(binary_op->inputs[1]);
  // Check if both inputs are constant parameters.
  if (!input0_array.buffer || !input1_array.buffer) {
    return ::tensorflow::Status::OK();
  }

  auto& output_array = model->GetArray(binary_op->outputs[0]);
  // Yield until the output array dims have been resolved.
  if (!output_array.has_shape()) {
    return ::tensorflow::Status::OK();
  }

  // At the moment we don't want to care about fused activation functions.
  // The idea is that we should do the present constants-propagation before
  // activation functions get fused.
  if (binary_op->fused_activation_function !=
      FusedActivationFunctionType::kNone) {
    AddMessageF(
        "Not resolving constant %s because it has a fused activation function",
        LogName(*binary_op));
    return ::tensorflow::Status::OK();
  }

  // Check that input data types agree.
  CHECK(input0_array.data_type == input1_array.data_type)
      << "Dissimilar data types given to op outputting \""
      << binary_op->outputs[0] << "\". 0:\"" << binary_op->inputs[0] << "\"("
      << static_cast<int>(input0_array.data_type) << ")   1:\""
      << binary_op->inputs[1] << "\"("
      << static_cast<int>(input1_array.data_type) << ").";

  // Do the actual constants propagation
  if (!EvaluateBinaryOperatorOnConstantInputs(model, binary_op)) {
    return ::tensorflow::Status::OK();
  }

  DeleteOpAndArrays(model, binary_op);
  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
