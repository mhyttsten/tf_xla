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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_transposeDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_transposeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_transposeDTcc() {
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
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

// Transposes an array up to rank 4.
// This is ShuffleArrayTemplate with non-enum permutation.
template <ArrayDataType Type>
void Transpose(Model* model, const Array& input_array,
               const std::vector<int>& perm, Array* output_array) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_transposeDTcc mht_0(mht_0_v, 199, "", "./tensorflow/lite/toco/graph_transformations/resolve_constant_transpose.cc", "Transpose");

  const Shape& input_shape = input_array.shape();
  const std::vector<DataType<Type>>& input_data =
      input_array.GetBuffer<Type>().data;

  const Shape& output_shape = output_array->shape();
  std::vector<DataType<Type>>& output_data =
      output_array->GetMutableBuffer<Type>().data;
  output_data.resize(RequiredBufferSizeForShape(output_shape));

  CHECK(input_shape.dimensions_count() == output_shape.dimensions_count());
  const int dim = input_shape.dimensions_count();
  CHECK_LE(dim, 4);
  CHECK(static_cast<int>(perm.size()) >= dim);
  for (int i = 0; i < dim; i++) {
    CHECK(perm[i] >= 0 && perm[i] < dim);
    CHECK(input_shape.dims(perm[i]) == output_shape.dims(i));
  }
  Shape extended_input_shape = input_shape;
  ExtendShape(&extended_input_shape, 4);
  Shape extended_output_shape = output_shape;
  ExtendShape(&extended_output_shape, 4);
  std::vector<int> extended_perm;
  ExtendShuffle(perm, 4, &extended_perm);

  const std::vector<int>& extended_input_dims = extended_input_shape.dims();
  const std::vector<int>& extended_output_dims = extended_output_shape.dims();

  // TODO(starka): Rework to handle different numbers of dimensions.
  int input_strides[4];
  input_strides[3] = 1;
  input_strides[2] = extended_input_dims[3];
  input_strides[1] = input_strides[2] * extended_input_dims[2];
  input_strides[0] = input_strides[1] * extended_input_dims[1];
  const int input_stride_0 = input_strides[extended_perm[3]];
  const int input_stride_1 = input_strides[extended_perm[2]];
  const int input_stride_2 = input_strides[extended_perm[1]];
  const int input_stride_3 = input_strides[extended_perm[0]];

  const int output_size_0 = extended_output_dims[3];
  const int output_size_1 = extended_output_dims[2];
  const int output_size_2 = extended_output_dims[1];
  const int output_size_3 = extended_output_dims[0];
  const int output_stride_0 = 1;
  const int output_stride_1 = output_size_0;
  const int output_stride_2 = output_stride_1 * output_size_1;
  const int output_stride_3 = output_stride_2 * output_size_2;

  for (int i3 = 0; i3 < output_size_3; i3++) {
    const DataType<Type>* const input_ptr_3 =
        input_data.data() + i3 * input_stride_3;
    DataType<Type>* const output_ptr_3 =
        output_data.data() + i3 * output_stride_3;
    for (int i2 = 0; i2 < output_size_2; i2++) {
      const DataType<Type>* const input_ptr_2 =
          input_ptr_3 + i2 * input_stride_2;
      DataType<Type>* const output_ptr_2 = output_ptr_3 + i2 * output_stride_2;
      for (int i1 = 0; i1 < output_size_1; i1++) {
        const DataType<Type>* input_ptr = input_ptr_2 + i1 * input_stride_1;
        DataType<Type>* output_ptr = output_ptr_2 + i1 * output_stride_1;
        DataType<Type>* const output_ptr_end =
            output_ptr + output_size_0 * output_stride_0;
        while (output_ptr != output_ptr_end) {
          *output_ptr = *input_ptr;
          input_ptr += input_stride_0;
          output_ptr += output_stride_0;
        }
      }
    }
  }
}

}  // namespace

::tensorflow::Status ResolveConstantTranspose::Run(Model* model,
                                                   std::size_t op_index,
                                                   bool* modified) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_transposeDTcc mht_1(mht_1_v, 278, "", "./tensorflow/lite/toco/graph_transformations/resolve_constant_transpose.cc", "ResolveConstantTranspose::Run");

  *modified = false;
  auto it = model->operators.begin() + op_index;
  const auto* base_op = it->get();
  if (base_op->type != OperatorType::kTranspose) {
    return ::tensorflow::Status::OK();
  }
  const auto* op = static_cast<const TransposeOperator*>(base_op);

  CHECK_EQ(op->inputs.size(), 2);
  CHECK_EQ(op->outputs.size(), 1);
  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.data_type == ArrayDataType::kNone) {
    // Yield until the output type has been set by PropagateArrayDataTypes.
    return ::tensorflow::Status::OK();
  }
  if (!output_array.has_shape()) {
    // Yield until the output shape has been set by PropagateFixedShapes.
    return ::tensorflow::Status::OK();
  }

  // We require constant inputs.
  if (!IsConstantParameterArray(*model, op->inputs[0]) ||
      !IsConstantParameterArray(*model, op->inputs[1])) {
    return ::tensorflow::Status::OK();
  }
  const Array& input_array = model->GetArray(op->inputs[0]);

  CopyMinMaxAndQuantizationRelatedFields(input_array, &output_array);

  if (op->perm.empty()) {
    // Yield until perm has been populated by ResolveTransposeAttributes.
    return ::tensorflow::Status::OK();
  }

  // We currently only support 1-4 dimensions.
  CHECK_LE(op->perm.size(), 4);

  CHECK(!output_array.buffer);
  switch (output_array.data_type) {
    case ArrayDataType::kFloat:
      Transpose<ArrayDataType::kFloat>(model, input_array, op->perm,
                                       &output_array);
      break;
    case ArrayDataType::kUint8:
      Transpose<ArrayDataType::kUint8>(model, input_array, op->perm,
                                       &output_array);
      break;
    case ArrayDataType::kInt32:
      Transpose<ArrayDataType::kInt32>(model, input_array, op->perm,
                                       &output_array);
      break;
    case ArrayDataType::kInt64:
      Transpose<ArrayDataType::kInt64>(model, input_array, op->perm,
                                       &output_array);
      break;
    case ArrayDataType::kComplex64:
      Transpose<ArrayDataType::kComplex64>(model, input_array, op->perm,
                                           &output_array);
      break;
    default:
      LOG(FATAL) << "Unsupported data type given to Transpose op with output \""
                 << op->outputs[0] << "\"";
      break;
  }

  AddMessageF("Resolving constant transpose of %s", LogName(*op));

  DeleteOpAndArrays(model, op);
  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
