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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_concatenationDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_concatenationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_concatenationDTcc() {
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
#include "absl/strings/str_join.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

// Copies data from multiple source arrays to a destination array based on a
// concatenation dimension. From each array in input_arrays, it copies chunk
// sizes provided in array_copy_size vector (per array). It uses the buffer
// in concatenated_array as destination buffer.
template <ArrayDataType A, typename T>
void CopyTensorSegments(const std::vector<Array*>& input_arrays,
                        const std::vector<int>& array_copy_size,
                        const int num_elements_concatenated_array,
                        Array* concatenated_array) {
  for (Array* input_array : input_arrays) {
    if (!input_array->buffer) {
      return;
    }
  }

  auto& concatenated_array_buffer =
      concatenated_array->GetMutableBuffer<A>().data;
  concatenated_array_buffer.resize(num_elements_concatenated_array);

  // It does not matter which array to use to find the value for the total
  // number of copy steps.
  CHECK(!input_arrays.empty());
  CHECK_NE(array_copy_size[0], 0);
  const int total_copy_steps =
      input_arrays[0]->GetBuffer<A>().data.size() / array_copy_size[0];

  // Initialize the source pointers to point to beginning of the array buffers.
  std::vector<const T*> src_ptr;
  src_ptr.reserve(input_arrays.size());
  for (Array* input_array : input_arrays) {
    src_ptr.push_back(input_array->GetBuffer<A>().data.data());
  }

  // Copy the data from input_arrays to concatenated_array_buffer.
  T* dest_ptr = concatenated_array_buffer.data();
  for (int s = 0; s < total_copy_steps; s++) {
    for (size_t i = 0; i < input_arrays.size(); i++) {
      std::copy(src_ptr[i], src_ptr[i] + array_copy_size[i], dest_ptr);
      src_ptr[i] += array_copy_size[i];
      dest_ptr += array_copy_size[i];
    }
  }
}

// Receives a series of input arrays of type Array and an integer showing the
// axis on which those arrays will be concatenated. It returns the concatenated
// array.
template <ArrayDataType A>
void ConcatenateTensorBuffers(const std::vector<Array*>& input_arrays,
                              int concatenation_axis,
                              Array* concatenated_array) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_concatenationDTcc mht_0(mht_0_v, 250, "", "./tensorflow/lite/toco/graph_transformations/resolve_constant_concatenation.cc", "ConcatenateTensorBuffers");

  int num_elements_concatenated_array = 1;
  for (int i = 0; i < concatenated_array->shape().dimensions_count(); i++) {
    num_elements_concatenated_array *= concatenated_array->shape().dims()[i];
  }
  // Prepare the data needed for segmented copy from multiple source arrays to
  // a destination array based on a oncatenation dimension.
  std::vector<int> array_copy_size(input_arrays.size());
  int count = 0;
  for (Array* input_array : input_arrays) {
    const Shape array_shape = input_array->shape();
    array_copy_size[count] = 1;
    for (int i = concatenation_axis; i < array_shape.dimensions_count(); i++) {
      array_copy_size[count] *= array_shape.dims()[i];
    }
    count++;
  }

  // Do the actual data copy.
  CopyTensorSegments<A, DataType<A>>(input_arrays, array_copy_size,
                                     num_elements_concatenated_array,
                                     concatenated_array);
}

// Sets the minimum and maximum values for the concatenated array. If it's
// already set (e.g. because of previous pass in TOCO), it doesn't change it and
// returns. Otherwise it uses the input arrays min and max values to compute the
// concatenated array min and max.
void SetMinMaxForConcatenedArray(GraphTransformation* transformation,
                                 const std::vector<Array*>& input_arrays,
                                 Array* concatenated_array) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_concatenationDTcc mht_1(mht_1_v, 283, "", "./tensorflow/lite/toco/graph_transformations/resolve_constant_concatenation.cc", "SetMinMaxForConcatenedArray");

  CHECK(concatenated_array->data_type == ArrayDataType::kFloat);
  // If the minmax is already set, use it
  if (concatenated_array->minmax) return;

  double concat_min = std::numeric_limits<double>::infinity();
  double concat_max = -std::numeric_limits<double>::infinity();

  for (Array* input_array : input_arrays) {
    // If any of the input arrays minmax is not set,  return.
    // TODO(ghodrat): shall we add the logic to compute the minmax?
    if (!input_array->minmax) return;
    const MinMax& input_minmax = input_array->GetMinMax();
    concat_min = std::min(concat_min, input_minmax.min);
    concat_max = std::max(concat_max, input_minmax.max);
  }
  MinMax& minmax = concatenated_array->GetOrCreateMinMax();
  minmax.min = concat_min;
  minmax.max = concat_max;

  transformation->AddMessageF("Setting concatenated array min/max to %g,%g",
                              concat_min, concat_max);
}

}  // namespace

// Resolves the concatenation operator if all its inputs are constant arrays.
::tensorflow::Status ResolveConstantConcatenation::Run(Model* model,
                                                       std::size_t op_index,
                                                       bool* modified) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_concatenationDTcc mht_2(mht_2_v, 315, "", "./tensorflow/lite/toco/graph_transformations/resolve_constant_concatenation.cc", "ResolveConstantConcatenation::Run");

  *modified = false;
  const auto concat_it = model->operators.begin() + op_index;
  const auto* concat_base_op = concat_it->get();
  if (concat_base_op->type != OperatorType::kConcatenation) {
    return ::tensorflow::Status::OK();
  }
  const auto* concat_op =
      static_cast<const ConcatenationOperator*>(concat_base_op);

  for (const std::string& input_name : concat_op->inputs) {
    // We only expect constant unquantized arrays as input, otherwise we return.
    // We  also make sure the shapes of the input arrays are known and they are
    // all discardable.
    const Operator* input_op = GetOpWithOutput(*model, input_name);
    if (input_op) return ::tensorflow::Status::OK();
    if (!IsConstantParameterArray(*model, input_name))
      return ::tensorflow::Status::OK();
    if (!model->GetArray(input_name).has_shape())
      return ::tensorflow::Status::OK();
    if (model->GetArray(input_name).quantization_params)
      return ::tensorflow::Status::OK();
    if (!IsDiscardableArray(*model, input_name))
      return ::tensorflow::Status::OK();
  }

  const int concatenation_axis = concat_op->axis;

  CHECK_EQ(concat_op->outputs.size(), 1);
  std::string concatenated_array_name = concat_op->outputs[0];
  Array& concatenated_array = model->GetOrCreateArray(concatenated_array_name);
  std::vector<Array*> input_arrays;
  for (const std::string& input_name : concat_op->inputs) {
    input_arrays.push_back(&model->GetArray(input_name));
  }

  AddMessageF("Performing constant concat of %s into %s",
              absl::StrJoin(concat_op->inputs, ", "), concatenated_array_name);

  switch (concatenated_array.data_type) {
    case ArrayDataType::kFloat:
      ConcatenateTensorBuffers<ArrayDataType::kFloat>(
          input_arrays, concatenation_axis, &concatenated_array);
      SetMinMaxForConcatenedArray(this, input_arrays, &concatenated_array);
      break;
    case ArrayDataType::kUint8:
      ConcatenateTensorBuffers<ArrayDataType::kUint8>(
          input_arrays, concatenation_axis, &concatenated_array);
      break;
    case ArrayDataType::kInt32:
      ConcatenateTensorBuffers<ArrayDataType::kInt32>(
          input_arrays, concatenation_axis, &concatenated_array);
      break;
    case ArrayDataType::kInt64:
      ConcatenateTensorBuffers<ArrayDataType::kInt64>(
          input_arrays, concatenation_axis, &concatenated_array);
      break;
    case ArrayDataType::kString:
      ConcatenateTensorBuffers<ArrayDataType::kString>(
          input_arrays, concatenation_axis, &concatenated_array);
      break;
    case ArrayDataType::kComplex64:
      ConcatenateTensorBuffers<ArrayDataType::kComplex64>(
          input_arrays, concatenation_axis, &concatenated_array);
      break;
    default:
      LOG(FATAL) << "ArrayDataType not supported";
  }

  DeleteOpAndArrays(model, concat_op);
  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
