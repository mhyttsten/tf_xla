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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_strided_sliceDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_strided_sliceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_strided_sliceDTcc() {
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
#include <vector>

#include "tensorflow/lite/kernels/internal/strided_slice_logic.h"
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

template <ArrayDataType Type>
void StridedSlice(StridedSliceOperator const& op, Array const& input_array,
                  Array* output_array) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_strided_sliceDTcc mht_0(mht_0_v, 198, "", "./tensorflow/lite/toco/graph_transformations/resolve_constant_strided_slice.cc", "StridedSlice");

  // The TensorFlow documentation for StridedSlice is a bit ambiguous in places
  // (https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/strided-slice).
  // Use the source code at /third_party/tensorflow/core/util/strided_op.cc as
  // "master documentation".

  CHECK(input_array.data_type == Type);
  CHECK(output_array->data_type == Type);
  CHECK_EQ(op.ellipsis_mask, 0);
  CHECK_EQ(op.new_axis_mask, 0);

  int num_input_axes = op.start_indices.size();
  CHECK_EQ(num_input_axes, op.start_indices.size());
  CHECK_EQ(num_input_axes, op.stop_indices.size());
  CHECK_EQ(num_input_axes, op.strides.size());

  // Create a buffer for the output array
  std::vector<DataType<Type>>& output_data =
      output_array->GetMutableBuffer<Type>().data;
  output_data.resize(RequiredBufferSizeForShape(output_array->shape()));

  // Initialize source coordinate
  Shape const& input_shape = input_array.shape();
  Buffer<Type> const& input_buffer = input_array.GetBuffer<Type>();
  std::vector<int> src_coord(num_input_axes);
  std::vector<int> stop_for_axis(num_input_axes);
  const auto strided_slice_params =
      tflite::strided_slice::BuildStridedSliceParams(
          op.begin_mask, op.end_mask, op.shrink_axis_mask, op.start_indices,
          op.stop_indices, op.strides);

  for (int axis = 0; axis < num_input_axes; axis++) {
    int start_index = tflite::strided_slice::StartForAxis(
        strided_slice_params, ToRuntimeShape(input_array.shape()), axis);
    src_coord[axis] = start_index;
    stop_for_axis[axis] = tflite::strided_slice::StopForAxis(
        strided_slice_params, ToRuntimeShape(input_array.shape()), axis,
        start_index);
  }

  // In order to handle any number (N) of dimensions, we copy elements one by
  // one and treat the source coordinate as an N digit number (src_coord here).
  // Each "digit" is incremented individually (by the stride). When it overflows
  // (becomes greater than the stop), that digit is reset and a carry flag is
  // used to increment the next digit.
  for (size_t dst_offset = 0; dst_offset < output_data.size(); ++dst_offset) {
    // Copy element.
    output_data[dst_offset] = input_buffer.data[Offset(input_shape, src_coord)];

    // Note we consider elements in the highest dimension are stored
    // contiguously. So, we increment the stride starting from the highest
    // dimension.
    for (int axis = num_input_axes - 1; axis >= 0; --axis) {
      int stride = op.strides[axis];
      src_coord[axis] += stride;

      // Check if we've overflowed. If not, we just break from the loop to
      // continue w/ the element copy. Otherwise, reset the starting coordinate
      // for this axis and move to the next lower axis.
      int stop = stop_for_axis[axis];
      if (!tflite::strided_slice::LoopCondition(src_coord[axis], stop,
                                                stride)) {
        break;
      }
      src_coord[axis] = tflite::strided_slice::StartForAxis(
          strided_slice_params, ToRuntimeShape(input_shape), axis);
    }
  }
}

}  // anonymous namespace

::tensorflow::Status ResolveConstantStridedSlice::Run(Model* model,
                                                      std::size_t op_index,
                                                      bool* modified) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_constant_strided_sliceDTcc mht_1(mht_1_v, 275, "", "./tensorflow/lite/toco/graph_transformations/resolve_constant_strided_slice.cc", "ResolveConstantStridedSlice::Run");

  *modified = false;
  const auto it = model->operators.begin() + op_index;
  const auto* base_op = it->get();
  if (base_op->type != OperatorType::kStridedSlice) {
    return ::tensorflow::Status::OK();
  }

  const StridedSliceOperator* op =
      static_cast<const StridedSliceOperator*>(base_op);

  CHECK_EQ(op->outputs.size(), 1);
  auto& output_array = model->GetArray(op->outputs[0]);
  if (output_array.data_type == ArrayDataType::kNone) {
    // Yield until the output type has been set by PropagateArrayDataTypes
    return ::tensorflow::Status::OK();
  }

  if (!output_array.has_shape()) {
    // Yield until the output shape has been set by PropagateFixedShapes
    return ::tensorflow::Status::OK();
  }

  if (op->start_indices.empty() || op->stop_indices.empty() ||
      op->strides.empty()) {
    // Attributes have not resolved yet.
    return ::tensorflow::Status::OK();
  }

  const auto& input_array = model->GetArray(op->inputs[0]);
  if (!input_array.has_shape()) {
    // Yield until the value shape has been resolved.
    return ::tensorflow::Status::OK();
  }
  if (!IsConstantParameterArray(*model, op->inputs[0])) {
    // Yield until the value is constant.
    return ::tensorflow::Status::OK();
  }

  CHECK(!output_array.buffer);
  switch (output_array.data_type) {
    case ArrayDataType::kFloat:
      StridedSlice<ArrayDataType::kFloat>(*op, input_array, &output_array);
      break;
    case ArrayDataType::kUint8:
      StridedSlice<ArrayDataType::kUint8>(*op, input_array, &output_array);
      break;
    case ArrayDataType::kInt32:
      StridedSlice<ArrayDataType::kInt32>(*op, input_array, &output_array);
      break;
    case ArrayDataType::kInt64:
      StridedSlice<ArrayDataType::kInt64>(*op, input_array, &output_array);
      break;
    case ArrayDataType::kComplex64:
      StridedSlice<ArrayDataType::kComplex64>(*op, input_array, &output_array);
      break;
    default:
      LOG(FATAL)
          << "Unsupported data type input to StridedSlice op with output \""
          << op->outputs[0] << "\"";
      break;
  }

  DeleteOpAndArrays(model, op);
  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
