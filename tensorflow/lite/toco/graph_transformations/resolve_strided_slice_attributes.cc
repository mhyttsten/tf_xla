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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_strided_slice_attributesDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_strided_slice_attributesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_strided_slice_attributesDTcc() {
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
#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

int PadAttributeArray(Array* attribute_array, std::vector<int> pad_values,
                      int mask) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_strided_slice_attributesDTcc mht_0(mht_0_v, 192, "", "./tensorflow/lite/toco/graph_transformations/resolve_strided_slice_attributes.cc", "PadAttributeArray");

  int attribute_dim_count = attribute_array->shape().dims(0);
  int dim_count = pad_values.size();
  if (attribute_dim_count < dim_count) {
    Shape strided_slice_shape = Shape({dim_count});
    attribute_array->copy_shape(strided_slice_shape);
    Buffer<ArrayDataType::kInt32>* buffer =
        &(attribute_array->GetMutableBuffer<ArrayDataType::kInt32>());
    buffer->data.resize(RequiredBufferSizeForShape(strided_slice_shape));
    for (int i = attribute_dim_count; i < dim_count; i++) {
      buffer->data[i] = pad_values[i];
      mask |= 1 << i;
    }
  }
  return mask;
}

::tensorflow::Status ResolveStridedSliceAttributes::Run(Model* model,
                                                        std::size_t op_index,
                                                        bool* modified) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_strided_slice_attributesDTcc mht_1(mht_1_v, 214, "", "./tensorflow/lite/toco/graph_transformations/resolve_strided_slice_attributes.cc", "ResolveStridedSliceAttributes::Run");

  *modified = false;
  const auto slice_it = model->operators.begin() + op_index;
  auto* slice_op = slice_it->get();
  if (slice_op->type != OperatorType::kStridedSlice)
    return ::tensorflow::Status::OK();

  auto* op = static_cast<StridedSliceOperator*>(slice_op);
  if (!op->start_indices.empty()) {
    // We have already resolved these attributes
    return ::tensorflow::Status::OK();
  }

  CHECK_EQ(op->inputs.size(), 4);
  const auto& input_array = model->GetArray(op->inputs[0]);
  if (!input_array.has_shape()) {
    // We require the dimensionality of the input to pad the indices
    return ::tensorflow::Status::OK();
  }

  auto& start_array = model->GetArray(op->inputs[1]);
  if (!start_array.has_shape()) return ::tensorflow::Status::OK();
  if (toco::RequiredBufferSizeForShape(start_array.shape()) > 4) {
    // Only 1-4D arrays are supported for now.
    return ::tensorflow::Status::OK();
  }

  auto& stop_array = model->GetArray(op->inputs[2]);
  if (!stop_array.has_shape()) return ::tensorflow::Status::OK();

  auto& stride_array = model->GetArray(op->inputs[3]);
  if (!stride_array.has_shape()) return ::tensorflow::Status::OK();

  if (!IsConstantParameterArray(*model, op->inputs[1]))
    return ::tensorflow::Status::OK();
  if (!IsConstantParameterArray(*model, op->inputs[2]))
    return ::tensorflow::Status::OK();
  if (!IsConstantParameterArray(*model, op->inputs[3]))
    return ::tensorflow::Status::OK();

  int num_input_axes = input_array.shape().dimensions_count();
  int start_indices_size = start_array.shape().dims(0);
  int stop_indices_size = stop_array.shape().dims(0);
  int stride_indices_size = stride_array.shape().dims(0);

  CHECK_GE(start_indices_size, 1);
  CHECK_LE(start_indices_size, 4);
  CHECK_LE(stop_indices_size, 4);
  CHECK_LE(stride_indices_size, 4);

  // The TensorFlow documentation is not explicit on how it handles fewer
  // supplied indices than dimensions, but they are accepted. We emulate TF's
  // behavior by fully iterating over each omitted dimension.
  CHECK_LE(start_indices_size, num_input_axes)
      << "StridedSlice op requires no more than " << num_input_axes
      << " start indices";
  CHECK_LE(stop_indices_size, num_input_axes)
      << "StridedSlice op requires no more than " << num_input_axes
      << " stop indices";
  CHECK_LE(stride_indices_size, num_input_axes)
      << "StridedSlice op requires no more than " << num_input_axes
      << " strides";

  // Ideally, we would remove the input arrays after they have been resolved.
  // However, we must then reconstitute these input arrays for all supported
  // export formats. For now, leave the arrays so we don't have to modify our
  // exporters. Ideally, we wouldn't have op attributes, and would work directly
  // with the input arrays.
  std::vector<int> begin_pad_values(num_input_axes, 0);
  op->begin_mask =
      PadAttributeArray(&start_array, begin_pad_values, op->begin_mask);
  op->end_mask =
      PadAttributeArray(&stop_array, input_array.shape().dims(), op->end_mask);
  std::vector<int> stride_pad_values(num_input_axes, 1);
  PadAttributeArray(&stride_array, stride_pad_values, 0);

  op->start_indices = start_array.GetBuffer<ArrayDataType::kInt32>().data;
  op->stop_indices = stop_array.GetBuffer<ArrayDataType::kInt32>().data;
  op->strides = stride_array.GetBuffer<ArrayDataType::kInt32>().data;

  *modified = true;
  return ::tensorflow::Status::OK();
}
}  // namespace toco
