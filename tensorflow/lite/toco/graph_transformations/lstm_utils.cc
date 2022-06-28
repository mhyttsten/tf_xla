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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSlstm_utilsDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSlstm_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSlstm_utilsDTcc() {
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
#include "tensorflow/lite/toco/graph_transformations/lstm_utils.h"

namespace toco {

void CreateOptionalArray(Model* model, std::string* input_array_buffer,
                         const std::string& array_name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("array_name: \"" + array_name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSlstm_utilsDTcc mht_0(mht_0_v, 190, "", "./tensorflow/lite/toco/graph_transformations/lstm_utils.cc", "CreateOptionalArray");

  *input_array_buffer = array_name;
  model->CreateOptionalArray(array_name);
}

void CopyArrayData(const Buffer<ArrayDataType::kFloat>& src_buffer,
                   int src_stride, int src_start_idx1, int src_start_idx2,
                   Buffer<ArrayDataType::kFloat>* dst_buffer, int dst_stride,
                   int dst_start_idx1, int dst_start_idx2, int dim1_copy_size,
                   int dim2_copy_size) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSlstm_utilsDTcc mht_1(mht_1_v, 202, "", "./tensorflow/lite/toco/graph_transformations/lstm_utils.cc", "CopyArrayData");

  int src_offset = src_start_idx1 * src_stride + src_start_idx2;
  int dst_offset = dst_start_idx1 * dst_stride + dst_start_idx2;
  for (int i = 0; i < dim1_copy_size; i++) {
    for (int j = 0; j < dim2_copy_size; j++) {
      int idx_src = src_offset + i * src_stride + j;
      int idx_dst = dst_offset + i * dst_stride + j;
      dst_buffer->data[idx_dst] = src_buffer.data[idx_src];
    }
  }
}

Buffer<ArrayDataType::kFloat>* CreateFloatArrayBuffer(Model* model,
                                                      std::string* array_name,
                                                      const Shape& shape) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSlstm_utilsDTcc mht_2(mht_2_v, 219, "", "./tensorflow/lite/toco/graph_transformations/lstm_utils.cc", "CreateFloatArrayBuffer");

  *array_name = AvailableArrayName(*model, *array_name);
  auto& array = model->GetOrCreateArray(*array_name);
  array.data_type = ArrayDataType::kFloat;
  array.copy_shape(shape);
  Buffer<ArrayDataType::kFloat>* buffer =
      &(array.GetMutableBuffer<ArrayDataType::kFloat>());
  buffer->data.resize(RequiredBufferSizeForShape(shape));
  return buffer;
}

void CopySubArrayToArray(Model* model, std::string* array_name,
                         const std::string& tensor_name, int dim1_size,
                         int dim2_size, const Array& original_array,
                         int start_idx1, int start_idx2) {
   std::vector<std::string> mht_3_v;
   mht_3_v.push_back("tensor_name: \"" + tensor_name + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSlstm_utilsDTcc mht_3(mht_3_v, 237, "", "./tensorflow/lite/toco/graph_transformations/lstm_utils.cc", "CopySubArrayToArray");

  // Determine whether it's bias or not, create shape, buffer.
  bool is_bias = dim2_size == 1;
  Shape shape = is_bias ? Shape({dim1_size}) : Shape({dim1_size, dim2_size});
  Buffer<ArrayDataType::kFloat>* buffer =
      CreateFloatArrayBuffer(model, array_name, shape);
  auto& orig_buffer = original_array.GetBuffer<ArrayDataType::kFloat>();

  // Copy data from big tensor.
  CopyArrayData(orig_buffer, is_bias ? 1 : original_array.shape().dims(1),
                start_idx1, start_idx2, buffer, dim2_size, 0, 0, dim1_size,
                dim2_size);
}

void CopyArrayToSubArray(Buffer<ArrayDataType::kFloat>& tensor_buffer,
                         int tensor_stride, const Array& sub_array,
                         int start_idx1, int start_idx2) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSlstm_utilsDTcc mht_4(mht_4_v, 256, "", "./tensorflow/lite/toco/graph_transformations/lstm_utils.cc", "CopyArrayToSubArray");

  // Get tensor data.
  bool is_bias = sub_array.shape().dims().size() == 1;
  int dim1_copy_size = sub_array.shape().dims()[0];
  int dim2_copy_size = is_bias ? 1 : sub_array.shape().dims(1);
  auto& sub_buffer = sub_array.GetBuffer<ArrayDataType::kFloat>();

  // Copy data from sub tensor.
  CopyArrayData(sub_buffer, dim2_copy_size, 0, 0, &tensor_buffer,
                is_bias ? 1 : tensor_stride, start_idx1, start_idx2,
                dim1_copy_size, dim2_copy_size);
}

bool GetMatchingRnnArray(Model* model,
                         const std::string& back_edge_source_array,
                         std::string* rnn_array) {
   std::vector<std::string> mht_5_v;
   mht_5_v.push_back("back_edge_source_array: \"" + back_edge_source_array + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSlstm_utilsDTcc mht_5(mht_5_v, 275, "", "./tensorflow/lite/toco/graph_transformations/lstm_utils.cc", "GetMatchingRnnArray");

  for (const auto& rnn_state : model->flags.rnn_states()) {
    if (rnn_state.back_edge_source_array() == back_edge_source_array) {
      *rnn_array = rnn_state.state_array();
      return true;
    }
  }
  return false;
}

}  // namespace toco
