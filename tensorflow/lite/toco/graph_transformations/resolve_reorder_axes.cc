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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_reorder_axesDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_reorder_axesDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_reorder_axesDTcc() {
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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

namespace {

void RenameArray(Model* model, const std::string& oldname,
                 const std::string& desired_newname) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("oldname: \"" + oldname + "\"");
   mht_0_v.push_back("desired_newname: \"" + desired_newname + "\"");
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_reorder_axesDTcc mht_0(mht_0_v, 202, "", "./tensorflow/lite/toco/graph_transformations/resolve_reorder_axes.cc", "RenameArray");

  const std::string& newname = AvailableArrayName(*model, desired_newname);
  auto& arrays = model->GetMutableArrayMap();
  arrays[newname] = std::move(arrays[oldname]);
  arrays.erase(oldname);
  for (const auto& op : model->operators) {
    for (std::string& input : op->inputs) {
      if (input == oldname) {
        input = newname;
      }
    }
    for (std::string& output : op->outputs) {
      if (output == oldname) {
        output = newname;
      }
    }
  }
}

}  // namespace

// Reorder the elements of an input_array according to the input_axes_order and
// output_axes_order. Then adjust the shapes of the input and output arrays
// accordingly. Note that input_array must have a buffer (that is, it is a
// constant array).
template <typename T, ArrayDataType DataType>
void ReorderAxes(AxesOrder input_axes_order, AxesOrder output_axes_order,
                 const Array& input_array, Array* output_array) {
  DCHECK(input_array.buffer->type == DataType);
  DCHECK(!output_array->buffer);
  const auto& input_data = input_array.GetBuffer<DataType>().data;
  auto& output_data = output_array->GetMutableBuffer<DataType>().data;
  output_data.resize(RequiredBufferSizeForShape(output_array->shape()));
  // TODO(b/62904716) Shapes should be used directly.
  Shape input_shape = input_array.shape();
  Shape output_shape = output_array->shape();
  if (AxesCount(input_axes_order) == 2) {
    UnextendShape(&input_shape, 2);
    UnextendShape(&output_shape, 2);
  }
  ShuffleArray(input_shape, input_axes_order, output_axes_order, output_shape,
               input_data.data(), output_data.data());
  if (input_array.minmax) {
    output_array->GetOrCreateMinMax() = input_array.GetMinMax();
  }
  if (input_array.narrow_range) {
    output_array->narrow_range = true;
  }
}

::tensorflow::Status ResolveReorderAxes::Run(Model* model, std::size_t op_index,
                                             bool* modified) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSresolve_reorder_axesDTcc mht_1(mht_1_v, 256, "", "./tensorflow/lite/toco/graph_transformations/resolve_reorder_axes.cc", "ResolveReorderAxes::Run");

  *modified = false;
  auto it = model->operators.begin() + op_index;
  auto* op = it->get();
  if (op->type != OperatorType::kReorderAxes) {
    return ::tensorflow::Status::OK();
  }
  auto* reorder_op = static_cast<ReorderAxesOperator*>(op);

  // Intentionally copies, not references.
  const std::string input_array_name = reorder_op->inputs[0];
  const std::string output_array_name = reorder_op->outputs[0];

  auto& input_array = model->GetArray(input_array_name);
  auto& output_array = model->GetArray(output_array_name);
  if (!input_array.buffer) {
    return ::tensorflow::Status::OK();
  }
  // Yield until output dims have been resolved.
  if (!output_array.has_shape()) {
    return ::tensorflow::Status::OK();
  }
  // Reorder the input array dims and buffer data
  if (input_array.buffer->type == ArrayDataType::kFloat) {
    ReorderAxes<float, ArrayDataType::kFloat>(reorder_op->input_axes_order,
                                              reorder_op->output_axes_order,
                                              input_array, &output_array);
  } else if (input_array.buffer->type == ArrayDataType::kUint8) {
    // TODO(benoitjacob): This path seems unused.
    // ReorderAxes is only used when importing from
    // TensorFlow GraphDef, which does not support quantized nodes.
    ReorderAxes<uint8, ArrayDataType::kUint8>(reorder_op->input_axes_order,
                                              reorder_op->output_axes_order,
                                              input_array, &output_array);
  } else {
    LOG(FATAL) << "Cannot ReorderAxes unless input buffer is float or uint8.";
  }

  AddMessageF("Reordered axes for array %s", input_array_name);

  DeleteOpAndArrays(model, op);
  RenameArray(model, output_array_name, input_array_name);

  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
