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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSshuffle_fc_weightsDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSshuffle_fc_weightsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSshuffle_fc_weightsDTcc() {
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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensorflow/lite/toco/graph_transformations/graph_transformations.h"
#include "tensorflow/lite/toco/model.h"
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

::tensorflow::Status ShuffleFCWeights::Run(Model* model, std::size_t op_index,
                                           bool* modified) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSshuffle_fc_weightsDTcc mht_0(mht_0_v, 197, "", "./tensorflow/lite/toco/graph_transformations/shuffle_fc_weights.cc", "ShuffleFCWeights::Run");

  *modified = false;
  Operator* op = model->operators[op_index].get();
  if (op->type != OperatorType::kFullyConnected) {
    return ::tensorflow::Status::OK();
  }
  FullyConnectedOperator* fc_op = static_cast<FullyConnectedOperator*>(op);
  // Exit if this FC op already has shuffled weights
  if (fc_op->weights_format != FullyConnectedWeightsFormat::kDefault) {
    return ::tensorflow::Status::OK();
  }
  const Array& input_array = model->GetArray(fc_op->inputs[0]);
  const std::string& weights_name = fc_op->inputs[1];
  Array& weights_array = model->GetArray(weights_name);
  const Array& output_array = model->GetArray(fc_op->outputs[0]);
  // Exit if this FC op isn't quantized with uint8 inputs and int16 outputs,
  // the only case where we are currently interested in providing a fast path
  // with shuffled weights.
  if (input_array.data_type != ArrayDataType::kUint8 ||
      weights_array.data_type != ArrayDataType::kUint8 ||
      output_array.data_type != ArrayDataType::kInt16 ||
      !input_array.quantization_params || !weights_array.quantization_params ||
      !output_array.quantization_params) {
    return ::tensorflow::Status::OK();
  }
  // Exit if the shapes aren't known
  if (!input_array.has_shape() || !weights_array.has_shape()) {
    return ::tensorflow::Status::OK();
  }
  // Exit if, based on the known shapes, this FC op is not a GEMV.
  // The shuffling of FC weights is only useful to enable fast GEMV paths.
  const Shape& input_shape = input_array.shape();
  for (int i = 1; i < input_shape.dimensions_count() - 1; i++) {
    if (input_shape.dims(i) != 1) {
      // The input activations, shaped as a matrix, have multiple columns.
      // This FC op isn't a matrix*vector multiplication.
      AddMessageF(
          "Not applying experimental shuffling to the weights of %s because "
          "the input shape is not 1D or 2D (possibly with additional inner "
          "dimensions of size 1)",
          LogName(*op));
      return ::tensorflow::Status::OK();
    }
  }
  if (input_shape.dims(0) != 1 && input_shape.dims(0) != 4) {
    AddMessageF(
        "Not applying experimental shuffling to the weights of %s because "
        "the input shape's leading dimension, i.e. the 'batch size', is not "
        "equal to 1 or 4",
        LogName(*op));
    return ::tensorflow::Status::OK();
  }
  // Exit if the weights shape isn't an integral multiple of the shuffled
  // block shape, 4x16. We don't want to have to write code dealing with
  // odd sizes, that would go un-exercised at the moment as the models
  // for which we need this shuffling have shapes that are multiples of that
  // 4x16 block size. In fact, much of the rationale for this shuffling is
  // to avoid cache aliasin issue with large power-of-two depths, with our
  // models motivating this shuffling having FC weights shapes like
  // 4096x2048. Thus, if some model doesn't get the shuffling because of that
  // size requirement, that might be just fine --- that model might just not
  // suffer from that cache aliasing issue that we have with large powers of
  // two.
  const Shape& weights_shape = weights_array.shape();
  if (weights_shape.dimensions_count() != 2) {
    return ::tensorflow::Status::OK();
  }
  const int rows = weights_shape.dims(0);
  const int cols = weights_shape.dims(1);
  if (rows % 4 || cols % 16) {
    AddMessageF(
        "Not applying experimental shuffling to the weights of %s because its "
        "shape isn't a multiple of the shuffling block shape, 4x16",
        LogName(*op));
    return ::tensorflow::Status::OK();
  }
  // Exit if the weights aren't already a constant array.
  if (!weights_array.buffer) {
    return ::tensorflow::Status::OK();
  }
  // Exit if the weights are used by more than one op.
  if (CountOpsWithInput(*model, weights_name) != 1) {
    AddMessageF(
        "Not applying experimental shuffling to the weights of %s because that "
        "array is consumed by other operators",
        LogName(*op));
    return ::tensorflow::Status::OK();
  }
  // Compute the shuffled weights
  auto& weights_data =
      weights_array.GetMutableBuffer<ArrayDataType::kUint8>().data;
  CHECK_EQ(rows * cols, weights_data.size());
  std::vector<uint8> shuffled_data(weights_data.size());
  uint8* shuffled_data_ptr = shuffled_data.data();
  for (int r = 0; r < rows; r += 4) {
    for (int c = 0; c < cols; c += 16) {
      for (int i = 0; i < 4; i++) {
        const uint8* src_data_ptr = weights_data.data() + (r + i) * cols + c;
        for (int j = 0; j < 16; j++) {
          uint8 src_val = *src_data_ptr++;
          // Flip the sign bit, so that the runtime will only need to
          // reinterpret these uint8 values as int8, getting for free the
          // subtraction of the zero_point value 128.
          uint8 dst_val = src_val ^ 0x80;
          *shuffled_data_ptr++ = dst_val;
        }
      }
    }
  }
  CHECK_EQ(shuffled_data_ptr, shuffled_data.data() + rows * cols);
  // Switch this FC op to using the shuffled weights.
  weights_data = std::move(shuffled_data);
  fc_op->weights_format = FullyConnectedWeightsFormat::kShuffled4x16Int8;
  AddMessageF("Applied experimental shuffling to the weights of %s",
              LogName(*op));
  // Add a second output array to this FC op, serving as a workspace to perform
  // runtime shuffling/xoring of its input activations.
  CHECK_EQ(fc_op->outputs.size(), 1);
  const std::string& shuffled_input_workspace_array_name =
      AvailableArrayName(*model, fc_op->inputs[0] + "_shuffled");
  fc_op->outputs.push_back(shuffled_input_workspace_array_name);
  auto& shuffled_input_workspace_array =
      model->GetOrCreateArray(shuffled_input_workspace_array_name);
  shuffled_input_workspace_array.data_type = input_array.data_type;
  *shuffled_input_workspace_array.mutable_shape() = input_array.shape();
  shuffled_input_workspace_array.GetOrCreateMinMax() = input_array.GetMinMax();
  shuffled_input_workspace_array.GetOrCreateQuantizationParams() =
      input_array.GetQuantizationParams();

  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
