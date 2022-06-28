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
class MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSconvert_pure_conv_to_depthwiseDTcc {
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
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSconvert_pure_conv_to_depthwiseDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSconvert_pure_conv_to_depthwiseDTcc() {
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
#include "tensorflow/lite/toco/tooling_util.h"
#include "tensorflow/core/platform/logging.h"

namespace toco {

::tensorflow::Status ConvertPureConvToDepthwise::Run(Model* model,
                                                     std::size_t op_index,
                                                     bool* modified) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePStocoPSgraph_transformationsPSconvert_pure_conv_to_depthwiseDTcc mht_0(mht_0_v, 198, "", "./tensorflow/lite/toco/graph_transformations/convert_pure_conv_to_depthwise.cc", "ConvertPureConvToDepthwise::Run");

  *modified = false;
  auto conv_it = model->operators.begin() + op_index;
  if (conv_it->get()->type != OperatorType::kConv) {
    return ::tensorflow::Status::OK();
  }
  const auto* conv_op = static_cast<ConvOperator*>(conv_it->get());
  if (conv_op->stride_width != conv_op->stride_height) {
    return ::tensorflow::Status::OK();
  }
  if ((conv_op->dilation_width_factor != 1) ||
      (conv_op->dilation_height_factor != 1)) {
    // Depthwise conv does not support dilation
    return ::tensorflow::Status::OK();
  }
  auto& input_array = model->GetArray(conv_op->inputs[0]);
  if (!input_array.has_shape()) {
    // Shapes not propagated yet
    return ::tensorflow::Status::OK();
  }
  if (input_array.shape().dims(3) != 1) {
    // Not a pure convolution: Conv does accumulation across the depth
    // dimension.
    return ::tensorflow::Status::OK();
  }

  const auto& weights_name = conv_op->inputs[1];
  if (CountOpsWithInput(*model, weights_name) > 1) {
    // TODO(yunluli): Come up with a way to do the weights shuffling only once.
    AddMessageF(
        "Not changing %s to DepthwiseConv because the weights is consumed by "
        "another op.",
        LogName(*conv_op));
    return ::tensorflow::Status::OK();
  }
  auto& weights_array = model->GetArray(weights_name);
  if (!weights_array.buffer) {
    // Yield until the weights are resolved as a constant array.
    return ::tensorflow::Status::OK();
  }
  if (weights_array.data_type != ArrayDataType::kFloat) {
    return ::tensorflow::Status::OK();
  }
  // At this point we know we have a pure conv. Rewrite it as DepthwiseConv.
  AddMessageF(
      "%s is purely convolutional (input/weights depth is 1), replacing it by "
      "a DepthwiseConv.",
      LogName(*conv_op));
  auto* depthwiseconv_op = new DepthwiseConvOperator;
  // Conv and DepthwiseConv take the same inputs
  depthwiseconv_op->inputs = conv_op->inputs;
  // Conv may have a 2nd output for im2col
  depthwiseconv_op->outputs = {conv_op->outputs[0]};
  if (conv_op->outputs.size() > 1) {
    // delete the im2col array.
    model->EraseArray(conv_op->outputs[1]);
  }
  depthwiseconv_op->fused_activation_function =
      conv_op->fused_activation_function;
  // Let PropagateFixedSizes recompute fixed padding, just in case some day it
  // may be different for Conv vs DepthwiseConv.
  depthwiseconv_op->padding.type = conv_op->padding.type;
  depthwiseconv_op->stride_height = conv_op->stride_height;
  depthwiseconv_op->stride_width = conv_op->stride_width;
  depthwiseconv_op->depth_multiplier = weights_array.shape().dims(0);
  // Replace the operator in the graph.
  model->operators.emplace(conv_it, depthwiseconv_op);
  DeleteOpAndArrays(model, conv_op);
  // Shuffle the weights.
  const auto& weights_shape = weights_array.shape();
  auto& weights_buffer =
      weights_array.GetMutableBuffer<ArrayDataType::kFloat>();
  const std::vector<float>& conv_weights_data = weights_buffer.data;
  std::vector<float> depthwise_conv_weights_data(conv_weights_data.size());
  const int depth = weights_shape.dims(0);
  const int width = weights_shape.dims(1);
  const int height = weights_shape.dims(2);
  const int width_height = width * height;
  for (int c = 0; c < depth; c++) {
    for (int xy = 0; xy < width_height; xy++) {
      depthwise_conv_weights_data[c + depth * xy] =
          conv_weights_data[xy + width_height * c];
    }
  }
  *weights_array.mutable_shape()->mutable_dims() = {1, width, height, depth};
  weights_buffer.data = depthwise_conv_weights_data;
  *modified = true;
  return ::tensorflow::Status::OK();
}

}  // namespace toco
