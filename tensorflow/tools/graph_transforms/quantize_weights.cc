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
class MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSquantize_weightsDTcc {
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
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSquantize_weightsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSquantize_weightsDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/threadpool_device.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/kernels/quantization_utils.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

// Converts any large float constants into eight-bit equivalents, with a
// Dequantize op so that subsequent nodes can still access the results in a
// float form.
Status QuantizeWeights(const GraphDef& input_graph_def,
                       const TransformFuncContext& context,
                       GraphDef* output_graph_def) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSquantize_weightsDTcc mht_0(mht_0_v, 205, "", "./tensorflow/tools/graph_transforms/quantize_weights.cc", "QuantizeWeights");

  int32_t minimum_size;
  TF_RETURN_IF_ERROR(
      context.GetOneInt32Parameter("minimum_size", 1024, &minimum_size));
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def, {"Const"},
      [minimum_size](const NodeMatch& match,
                     const std::set<string>& input_nodes,
                     const std::set<string>& output_nodes,
                     std::vector<NodeDef>* new_nodes) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSquantize_weightsDTcc mht_1(mht_1_v, 217, "", "./tensorflow/tools/graph_transforms/quantize_weights.cc", "lambda");

        const NodeDef& old_const_node = match.node;
        if (!old_const_node.attr().count("dtype")) {
          return errors::InvalidArgument("No 'dtype' attribute for Const node ",
                                         old_const_node.name());
        }
        if (!old_const_node.attr().count("value")) {
          return errors::InvalidArgument("No 'value' attribute for Const node ",
                                         old_const_node.name());
        }
        const DataType old_dtype = old_const_node.attr().at("dtype").type();
        Tensor old_tensor;
        if (!old_tensor.FromProto(old_const_node.attr().at("value").tensor())) {
          return errors::InvalidArgument("Decoding Tensor failed for node",
                                         old_const_node.name());
        }
        const size_t num_elements = old_tensor.NumElements();
        // If this isn't a float constant, or it's too small, then reuse the
        // same node with no changes.
        if ((old_dtype != DT_FLOAT) || (num_elements < minimum_size)) {
          new_nodes->push_back(old_const_node);
          return Status::OK();
        }
        const float* old_values = old_tensor.flat<float>().data();
        float min = std::numeric_limits<float>::max();
        float max = std::numeric_limits<float>::min();
        for (int i = 0; i < num_elements; ++i) {
          const float value = old_values[i];
          min = std::min(min, value);
          max = std::max(max, value);
        }
        // Make sure the quantization range includes 0.0f. Not all quantized
        // Ops behave properly if 0.0f is not in the range.
        min = std::min(min, 0.0f);
        max = std::max(0.0f, max);
        // min_value == max_value is a tricky case. It can occur for general
        // tensors, and of course for scalars. The quantized ops cannot deal
        // with this case, so we set max_value to something else.
        // It's a tricky question what is the numerically best solution to
        // deal with this degeneracy.
        // TODO(petewarden): Better use a tolerance than a hard comparison?
        if (min == max) {
          if (std::abs(min) < 0.000001f) {
            max = min + 1.0f;
          } else if (min > 0) {
            max = 2.0f * min;
          } else {
            max = min / 2.0f;
          }
        }
        Tensor quantized_tensor(DT_QUINT8, old_tensor.shape());
        FloatTensorToQuantizedInPlace<quint8>(old_tensor, min, max,
                                              &quantized_tensor);

        NodeDef quantized_const_node;
        quantized_const_node.set_op("Const");
        quantized_const_node.set_name(old_const_node.name() +
                                      "_quantized_const");
        SetNodeAttr("dtype", DT_QUINT8, &quantized_const_node);
        SetNodeTensorAttr<float>("value", quantized_tensor,
                                 &quantized_const_node);
        new_nodes->push_back(quantized_const_node);

        NodeDef min_node;
        min_node.set_op("Const");
        min_node.set_name(old_const_node.name() + "_quantized_min");
        SetNodeAttr("dtype", DT_FLOAT, &min_node);
        Tensor min_tensor(DT_FLOAT, {});
        min_tensor.scalar<float>()() = min;
        SetNodeTensorAttr<float>("value", min_tensor, &min_node);
        new_nodes->push_back(min_node);

        NodeDef max_node;
        max_node.set_op("Const");
        max_node.set_name(old_const_node.name() + "_quantized_max");
        SetNodeAttr("dtype", DT_FLOAT, &max_node);
        Tensor max_tensor(DT_FLOAT, {});
        max_tensor.scalar<float>()() = max;
        SetNodeTensorAttr<float>("value", max_tensor, &max_node);
        new_nodes->push_back(max_node);

        NodeDef dequantize_node;
        dequantize_node.set_op("Dequantize");
        dequantize_node.set_name(old_const_node.name());
        SetNodeAttr("T", DT_QUINT8, &dequantize_node);
        SetNodeAttr("mode", "MIN_FIRST", &dequantize_node);
        AddNodeInput(quantized_const_node.name(), &dequantize_node);
        AddNodeInput(min_node.name(), &dequantize_node);
        AddNodeInput(max_node.name(), &dequantize_node);
        new_nodes->push_back(dequantize_node);

        return Status::OK();
      },
      {}, output_graph_def));

  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("quantize_weights", QuantizeWeights);

}  // namespace graph_transforms
}  // namespace tensorflow
