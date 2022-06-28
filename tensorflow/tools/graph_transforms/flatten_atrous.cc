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
class MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSflatten_atrousDTcc {
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
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSflatten_atrousDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSflatten_atrousDTcc() {
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

#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

Status FlattenAtrousConv(const GraphDef& input_graph_def,
                         const TransformFuncContext& context,
                         GraphDef* output_graph_def) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSflatten_atrousDTcc mht_0(mht_0_v, 197, "", "./tensorflow/tools/graph_transforms/flatten_atrous.cc", "FlattenAtrousConv");

  GraphDef replaced_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def,  // clang-format off
      {"BatchToSpaceND",
          {
              {"Conv2D|DepthwiseConv2dNative",
                  {
                      {"SpaceToBatchND",
                          {
                              {"*"},          // Input to the flattened op.
                              {"*"},          // block_shape
                              {"*"}           // paddings
                          }
                      },
                      {"*"}                   // filter
                  }
              },
              {"*"},                          // block_shape
              {"*"}                           // crops
          }
      },  // clang-format on
      [](const NodeMatch& match, const std::set<string>& input_nodes,
         const std::set<string>& output_nodes,
         std::vector<NodeDef>* new_nodes) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSflatten_atrousDTcc mht_1(mht_1_v, 224, "", "./tensorflow/tools/graph_transforms/flatten_atrous.cc", "lambda");

        // Find all the nodes we expect in the subgraph.
        const NodeDef& batch_to_space_node = match.node;
        const NodeDef& conv_node = match.inputs[0].node;
        const NodeDef& filter_node = match.inputs[0].inputs[1].node;
        const NodeDef& input_node = match.inputs[0].inputs[0].inputs[0].node;
        const NodeDef& space_to_batch_block_shape_node =
            match.inputs[0].inputs[0].inputs[1].node;

        // The atrous rate value is inferred from the block shape.
        Tensor block_shape =
            GetNodeTensorAttr(space_to_batch_block_shape_node, "value");
        const int32_t block_height = block_shape.flat<int32>()(0);
        const int32_t block_width = block_shape.flat<int32>()(1);

        // Compute the upsampled filter.
        const Tensor& filter = GetNodeTensorAttr(filter_node, "value");
        const int32_t filter_height = filter.dim_size(0);
        const int32_t filter_width = filter.dim_size(1);
        const int32_t in_channels = filter.dim_size(2);
        const int32_t out_channels = filter.dim_size(3);

        const int32_t upsampled_filter_height =
            (filter_height - 1) * block_height + 1;
        const int32_t upsampled_filter_width =
            (filter_width - 1) * block_width + 1;
        Tensor upsampled_filter(
            DT_FLOAT,
            TensorShape({upsampled_filter_height, upsampled_filter_width,
                         in_channels, out_channels}));

        auto filter_eigen = filter.tensor<float, 4>();
        auto upsampled_filter_eigen = upsampled_filter.tensor<float, 4>();

        upsampled_filter_eigen.setZero();
        for (int h = 0; h < filter_height; ++h) {
          for (int w = 0; w < filter_width; ++w) {
            for (int c_in = 0; c_in < in_channels; ++c_in) {
              for (int c_out = 0; c_out < out_channels; ++c_out) {
                upsampled_filter_eigen(block_height * h, block_width * w, c_in,
                                       c_out) = filter_eigen(h, w, c_in, c_out);
              }
            }
          }
        }

        NodeDef upsampled_filter_node;
        upsampled_filter_node.set_op("Const");
        upsampled_filter_node.set_name(filter_node.name());
        SetNodeAttr("dtype", DT_FLOAT, &upsampled_filter_node);
        SetNodeTensorAttr<float>("value", upsampled_filter,
                                 &upsampled_filter_node);

        // Set up the new flattened version of the convolution op.
        NodeDef flattened_conv_node;

        flattened_conv_node.set_name(batch_to_space_node.name());
        flattened_conv_node.set_op(conv_node.op());
        flattened_conv_node.set_device(conv_node.device());

        AddNodeInput(input_node.name(), &flattened_conv_node);
        AddNodeInput(upsampled_filter_node.name(), &flattened_conv_node);

        CopyNodeAttr(conv_node, "T", "T", &flattened_conv_node);
        CopyNodeAttr(conv_node, "strides", "strides", &flattened_conv_node);
        SetNodeAttr("padding", "SAME", &flattened_conv_node);
        CopyNodeAttr(conv_node, "data_format", "data_format",
                     &flattened_conv_node);

        if (conv_node.op() == "Conv2D") {
          CopyNodeAttr(conv_node, "use_cudnn_on_gpu", "use_cudnn_on_gpu",
                       &flattened_conv_node);
        }

        new_nodes->push_back(input_node);
        new_nodes->push_back(upsampled_filter_node);
        new_nodes->push_back(flattened_conv_node);

        return Status::OK();
      },
      {}, &replaced_graph_def));
  *output_graph_def = replaced_graph_def;
  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("flatten_atrous_conv", FlattenAtrousConv);

}  // namespace graph_transforms
}  // namespace tensorflow
