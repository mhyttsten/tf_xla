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
class MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfuse_convolutionsDTcc {
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
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfuse_convolutionsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfuse_convolutionsDTcc() {
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

#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/fold_constants_lib.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

Status FuseResizePadAndConv(const GraphDef& input_graph_def,
                            const TransformFuncContext& context,
                            GraphDef* output_graph_def) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfuse_convolutionsDTcc mht_0(mht_0_v, 199, "", "./tensorflow/tools/graph_transforms/fuse_convolutions.cc", "FuseResizePadAndConv");

  GraphDef replaced_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def,  // clang-format off
      {"Conv2D",
          {
              {"MirrorPad",
                  {
                      {"ResizeBilinear"},
                      {"*"}
                  }
              },
              {"*"}
          }
      },  // clang-format on
      [](const NodeMatch& match, const std::set<string>& input_nodes,
         const std::set<string>& output_nodes,
         std::vector<NodeDef>* new_nodes) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfuse_convolutionsDTcc mht_1(mht_1_v, 219, "", "./tensorflow/tools/graph_transforms/fuse_convolutions.cc", "lambda");

        // Find all the nodes we expect in the subgraph.
        const NodeDef& conv_node = match.node;
        const NodeDef& mirror_pad_node = match.inputs[0].node;
        const NodeDef& weights_node = match.inputs[1].node;
        const NodeDef& resize_node = match.inputs[0].inputs[0].node;
        const NodeDef& pad_dims_node = match.inputs[0].inputs[1].node;

        // We'll be reusing the old weights and pad dimensions.
        new_nodes->push_back(weights_node);
        new_nodes->push_back(pad_dims_node);

        // Set up the new fused version of the convolution op.
        NodeDef fused_conv;
        fused_conv.set_op("FusedResizeAndPadConv2D");
        fused_conv.set_name(match.node.name());
        AddNodeInput(resize_node.input(0), &fused_conv);
        AddNodeInput(resize_node.input(1), &fused_conv);
        AddNodeInput(mirror_pad_node.input(1), &fused_conv);
        AddNodeInput(conv_node.input(1), &fused_conv);
        CopyNodeAttr(resize_node, "align_corners", "resize_align_corners",
                     &fused_conv);
        CopyNodeAttr(mirror_pad_node, "mode", "mode", &fused_conv);
        CopyNodeAttr(conv_node, "T", "T", &fused_conv);
        CopyNodeAttr(conv_node, "padding", "padding", &fused_conv);
        CopyNodeAttr(conv_node, "strides", "strides", &fused_conv);
        new_nodes->push_back(fused_conv);

        return Status::OK();
      },
      {}, &replaced_graph_def));
  *output_graph_def = replaced_graph_def;
  return Status::OK();
}

Status FuseResizeAndConv(const GraphDef& input_graph_def,
                         const TransformFuncContext& context,
                         GraphDef* output_graph_def) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfuse_convolutionsDTcc mht_2(mht_2_v, 259, "", "./tensorflow/tools/graph_transforms/fuse_convolutions.cc", "FuseResizeAndConv");

  GraphDef replaced_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def,  // clang-format off
      {"Conv2D",
          {
              {"ResizeBilinear"},
              {"*"}
          }
      },  // clang-format on
      [](const NodeMatch& match, const std::set<string>& input_nodes,
         const std::set<string>& output_nodes,
         std::vector<NodeDef>* new_nodes) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfuse_convolutionsDTcc mht_3(mht_3_v, 274, "", "./tensorflow/tools/graph_transforms/fuse_convolutions.cc", "lambda");

        // Find all the nodes we expect in the subgraph.
        const NodeDef& conv_node = match.node;
        const NodeDef& resize_node = match.inputs[0].node;
        const NodeDef& weights_node = match.inputs[1].node;

        // We'll be reusing the old weights.
        new_nodes->push_back(weights_node);

        // Create a 'no-op' mirror padding node that has no effect.
        NodeDef pad_dims_node;
        pad_dims_node.set_op("Const");
        pad_dims_node.set_name(conv_node.name() + "_dummy_paddings");
        SetNodeAttr("dtype", DT_INT32, &pad_dims_node);
        SetNodeTensorAttr<int32>("value", {4, 2}, {0, 0, 0, 0, 0, 0, 0, 0},
                                 &pad_dims_node);
        new_nodes->push_back(pad_dims_node);

        // Set up the new fused version of the convolution op.
        NodeDef fused_conv;
        fused_conv.set_op("FusedResizeAndPadConv2D");
        fused_conv.set_name(match.node.name());
        AddNodeInput(resize_node.input(0), &fused_conv);
        AddNodeInput(resize_node.input(1), &fused_conv);
        AddNodeInput(pad_dims_node.name(), &fused_conv);
        AddNodeInput(conv_node.input(1), &fused_conv);
        CopyNodeAttr(resize_node, "align_corners", "resize_align_corners",
                     &fused_conv);
        SetNodeAttr("mode", "REFLECT", &fused_conv);
        CopyNodeAttr(conv_node, "T", "T", &fused_conv);
        CopyNodeAttr(conv_node, "padding", "padding", &fused_conv);
        CopyNodeAttr(conv_node, "strides", "strides", &fused_conv);
        new_nodes->push_back(fused_conv);

        return Status::OK();
      },
      {}, &replaced_graph_def));
  *output_graph_def = replaced_graph_def;
  return Status::OK();
}

Status FusePadAndConv(const GraphDef& input_graph_def,
                      const TransformFuncContext& context,
                      GraphDef* output_graph_def) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfuse_convolutionsDTcc mht_4(mht_4_v, 320, "", "./tensorflow/tools/graph_transforms/fuse_convolutions.cc", "FusePadAndConv");

  GraphDef replaced_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def,  // clang-format off
      {"Conv2D",
          {
              {"MirrorPad",
                  {
                      {"*"},
                      {"*"},
                  }
              },
              {"*"}
          }
      },  // clang-format on
      [](const NodeMatch& match, const std::set<string>& input_nodes,
         const std::set<string>& output_nodes,
         std::vector<NodeDef>* new_nodes) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSfuse_convolutionsDTcc mht_5(mht_5_v, 340, "", "./tensorflow/tools/graph_transforms/fuse_convolutions.cc", "lambda");

        // Find all the nodes we expect in the subgraph.
        const NodeDef& conv_node = match.node;
        CHECK_EQ("Conv2D", conv_node.op());
        const NodeDef& mirror_pad_node = match.inputs[0].node;
        CHECK_EQ("MirrorPad", mirror_pad_node.op());
        const NodeDef& weights_node = match.inputs[1].node;
        const NodeDef& input_node = match.inputs[0].inputs[0].node;
        const NodeDef& pad_dims_node = match.inputs[0].inputs[1].node;

        // We'll be reusing the old weights and pad dimensions.
        new_nodes->push_back(weights_node);
        new_nodes->push_back(input_node);
        new_nodes->push_back(pad_dims_node);

        // Set up the new fused version of the convolution op.
        NodeDef fused_conv;
        fused_conv.set_op("FusedPadConv2D");
        fused_conv.set_name(match.node.name());
        AddNodeInput(mirror_pad_node.input(0), &fused_conv);
        AddNodeInput(mirror_pad_node.input(1), &fused_conv);
        AddNodeInput(conv_node.input(1), &fused_conv);
        CopyNodeAttr(mirror_pad_node, "mode", "mode", &fused_conv);
        CopyNodeAttr(conv_node, "T", "T", &fused_conv);
        CopyNodeAttr(conv_node, "padding", "padding", &fused_conv);
        CopyNodeAttr(conv_node, "strides", "strides", &fused_conv);
        new_nodes->push_back(fused_conv);

        return Status::OK();
      },
      {}, &replaced_graph_def));
  *output_graph_def = replaced_graph_def;
  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("fuse_resize_pad_and_conv", FuseResizePadAndConv);

REGISTER_GRAPH_TRANSFORM("fuse_resize_and_conv", FuseResizeAndConv);

REGISTER_GRAPH_TRANSFORM("fuse_pad_and_conv", FusePadAndConv);

}  // namespace graph_transforms
}  // namespace tensorflow
