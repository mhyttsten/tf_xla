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
class MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSbackportsDTcc {
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
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSbackportsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSbackportsDTcc() {
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

// Switch any ConcatV2 nodes to the v1 version, swapping the input order.
Status BackportConcatV2Transform(const GraphDef& input_graph_def,
                                 const TransformFuncContext& context,
                                 GraphDef* output_graph_def) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSbackportsDTcc mht_0(mht_0_v, 200, "", "./tensorflow/tools/graph_transforms/backports.cc", "BackportConcatV2Transform");

  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def, {"ConcatV2"},
      [](const NodeMatch& match, const std::set<string>& input_nodes,
         const std::set<string>& output_nodes,
         std::vector<NodeDef>* new_nodes) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSbackportsDTcc mht_1(mht_1_v, 208, "", "./tensorflow/tools/graph_transforms/backports.cc", "lambda");

        const NodeDef& concat_v2_node = match.node;
        NodeDef concat_node = concat_v2_node;
        concat_node.set_op("Concat");
        // The last input is inserted at the head of the inputs, because Concat
        // expects the dimension as the first input (not the last as in
        // ConcatV2).
        concat_node.mutable_input()->Clear();
        const string& dim_input =
            concat_v2_node.input(concat_v2_node.input_size() - 1);
        concat_node.add_input(dim_input);
        for (int i = 0; i < (concat_v2_node.input_size() - 1); ++i) {
          concat_node.add_input(concat_v2_node.input(i));
        }
        // Tidx attribute must be deleted because it's not used in Concat.
        concat_node.mutable_attr()->erase("Tidx");
        new_nodes->push_back(concat_node);
        return Status::OK();
      },
      {true}, output_graph_def));

  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("backport_concatv2", BackportConcatV2Transform);

// Switch any TensorArrayV3 nodes to the v2 version, removing the second output.
Status BackportTensorArrayV3Transform(const GraphDef& input_graph_def,
                                      const TransformFuncContext& context,
                                      GraphDef* output_graph_def) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSbackportsDTcc mht_2(mht_2_v, 240, "", "./tensorflow/tools/graph_transforms/backports.cc", "BackportTensorArrayV3Transform");

  std::map<string, string> inputs_to_rename;
  GraphDef replaced_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def, {"TensorArrayV3|TensorArrayGradV3"},
      [&inputs_to_rename](const NodeMatch& match,
                          const std::set<string>& input_nodes,
                          const std::set<string>& output_nodes,
                          std::vector<NodeDef>* new_nodes) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSbackportsDTcc mht_3(mht_3_v, 251, "", "./tensorflow/tools/graph_transforms/backports.cc", "lambda");

        const NodeDef& tensor_array_v3_node = match.node;

        // All we need to do here is rename the op type, since the attributes
        // remain the same.
        NodeDef tensor_array_v2_node = tensor_array_v3_node;
        if (tensor_array_v3_node.op() == "TensorArrayV3") {
          tensor_array_v2_node.set_op("TensorArrayV2");
        } else {
          tensor_array_v2_node.set_op("TensorArrayGradV2");
        }

        // The v3 version has a second 'flow' output that's not present in v2,
        // so substitute a dummy constant instead in any places that use it.
        NodeDef replacement_flow_node;
        replacement_flow_node.set_op("Const");
        SetNodeAttr("dtype", DT_FLOAT, &replacement_flow_node);
        replacement_flow_node.set_name(tensor_array_v3_node.name() +
                                       "/replacement_flow_node");
        Tensor replacement_flow_tensor(DT_FLOAT, {});
        // I'm picking an arbitrary value for the gradient flow here, for lack
        // of a better alternative.
        replacement_flow_tensor.flat<float>()(0) = 1.0f;
        SetNodeTensorAttr<float>("value", replacement_flow_tensor,
                                 &replacement_flow_node);
        inputs_to_rename[tensor_array_v3_node.name() + ":1"] =
            replacement_flow_node.name();

        new_nodes->push_back(tensor_array_v2_node);
        new_nodes->push_back(replacement_flow_node);
        return Status::OK();
      },
      {true}, &replaced_graph_def));
  // Update the graph so that any nodes that referred to removed inputs now
  // pull from the substitute constants we've added.
  GraphDef renamed_graph_def;
  TF_RETURN_IF_ERROR(RenameNodeInputs(replaced_graph_def, inputs_to_rename,
                                      std::unordered_set<string>(),
                                      &renamed_graph_def));
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      renamed_graph_def,
      {"TensorArrayWriteV3|TensorArrayReadV3|TensorArrayGatherV3|"
       "TensorArrayScatterV3|TensorArrayConcatV3|TensorArraySplitV3|"
       "TensorArraySizeV3|TensorArrayCloseV3"},
      [](const NodeMatch& match, const std::set<string>& input_nodes,
         const std::set<string>& output_nodes,
         std::vector<NodeDef>* new_nodes) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPSbackportsDTcc mht_4(mht_4_v, 300, "", "./tensorflow/tools/graph_transforms/backports.cc", "lambda");

        const NodeDef& v3_node = match.node;
        NodeDef v2_node = v3_node;
        v2_node.set_op(v3_node.op().substr(0, v3_node.op().size() - 1) + "2");
        new_nodes->push_back(v2_node);
        return Status::OK();
      },
      {true}, output_graph_def));
  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("backport_tensor_array_v3",
                         BackportTensorArrayV3Transform);

}  // namespace graph_transforms
}  // namespace tensorflow
