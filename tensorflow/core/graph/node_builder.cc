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
class MHTracer_DTPStensorflowPScorePSgraphPSnode_builderDTcc {
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
   MHTracer_DTPStensorflowPScorePSgraphPSnode_builderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgraphPSnode_builderDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/graph/node_builder.h"

#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {

NodeBuilder::NodeOut::NodeOut(Node* n, int32_t i)  // NOLINT(runtime/explicit)
    : node(n),
      error(false),
      name(node != nullptr ? node->name() : (error = true, "")),
      index(i),
      dt(SafeGetOutput(node, i, &error)) {}

NodeBuilder::NodeOut::NodeOut(OutputTensor t) : NodeOut(t.node, t.index) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgraphPSnode_builderDTcc mht_0(mht_0_v, 206, "", "./tensorflow/core/graph/node_builder.cc", "NodeBuilder::NodeOut::NodeOut");
}

NodeBuilder::NodeOut::NodeOut(StringPiece n, int32_t i, DataType t)
    : node(nullptr), error(false), name(n), index(i), dt(t) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgraphPSnode_builderDTcc mht_1(mht_1_v, 212, "", "./tensorflow/core/graph/node_builder.cc", "NodeBuilder::NodeOut::NodeOut");
}

NodeBuilder::NodeOut::NodeOut()
    : node(nullptr), error(true), index(0), dt(DT_FLOAT) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgraphPSnode_builderDTcc mht_2(mht_2_v, 218, "", "./tensorflow/core/graph/node_builder.cc", "NodeBuilder::NodeOut::NodeOut");
}

NodeBuilder::NodeBuilder(StringPiece name, StringPiece op_name,
                         const OpRegistryInterface* op_registry,
                         const NodeDebugInfo* debug)
    : def_builder_(name, op_name, op_registry, debug) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgraphPSnode_builderDTcc mht_3(mht_3_v, 226, "", "./tensorflow/core/graph/node_builder.cc", "NodeBuilder::NodeBuilder");
}

NodeBuilder::NodeBuilder(StringPiece name, const OpDef* op_def)
    : def_builder_(name, op_def) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgraphPSnode_builderDTcc mht_4(mht_4_v, 232, "", "./tensorflow/core/graph/node_builder.cc", "NodeBuilder::NodeBuilder");
}

NodeBuilder::NodeBuilder(const NodeDefBuilder& def_builder)
    : def_builder_(def_builder) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgraphPSnode_builderDTcc mht_5(mht_5_v, 238, "", "./tensorflow/core/graph/node_builder.cc", "NodeBuilder::NodeBuilder");
}

NodeBuilder& NodeBuilder::Input(Node* src_node, int src_index) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgraphPSnode_builderDTcc mht_6(mht_6_v, 243, "", "./tensorflow/core/graph/node_builder.cc", "NodeBuilder::Input");

  inputs_.emplace_back(src_node, src_index);
  DataType dt;
  if (GetOutputType(src_node, src_index, &dt)) {
    def_builder_.Input(src_node->name(), src_index, dt);
  }
  return *this;
}

NodeBuilder& NodeBuilder::Input(NodeOut src) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgraphPSnode_builderDTcc mht_7(mht_7_v, 255, "", "./tensorflow/core/graph/node_builder.cc", "NodeBuilder::Input");

  if (src.error) {
    AddIndexError(src.node, src.index);
  } else {
    inputs_.emplace_back(src.node, src.index);
    def_builder_.Input(src.name, src.index, src.dt);
  }
  return *this;
}

NodeBuilder& NodeBuilder::Input(gtl::ArraySlice<NodeOut> src_list) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgraphPSnode_builderDTcc mht_8(mht_8_v, 268, "", "./tensorflow/core/graph/node_builder.cc", "NodeBuilder::Input");

  std::vector<NodeDefBuilder::NodeOut> srcs;
  srcs.reserve(src_list.size());
  for (const auto& node_out : src_list) {
    if (node_out.error) {
      AddIndexError(node_out.node, node_out.index);
    } else {
      srcs.emplace_back(node_out.name, node_out.index, node_out.dt);
      inputs_.emplace_back(node_out.node, node_out.index);
    }
  }
  def_builder_.Input(gtl::ArraySlice<NodeDefBuilder::NodeOut>(srcs));
  return *this;
}

NodeBuilder& NodeBuilder::ControlInput(Node* src_node) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgraphPSnode_builderDTcc mht_9(mht_9_v, 286, "", "./tensorflow/core/graph/node_builder.cc", "NodeBuilder::ControlInput");

  control_inputs_.emplace_back(src_node);
  def_builder_.ControlInput(src_node->name());
  return *this;
}

NodeBuilder& NodeBuilder::ControlInputs(gtl::ArraySlice<Node*> src_nodes) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgraphPSnode_builderDTcc mht_10(mht_10_v, 295, "", "./tensorflow/core/graph/node_builder.cc", "NodeBuilder::ControlInputs");

  control_inputs_.insert(control_inputs_.end(), src_nodes.begin(),
                         src_nodes.end());
  for (const Node* src_node : src_nodes) {
    def_builder_.ControlInput(src_node->name());
  }
  return *this;
}

NodeBuilder& NodeBuilder::Device(StringPiece device_spec) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgraphPSnode_builderDTcc mht_11(mht_11_v, 307, "", "./tensorflow/core/graph/node_builder.cc", "NodeBuilder::Device");

  def_builder_.Device(device_spec);
  return *this;
}

NodeBuilder& NodeBuilder::AssignedDevice(StringPiece device) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSgraphPSnode_builderDTcc mht_12(mht_12_v, 315, "", "./tensorflow/core/graph/node_builder.cc", "NodeBuilder::AssignedDevice");

  assigned_device_ = string(device);
  return *this;
}

NodeBuilder& NodeBuilder::XlaCluster(StringPiece xla_cluster) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSgraphPSnode_builderDTcc mht_13(mht_13_v, 323, "", "./tensorflow/core/graph/node_builder.cc", "NodeBuilder::XlaCluster");

  def_builder_.Attr("_XlaCluster", xla_cluster);
  return *this;
}

StatusOr<Node*> NodeBuilder::Finalize(Graph* graph, bool consume) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSgraphPSnode_builderDTcc mht_14(mht_14_v, 331, "", "./tensorflow/core/graph/node_builder.cc", "NodeBuilder::Finalize");

  Node* out;
  TF_RETURN_IF_ERROR(Finalize(graph, &out, consume));
  return out;
}

Status NodeBuilder::Finalize(Graph* graph, Node** created_node, bool consume) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSgraphPSnode_builderDTcc mht_15(mht_15_v, 340, "", "./tensorflow/core/graph/node_builder.cc", "NodeBuilder::Finalize");

  // In case of error, set *created_node to nullptr.
  if (created_node != nullptr) {
    *created_node = nullptr;
  }
  if (!errors_.empty()) {
    return errors::InvalidArgument(absl::StrJoin(errors_, "\n"));
  }

  NodeDef node_def;
  TF_RETURN_IF_ERROR(def_builder_.Finalize(&node_def, consume));
  TF_RETURN_IF_ERROR(ValidateNodeDef(node_def, def_builder_.op_def()));
  TF_RETURN_IF_ERROR(
      CheckOpDeprecation(def_builder_.op_def(), graph->versions().producer()));

  TF_ASSIGN_OR_RETURN(Node * node, graph->AddNode(std::move(node_def)));

  node->set_assigned_device_name(assigned_device_);

  for (size_t i = 0; i < inputs_.size(); ++i) {
    if (inputs_[i].node != nullptr) {  // Skip back edges.
      graph->AddEdge(inputs_[i].node, inputs_[i].index, node, i);
    }
  }
  for (Node* control_input : control_inputs_) {
    graph->AddControlEdge(control_input, node);
  }

  if (created_node != nullptr) *created_node = node;

  return Status::OK();
}

void NodeBuilder::AddIndexError(const Node* node, int i) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSgraphPSnode_builderDTcc mht_16(mht_16_v, 376, "", "./tensorflow/core/graph/node_builder.cc", "NodeBuilder::AddIndexError");

  if (node == nullptr) {
    errors_.emplace_back(
        strings::StrCat("Attempt to add nullptr Node to node with type ",
                        def_builder_.op_def().name()));
  } else {
    errors_.emplace_back(strings::StrCat(
        "Attempt to add output ", i, " of ", node->name(), " not in range [0, ",
        node->num_outputs(), ") to node with type ",
        def_builder_.op_def().name(), ". Node: ", FormatNodeForError(*node)));
  }
}

bool NodeBuilder::GetOutputType(const Node* node, int i, DataType* dt) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSgraphPSnode_builderDTcc mht_17(mht_17_v, 392, "", "./tensorflow/core/graph/node_builder.cc", "NodeBuilder::GetOutputType");

  bool error;
  *dt = SafeGetOutput(node, i, &error);
  if (error) AddIndexError(node, i);
  return !error;
}

}  // namespace tensorflow
