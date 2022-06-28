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
class MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgen_nodeDTcc {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgen_nodeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgen_nodeDTcc() {
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

#include "tensorflow/core/grappler/graph_analyzer/gen_node.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/grappler/graph_analyzer/hash_tools.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"

namespace tensorflow {
namespace grappler {
namespace graph_analyzer {

GenNode::GenNode(const NodeDef* node) : node_(node), op_(nullptr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgen_nodeDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/grappler/graph_analyzer/gen_node.cc", "GenNode::GenNode");
}

Status GenNode::BuildGraphInMap(const GraphDef& source, GenNodeMap* map) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgen_nodeDTcc mht_1(mht_1_v, 202, "", "./tensorflow/core/grappler/graph_analyzer/gen_node.cc", "GenNode::BuildGraphInMap");

  for (const auto& n : source.node()) {
    const string& name = n.name();
    if (map->find(name) != map->end()) {
      // This error code looks more meaningful than ALREADY_EXISTS.
      return Status(error::INVALID_ARGUMENT,
                    "Duplicate node name '" + name + "'.");
    }
    (*map)[name] = absl::make_unique<GenNode>(&n);
  }
  // Now parse the links.
  for (const auto& mapit : *map) {
    Status st = mapit.second->ParseInputs(map);
    if (!st.ok()) {
      return st;
    }
  }
  return Status::OK();
}

Status GenNode::ParseInputs(const GenNodeMap* map) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgen_nodeDTcc mht_2(mht_2_v, 225, "", "./tensorflow/core/grappler/graph_analyzer/gen_node.cc", "GenNode::ParseInputs");

  all_inputs_or_none_ = false;
  Status st = OpRegistry::Global()->LookUpOpDef(opcode(), &op_);
  if (!st.ok()) {
    return Status(
        error::INVALID_ARGUMENT,
        absl::StrFormat("Node '%s' contains an undefined operation '%s': %s",
                        name(), opcode(), st.error_message()));
  }

  int n_inputs = node_->input_size();

  int n_named_inputs = op_->input_arg_size();

  int n_multi_inputs = 0;
  for (const auto& inarg : op_->input_arg()) {
    if (!inarg.number_attr().empty() || !inarg.type_list_attr().empty()) {
      ++n_multi_inputs;
    }
  }
  bool is_commutative = grappler::IsCommutative(*node_);

  if (n_multi_inputs > 1 || (n_multi_inputs > 0 && n_named_inputs > 1)) {
    // Can't handle more than one multi-input at a time.
    // And can't handle the commutativeness of only some arguments
    // rather than all of them.
    is_commutative = false;
  }

  if (is_commutative) {
    // If truly commutative, can treat all the inputs as one multi-input.
    // It's possible to just treat the commutative nodes as AllInputsOrNone
    // but (1) this way is a bit more efficient and (2) I want to preserve this
    // more efficient code path that does all-or-none by a single input and
    // perhaps extend its use in the future.
    n_named_inputs = 1;
    all_inputs_or_none_ = false;
  } else if (n_multi_inputs > 0) {
    all_inputs_or_none_ = true;
  }

  for (int i = 0; i < n_inputs; ++i) {
    int other_position;
    string other_name = ParseNodeName(node_->input(i), &other_position);
    auto other_it = map->find(other_name);
    if (other_it == map->end()) {
      return Status(
          error::INVALID_ARGUMENT,
          absl::StrFormat(
              "Node '%s' input %d refers to a non-existing node '%s'.", name(),
              i, other_name));
    }
    GenNode* other_node = other_it->second.get();

    int this_position = other_position < 0 ? -1 : (is_commutative ? 0 : i);

    if (this_position >= 0 && n_multi_inputs == 0 &&
        this_position >= n_named_inputs) {
      return Status(
          error::INVALID_ARGUMENT,
          absl::StrFormat(
              "Node '%s' has a non-control input from '%s' at index %d but its "
              "operation '%s' defines only %d inputs.",
              name(), other_name, this_position, op_->name(), n_named_inputs));
    }

    Port this_port(/*inbound=*/true, this_position);
    Port other_port(/*inbound=*/false, other_position);

    links_[this_port].emplace_back(LinkTarget(other_node, other_port));
    other_node->links_[other_port].emplace_back(LinkTarget(this, this_port));
  }
  return Status::OK();
}

bool GenNode::IsMultiInput(Port port) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgen_nodeDTcc mht_3(mht_3_v, 303, "", "./tensorflow/core/grappler/graph_analyzer/gen_node.cc", "GenNode::IsMultiInput");

  if (!port.IsInbound()) {
    return false;
  }
  auto it = links_.find(port);
  if (it == links_.end()) {
    return false;  // Shouldn't happen.
  }
  return (it->second.size() > 1);
}

GenNode::Port::operator string() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgen_nodeDTcc mht_4(mht_4_v, 317, "", "./tensorflow/core/grappler/graph_analyzer/gen_node.cc", "string");

  string result = this->IsInbound() ? "i" : "o";
  if (this->IsControl()) {
    result.append("C");
  } else {
    result.append(absl::StrFormat("%d", this->Id()));
  }
  return result;
}

}  // end namespace graph_analyzer
}  // end namespace grappler
}  // end namespace tensorflow
