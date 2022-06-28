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

#ifndef TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_GEN_NODE_H_
#define TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_GEN_NODE_H_
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
class MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgen_nodeDTh {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgen_nodeDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgen_nodeDTh() {
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


#include <map>
#include <memory>
#include <unordered_map>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {
namespace grappler {
namespace graph_analyzer {

class GenNode;

// To find nodes by name.
using GenNodeMap = std::unordered_map<string, std::unique_ptr<GenNode>>;

// One node in the graph, in the form convenient for traversal and generation of
// subgraphs. It refers to the original NodeDef protobuf for most information
// and adds the extra enrichment.
//
// The graph building is 2-stage: first match a GenNode with each NodeDef and
// collect them into a map that finds them by name, then process the map,
// deep-parse the underlying NodeDefs and connect the GenNodes together.
class GenNode {
 public:
  // Will keep the pointer, so the underlying object must not be deleted while
  // GenNode is alive.
  explicit GenNode(const NodeDef* node);

  // Access wrappers.
  const string& name() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgen_nodeDTh mht_0(mht_0_v, 222, "", "./tensorflow/core/grappler/graph_analyzer/gen_node.h", "name");
 return node_->name(); }
  const string& opcode() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgen_nodeDTh mht_1(mht_1_v, 226, "", "./tensorflow/core/grappler/graph_analyzer/gen_node.h", "opcode");
 return node_->op(); }
  const NodeDef* node_def() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgen_nodeDTh mht_2(mht_2_v, 230, "", "./tensorflow/core/grappler/graph_analyzer/gen_node.h", "node_def");
 return node_; }

  // Parse the inputs of this node and update the map accordingly, creating the
  // links (i.e. edges, connections between nodes) in itself and in the nodes
  // it's linked to (the map itself is unchanged, only the nodes in it are
  // updated).
  Status ParseInputs(const GenNodeMap* map);

  // Does the full 2-stage build of the graph. The map should be initially
  // empty. The map keeps pointers to the nodes in source, so the source must
  // not be destroyed before the map.
  static Status BuildGraphInMap(const GraphDef& source, GenNodeMap* map);

  // The enrichment that constitutes the point of this class.

  // Representation of a connection on a node.
  class Port {
   public:
    // A port may be inbound or outbound.
    // Negative ids (canonically -1) mean a control port.
    Port(bool inbound, int32_t id) : value_(id << 1) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgen_nodeDTh mht_3(mht_3_v, 253, "", "./tensorflow/core/grappler/graph_analyzer/gen_node.h", "Port");

      if (inbound) {
        value_ |= 1;
      }
    }
    Port(const Port&) = default;
    Port& operator=(const Port&) = default;

    bool IsInbound() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgen_nodeDTh mht_4(mht_4_v, 264, "", "./tensorflow/core/grappler/graph_analyzer/gen_node.h", "IsInbound");
 return (value_ & 0x1); }

    bool IsControl() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgen_nodeDTh mht_5(mht_5_v, 269, "", "./tensorflow/core/grappler/graph_analyzer/gen_node.h", "IsControl");
 return (value_ < 0); }

    int32_t Id() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgen_nodeDTh mht_6(mht_6_v, 274, "", "./tensorflow/core/grappler/graph_analyzer/gen_node.h", "Id");

      // Arithmetic shift preserves the sign.
      return (value_ >> 1);
    }

    // Integer type used to represent the encoded port value.
    using IntPort = int32_t;

    // Returns the encoded form of this port, so that it can be used
    // as various map indexes.
    IntPort Encoded() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgen_nodeDTh mht_7(mht_7_v, 287, "", "./tensorflow/core/grappler/graph_analyzer/gen_node.h", "Encoded");
 return value_; }

    static Port Decode(IntPort encoded) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgen_nodeDTh mht_8(mht_8_v, 292, "", "./tensorflow/core/grappler/graph_analyzer/gen_node.h", "Decode");
 return Port(encoded); }

    bool operator==(const Port& other) const { return value_ == other.value_; }
    bool operator<(const Port& other) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgen_nodeDTh mht_9(mht_9_v, 298, "", "./tensorflow/core/grappler/graph_analyzer/gen_node.h", "operator<");
 return value_ < other.value_; }

    struct Hasher {
      size_t operator()(const Port& port) const noexcept {
        return hasher(port.Encoded());
      }
      std::hash<int32_t> hasher;
    };

    // Convenient for printing. I've really wanted it to be implicit but
    // ClangTidy insists on making it explicit.
    explicit operator string() const;

   private:
    explicit Port(IntPort value) : value_(value) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgen_nodeDTh mht_10(mht_10_v, 315, "", "./tensorflow/core/grappler/graph_analyzer/gen_node.h", "Port");
}

    IntPort value_;
  };

  struct LinkTarget {
    GenNode* node;  // Node where this link points.
    Port port;      // Port on the remote side of this link.

    LinkTarget(GenNode* a_node, Port a_port) : node(a_node), port(a_port) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgen_nodeDTh mht_11(mht_11_v, 327, "", "./tensorflow/core/grappler/graph_analyzer/gen_node.h", "LinkTarget");
}
  };
  // All the links that are connected to the same port of this node
  // are collected in one vector. A link is an edge of the graph that connects
  // 2 nodes. Each of the connected nodes has its own perspective on the link,
  // seeing its local port, remote port and the remote node. The direction of
  // the link is encoded in the ports, one port is always incoming and another
  // one outgoing.
  using LinkTargetVector = std::vector<LinkTarget>;
  // Both inputs and outputs are stored in the same map.
  using LinkMap = std::unordered_map<Port, LinkTargetVector, Port::Hasher>;

  // Access to the link map.
  const LinkMap& links() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgen_nodeDTh mht_12(mht_12_v, 343, "", "./tensorflow/core/grappler/graph_analyzer/gen_node.h", "links");
 return links_; }

  // Check whether the port is an input (including the controls) with multiple
  // connections. Such inputs get handled in a special way when building the
  // subgraphs, in an "all or nothing" fashion.
  bool IsMultiInput(Port port) const;

  // When building the subgraphs, must include either all non-control inputs of
  // this node into the subgraph or none of them. This happens when at least one
  // of the inputs is a multi-input (or if the opcode is commutative, thus
  // treating all the inputs as one multi-input).
  bool AllInputsOrNone() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSgraph_analyzerPSgen_nodeDTh mht_13(mht_13_v, 357, "", "./tensorflow/core/grappler/graph_analyzer/gen_node.h", "AllInputsOrNone");
 return all_inputs_or_none_; }

 private:
  const NodeDef* node_;
  // Becomes valid only after ParseInputs().
  const OpDef* op_;

  // The opcode has a complicated structure of input args, with multi-input args
  // that are not commutative. This means that to make sense, the subgraphs that
  // include this node must also include either all its inputs or none of them.
  bool all_inputs_or_none_ = false;

  LinkMap links_;
};

}  // end namespace graph_analyzer
}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_GRAPH_ANALYZER_GEN_NODE_H_
