/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_UTILS_GRAPH_VIEW_INTERNAL_H_
#define TENSORFLOW_CORE_GRAPPLER_UTILS_GRAPH_VIEW_INTERNAL_H_
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
class MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh {
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
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh() {
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


#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/map_util.h"

namespace tensorflow {
namespace grappler {
namespace utils {
namespace internal {

constexpr int kMissingSlot = -2;
constexpr int kMissingIndex = -1;
constexpr int kNodeNamePresent = -1;

// NodeIndexAndPortIndex is a helper class that represents fanins and fanouts
// of a node.
template <typename NodeViewT, typename GraphViewT>
class NodeIndexAndPortIndex {
 public:
  NodeIndexAndPortIndex()
      : graph_view_(nullptr),
        node_index_(kMissingIndex),
        port_index_(kMissingSlot) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_0(mht_0_v, 217, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "NodeIndexAndPortIndex");
}
  NodeIndexAndPortIndex(GraphViewT* graph_view, int node_index, int port_index)
      : graph_view_(graph_view),
        node_index_(node_index),
        port_index_(port_index) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_1(mht_1_v, 224, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "NodeIndexAndPortIndex");
}

  bool operator==(const NodeIndexAndPortIndex& other) const {
    return port_index_ == other.port_index_ &&
           node_index_ == other.node_index_ && graph_view_ == other.graph_view_;
  }

  template <typename Hash>
  friend Hash AbslHashValue(Hash h, const NodeIndexAndPortIndex& n) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_2(mht_2_v, 235, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "AbslHashValue");

    return Hash::combine(std::move(h), n.node_index_, n.port_index_);
  }

  // Returns NodeView from `graph_view_` at `node_index_`.
  NodeViewT* node_view() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_3(mht_3_v, 243, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "node_view");

    if (graph_view_ == nullptr) {
      return nullptr;
    }
    return graph_view_->GetNode(node_index_);
  }

  // Returns node index in graph.
  int node_index() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_4(mht_4_v, 254, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "node_index");
 return node_index_; }

  // Returns input/output port index.
  int index() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_5(mht_5_v, 260, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "index");
 return port_index_; }

 protected:
  GraphViewT* graph_view_;
  int node_index_;
  int port_index_;
};

// NodeDefAndPortIndex is a helper class that represents fanins hashed with
// pointer stability using the fanin's NodeDef.
class NodeDefAndPortIndex {
 public:
  NodeDefAndPortIndex(const NodeDef* node_def, int port_index)
      : node_def_(node_def), port_index_(port_index) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_6(mht_6_v, 276, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "NodeDefAndPortIndex");
}

  bool operator==(const NodeDefAndPortIndex& other) const {
    return node_def_ == other.node_def_ && port_index_ == other.port_index_;
  }

  template <typename Hash>
  friend Hash AbslHashValue(Hash h, const NodeDefAndPortIndex& n) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_7(mht_7_v, 286, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "AbslHashValue");

    return Hash::combine(std::move(h), n.node_def_, n.port_index_);
  }

 private:
  const NodeDef* node_def_;
  int port_index_;
};

// NodeViewInternal is a helper class to simplify graph traversal. It creates
// a view of a node and associated fanins and fanouts from the NodeDef
// protocol buffer.
//
// There are two public classes implementing NodeViewInternal:
//
// - NodeView: constructed from `const NodeDef` and doesn't allow mutating the
//   underlying node.
// - MutableNodeView: constructed from `NodeDef` and allows mutating the
//   underlying node.
//
// --------------------------- !!! WARNING !!! ---------------------------------
//     Modifying the node outside of implementations of NodeViewInternal
//     (i.e. modifying inputs of the NodeDef directly) may leave the NodeView
//     in an inconsistent/invalid state.
// -----------------------------------------------------------------------------
//
template <typename FaninViewT, typename FanoutViewT, typename GraphViewT,
          bool IsConst>
class NodeViewInternal {
 private:
  using NodeDefT =
      typename std::conditional<IsConst, const NodeDef, NodeDef>::type;

 public:
  explicit NodeViewInternal(GraphViewT* graph_view, int node_index)
      : graph_view_(graph_view),
        node_index_(node_index),
        attrs_(AttrSlice(graph_view->graph()->node(node_index))) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_8(mht_8_v, 326, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "NodeViewInternal");
}

  NodeViewInternal()
      : graph_view_(nullptr), node_index_(kMissingIndex), attrs_(AttrSlice()) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_9(mht_9_v, 332, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "NodeViewInternal");
}

  virtual ~NodeViewInternal() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_10(mht_10_v, 337, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "~NodeViewInternal");
}

  NodeViewInternal(NodeViewInternal&&) = default;
  NodeViewInternal& operator=(NodeViewInternal&&) = default;

  bool operator==(const NodeViewInternal& other) const {
    return node_index_ == other.node_index_ && graph_view_ == other.graph_view_;
  }

  template <typename Hash>
  friend Hash AbslHashValue(Hash h, const NodeViewInternal& n) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_11(mht_11_v, 350, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "AbslHashValue");

    return Hash::combine(std::move(h), n.node_index_);
  }

  // Returns NodeDef of view.
  virtual NodeDefT* node() const = 0;

  // Returns index of node in GraphDef/GraphView.
  int node_index() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_12(mht_12_v, 361, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "node_index");
 return node_index_; }

  // Returns the name of the node.
  const string& GetName() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_13(mht_13_v, 367, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "GetName");
 return node()->name(); }

  // Returns the op of the node.
  const string& GetOp() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_14(mht_14_v, 373, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "GetOp");
 return node()->op(); }

  // Returns the device set for the node.
  const string& GetDevice() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_15(mht_15_v, 379, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "GetDevice");
 return node()->device(); }

  // Returns all regular fanins, based on ordering in the node.
  const std::vector<FanoutViewT>& GetRegularFanins() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_16(mht_16_v, 385, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "GetRegularFanins");

    return regular_fanins_;
  }

  // Returns a regular fanin based on input index. If no such fanin exist, a
  // missing fanin is returned, with no NodeView set and an index of -2.
  const FanoutViewT& GetRegularFanin(int i) const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_17(mht_17_v, 394, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "GetRegularFanin");

    int regular_fanins_size = regular_fanins_.size();
    if (i < 0 || i >= regular_fanins_size) {
      return GetMissingFanin();
    }
    return regular_fanins_[i];
  }

  // Returns all controlling fanins, based on ordering in the node.
  const std::vector<FanoutViewT>& GetControllingFanins() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_18(mht_18_v, 406, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "GetControllingFanins");

    return controlling_fanins_;
  }

  // Returns all regular fanouts.
  const std::vector<std::vector<FaninViewT>>& GetRegularFanouts() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_19(mht_19_v, 414, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "GetRegularFanouts");

    return regular_fanouts_by_port_;
  }

  // Returns a regular fanout(s) based on output index. If no such output index
  // exists, no fanouts will be returned.
  const std::vector<FaninViewT>& GetRegularFanout(int i) const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_20(mht_20_v, 423, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "GetRegularFanout");

    int regular_fanouts_by_port_size = regular_fanouts_by_port_.size();
    if (i < 0 || i >= regular_fanouts_by_port_size) {
      return GetMissingFanout();
    }
    return regular_fanouts_by_port_[i];
  }

  // Returns all controlled fanouts.
  const std::vector<FaninViewT>& GetControlledFanouts() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_21(mht_21_v, 435, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "GetControlledFanouts");

    return controlled_fanouts_;
  }

  // Returns the number of regular fanins.
  int NumRegularFanins() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_22(mht_22_v, 443, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "NumRegularFanins");
 return regular_fanins_.size(); }

  // Returns the number of controlling fanins.
  int NumControllingFanins() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_23(mht_23_v, 449, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "NumControllingFanins");
 return controlling_fanins_.size(); }

  // Returns the number of regular fanouts.
  int NumRegularFanouts() const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_24(mht_24_v, 455, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "NumRegularFanouts");
 return num_regular_fanouts_; }

  // Returns the number of controlled fanouts.
  int NumControlledFanouts() const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_25(mht_25_v, 461, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "NumControlledFanouts");
 return controlled_fanouts_.size(); }

  // Checks if a fanin exists for the node.
  virtual bool HasFanin(const FanoutViewT& fanin) const = 0;

  // Checks if a fanout exists for the node.
  virtual bool HasFanout(const FaninViewT& fanout) const = 0;

  // Returns an attribute of the node by key. If no attribute for such key
  // exists, a `nullptr` is returned.
  const AttrValue* GetAttr(absl::string_view attr_name) const {
   std::vector<std::string> mht_26_v;
   mht_26_v.push_back("attr_name: \"" + std::string(attr_name.data(), attr_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_26(mht_26_v, 475, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "GetAttr");

    return attrs_.Find(attr_name);
  }

  // Returns all attributes of the node.
  const AttrSlice& GetAttrs() const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_27(mht_27_v, 483, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "GetAttrs");
 return attrs_; }

  // Returns the number of attributes in the node.
  int NumAttrs() const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_28(mht_28_v, 489, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "NumAttrs");
 return attrs_.size(); }

  // Checks if an attribute exist in the node.
  bool HasAttr(absl::string_view attr_name) const {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("attr_name: \"" + std::string(attr_name.data(), attr_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_29(mht_29_v, 496, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "HasAttr");

    return attrs_.Find(attr_name) != nullptr;
  }

 protected:
  virtual inline const FanoutViewT& GetMissingFanin() const = 0;
  virtual inline const std::vector<FaninViewT>& GetMissingFanout() const = 0;

  std::vector<FanoutViewT> regular_fanins_;
  std::vector<FanoutViewT> controlling_fanins_;
  std::vector<std::vector<FaninViewT>> regular_fanouts_by_port_;
  int num_regular_fanouts_ = 0;
  std::vector<FaninViewT> controlled_fanouts_;

  GraphViewT* graph_view_;
  int node_index_;
  AttrSlice attrs_;
};

// GraphViewInternal is a helper class to simplify graph traversal. It creates
// a view of the nodes and associated fanins and fanouts from the GraphDef
// protocol buffer.
//
// There are two public classes implementing GraphViewInternal:
//
// - GraphView: constructed from `const GraphDef` and doesn't allow mutating
//   the underlying graph and its nodes.
// - MutableGraphView: constructed from `GraphDef` and allows mutating the
//   underlying graph and its nodes.
//
// --------------------------- !!! WARNING !!! ---------------------------------
//     Modifying the graph outside of implementations of GraphViewInternal
//     (i.e. removing nodes from the GraphDef directly) may lead to
//     segfaults! Guaranteed by absl::string_view!
// -----------------------------------------------------------------------------
//
template <typename NodeViewT, typename FaninViewT, typename FanoutViewT,
          bool IsConst>
class GraphViewInternal {
 private:
  using GraphDefT =
      typename std::conditional<IsConst, const GraphDef, GraphDef>::type;

 public:
  explicit GraphViewInternal(GraphDefT* graph) : graph_(graph) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_30(mht_30_v, 543, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "GraphViewInternal");
}
  virtual ~GraphViewInternal() {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_31(mht_31_v, 547, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "~GraphViewInternal");
}

  bool operator==(const GraphViewInternal& other) const {
    return graph_ == other.graph_;
  }

  GraphDefT* graph() const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_32(mht_32_v, 556, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "graph");
 return graph_; }

  // Finds node by index in the graph. If no such node exists in the graph, a
  // `nullptr` is returned.
  const NodeViewT* GetNode(int node_index) const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_33(mht_33_v, 563, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "GetNode");

    int nodes_size = nodes_.size();
    if (node_index < 0 || node_index >= nodes_size) {
      return nullptr;
    }
    return &nodes_[node_index];
  }

  NodeViewT* GetNode(int node_index) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_34(mht_34_v, 574, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "GetNode");

    int nodes_size = nodes_.size();
    if (node_index < 0 || node_index >= nodes_size) {
      return nullptr;
    }
    return &nodes_[node_index];
  }

  // Finds node by name. If no such node exists in the graph, a `nullptr` is
  // returned.
  const NodeViewT* GetNode(absl::string_view node_name) const {
   std::vector<std::string> mht_35_v;
   mht_35_v.push_back("node_name: \"" + std::string(node_name.data(), node_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_35(mht_35_v, 588, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "GetNode");

    auto it = node_index_by_name_.find(node_name);
    if (it == node_index_by_name_.end()) {
      return nullptr;
    }
    return &nodes_[it->second];
  }

  NodeViewT* GetNode(absl::string_view node_name) {
   std::vector<std::string> mht_36_v;
   mht_36_v.push_back("node_name: \"" + std::string(node_name.data(), node_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_36(mht_36_v, 600, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "GetNode");

    auto it = node_index_by_name_.find(node_name);
    if (it == node_index_by_name_.end()) {
      return nullptr;
    }
    return &nodes_[it->second];
  }

  // Returns all nodes (as NodeView) in the graph.
  const std::vector<NodeViewT>& GetNodes() const {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_37(mht_37_v, 612, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "GetNodes");
 return nodes_; }

  // Checks if a node by name exists in the graph.
  bool HasNode(absl::string_view node_name) const {
   std::vector<std::string> mht_38_v;
   mht_38_v.push_back("node_name: \"" + std::string(node_name.data(), node_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_38(mht_38_v, 619, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "HasNode");

    return node_index_by_name_.contains(node_name);
  }

  // Returns the number of nodes in the graph.
  int NumNodes() const {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_39(mht_39_v, 627, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "NumNodes");
 return nodes_.size(); }

 protected:
  // Reset allocated node vector and node map in case of failure.
  void Reset() {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_40(mht_40_v, 634, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "Reset");

    std::vector<NodeViewT>().swap(nodes_);
    absl::flat_hash_map<absl::string_view, int>().swap(node_index_by_name_);
  }

  // nodes_[i] is a view of graph_.{mutable_}node(i).
  std::vector<NodeViewT> nodes_;
  absl::flat_hash_map<absl::string_view, int> node_index_by_name_;
  GraphDefT* graph_;
  const FanoutViewT missing_fanin_;
  const std::vector<FaninViewT> missing_fanout_;
};

inline SafeTensorId EmptyTensorId() {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_41(mht_41_v, 650, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "EmptyTensorId");

  return SafeTensorId("", internal::kMissingSlot);
}

inline bool IsEmptyTensorId(const TensorId tensor_id) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_42(mht_42_v, 657, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "IsEmptyTensorId");

  return tensor_id.node().empty() &&
         tensor_id.index() == internal::kMissingSlot;
}

// NodeViewDiff is a helper struct holding changes to be made to an existing
// node in GraphViewT. This should not be initialized or be used directly.
template <typename GraphViewT>
struct NodeViewDiff {
  explicit NodeViewDiff(GraphViewT* graph_view, int node_index)
      : graph_view(graph_view), node_index(node_index) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_43(mht_43_v, 670, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "NodeViewDiff");
}

  GraphViewT* graph_view;
  int node_index;
  string name;
  bool update_name = false;
  string op;
  bool update_op = false;
  string device;
  bool update_device = false;
  // Fanins to append after existing regular fanins.
  std::vector<SafeTensorId> regular_inputs_to_add;
  // Number of fanins to be appended. This is used for a quick comparison with
  // `regular_inputs_to_add` for if there will be any missing inputs in the
  // updated node.
  int num_regular_inputs_to_add = 0;
  // Fanins to update inplace.
  std::map<int, SafeTensorId> regular_inputs_to_update;
  // Fanins from end of regular fanins to remove. This keeps track of existing
  // regular fanins in the original node to remove.
  std::vector<bool> regular_inputs_to_remove;
  // Number of fanins marked for removal. This is used for a quick comparison
  // with `regular_inputs_to_remove` for if there will be any missing inputs
  // in the updated node.
  int num_regular_inputs_to_remove = 0;
  absl::flat_hash_set<string> controlling_inputs_to_add;
  std::set<int> controlling_inputs_to_remove;
  absl::flat_hash_map<string, AttrValue> attrs_to_add;
  absl::flat_hash_set<string> attrs_to_remove;
  // AttrValueMap constructor and destructor are very expensive, we will
  // initialize it lazily only if needed.
  absl::optional<AttrValueMap> processed_attrs;
};

// Updates node name. If `name` is the same as the name in the original node,
// the field will be cleared in the diff.
template <typename GraphViewT>
inline bool UpdateName(NodeViewDiff<GraphViewT>* diff, absl::string_view name) {
   std::vector<std::string> mht_44_v;
   mht_44_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_44(mht_44_v, 711, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "UpdateName");

  if (diff->graph_view->GetNode(diff->node_index)->GetName() == name) {
    diff->name.clear();
    diff->update_name = false;
  } else {
    diff->name = string(name);
    diff->update_name = true;
  }
  return true;
}

// Updates node op. If `op` is the same as the op in the original node, the
// field will be cleared in the diff.
template <typename GraphViewT>
inline bool UpdateOp(NodeViewDiff<GraphViewT>* diff, absl::string_view op) {
   std::vector<std::string> mht_45_v;
   mht_45_v.push_back("op: \"" + std::string(op.data(), op.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_45(mht_45_v, 729, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "UpdateOp");

  if (diff->graph_view->GetNode(diff->node_index)->GetOp() == op) {
    diff->op.clear();
    diff->update_op = false;
  } else {
    diff->op = string(op);
    diff->update_op = true;
  }
  return true;
}

// Updates node device. If `device` is the same as the device in the original
// node, the field will be cleared in the diff.
template <typename GraphViewT>
inline bool UpdateDevice(NodeViewDiff<GraphViewT>* diff,
                         absl::string_view device) {
   std::vector<std::string> mht_46_v;
   mht_46_v.push_back("device: \"" + std::string(device.data(), device.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_46(mht_46_v, 748, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "UpdateDevice");

  if (diff->graph_view->GetNode(diff->node_index)->GetDevice() == device) {
    diff->device.clear();
    diff->update_device = false;
  } else {
    diff->device = string(device);
    diff->update_device = true;
  }
  return true;
}

// Adds or updates value in vector `v` at index `i`. This will also resize the
// vector if index `i` is out of bounds, padding the vector with
// `default_value`. Returns true if a new value was appended or if an update
// occurred where an existing value was changed from `default_value`.
template <typename T, typename U>
inline bool AddOrUpdateAtIndex(std::vector<T>* v, int i, const U& value,
                               const T& default_value) {
  int v_size = v->size();
  if (i > v_size) {
    // Resize to include `value`, filling the newly introduced gap with
    // `default_value` for later checks of validity (gaps in vector).
    v->reserve(i + 1);
    v->resize(i, default_value);
    v->push_back({value});
  } else if (i == v_size) {
    // Vector is large enough, simply append `value` to the end.
    v->push_back({value});
  } else {
    // Update existing value.
    bool updated = (*v)[i] == default_value;
    (*v)[i] = {value};
    return updated;
  }
  return true;
}

// Checks if a node with name `node_name` will exist in the final mutated graph.
template <typename GraphViewT>
inline bool CheckNodeNameExists(
    absl::string_view node_name,
    const absl::flat_hash_map<absl::string_view, int>& updated_node_names,
    const GraphViewT* graph_view) {
   std::vector<std::string> mht_47_v;
   mht_47_v.push_back("node_name: \"" + std::string(node_name.data(), node_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_47(mht_47_v, 794, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "CheckNodeNameExists");

  auto it = updated_node_names.find(node_name);
  if (it != updated_node_names.end()) {
    return it->second == kNodeNamePresent;
  }
  return graph_view->HasNode(node_name);
}

// Adds or updates regular fanin at `index` of regular fanins. If `index` is
// less than the number of regular fanins in the original node, the fanin at
// `index` in the original node will be updated with `fanin` if the fanin
// differs. If `index` is greater than or equal to the number of regular fanins,
// `fanin` will be added beyond the end of regular fanins at `index`.
template <typename GraphViewT>
inline bool AddOrUpdateRegularFanin(NodeViewDiff<GraphViewT>* diff, int index,
                                    const TensorId& fanin) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_48(mht_48_v, 812, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "AddOrUpdateRegularFanin");

  if (index < 0) {
    // Not a valid index for regular fanins.
    return false;
  }
  auto* node_view = diff->graph_view->GetNode(diff->node_index);
  const int num_regular_fanins = node_view->NumRegularFanins();
  if (index < num_regular_fanins) {  // Updating existing fanins.
    // Calculate (relative) index from end of regular fanins, from absolute
    // index from beginning of regular fanins.
    const int relative_removal_index = num_regular_fanins - index - 1;
    // Check if at relative index fanin was already marked for removal.
    int diff_regular_inputs_to_remove_size =
        diff->regular_inputs_to_remove.size();
    if (relative_removal_index < diff_regular_inputs_to_remove_size &&
        diff->regular_inputs_to_remove[relative_removal_index]) {
      // Unmark fanin for removal.
      diff->regular_inputs_to_remove[relative_removal_index] = false;
      --diff->num_regular_inputs_to_remove;
    }
    const auto& existing_fanin = node_view->GetRegularFanin(index);
    if (existing_fanin.index() != fanin.index() ||
        existing_fanin.node_view()->GetName() != fanin.node()) {
      // Update fanin if it is different from original fanin in node.
      gtl::InsertOrUpdate(&diff->regular_inputs_to_update, index,
                          SafeTensorId(fanin));
    }
  } else {
    // Add fanin beyond current fanin range.
    const int relative_add_index = index - num_regular_fanins;
    if (AddOrUpdateAtIndex(&diff->regular_inputs_to_add, relative_add_index,
                           fanin, EmptyTensorId())) {
      // New fanin was added.
      ++diff->num_regular_inputs_to_add;
    }
  }
  return true;
}

// Remove regular fanin at `index` of regular fanins. This can remove existing
// fanins and updated/added fanins via AddOrUpdateRegularFanins.
template <typename GraphViewT>
inline bool RemoveRegularFanin(NodeViewDiff<GraphViewT>* diff, int index) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_49(mht_49_v, 857, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "RemoveRegularFanin");

  if (index < 0) {
    // Not a valid index for regular fanins.
    return false;
  }
  auto* node_view = diff->graph_view->GetNode(diff->node_index);
  const int num_regular_fanins = node_view->NumRegularFanins();
  if (index < num_regular_fanins) {  // Removing existing fanins.
    // Remove updated fanin if it exists.
    diff->regular_inputs_to_update.erase(index);
    // Calculate (relative) index from end of regular fanins, from absolute
    // index from beginning of regular fanins.
    const int relative_removal_index = num_regular_fanins - index - 1;
    if (AddOrUpdateAtIndex(&diff->regular_inputs_to_remove,
                           relative_removal_index,
                           /*value=*/true, /*default_value=*/false)) {
      ++diff->num_regular_inputs_to_remove;
    }
  } else {
    // Relative index from end of regular fanins.
    const int relative_add_index = index - num_regular_fanins;
    int diff_regular_inputs_to_add_size = diff->regular_inputs_to_add.size();
    if (relative_add_index >= diff_regular_inputs_to_add_size ||
        IsEmptyTensorId(diff->regular_inputs_to_add[relative_add_index])) {
      // At relative index, appended regular fanin was already marked for
      // removal.
      return false;
    }
    // Remove added fanin.
    diff->regular_inputs_to_add[relative_add_index] = EmptyTensorId();
    --diff->num_regular_inputs_to_add;
  }
  return true;
}

// Adds controlling fanin. If the controlling fanin already exists in the
// original node, it will be dedupped. If the controlling fanin is marked for
// removal, this will reverse it.
template <typename GraphViewT>
inline bool AddControllingFanin(NodeViewDiff<GraphViewT>* diff,
                                int control_index,
                                absl::string_view fanin_node_name) {
   std::vector<std::string> mht_50_v;
   mht_50_v.push_back("fanin_node_name: \"" + std::string(fanin_node_name.data(), fanin_node_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_50(mht_50_v, 902, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "AddControllingFanin");

  if (control_index == kMissingIndex) {
    diff->controlling_inputs_to_add.emplace(fanin_node_name);
  } else {
    diff->controlling_inputs_to_remove.erase(control_index);
  }
  return true;
}

// Remove controlling fanin. If the controlling fanin does not exist in the
// original node and diff, nothing will happen. If the controlling fanin exists
// in the diff, it will be removed. Otherwise the controlling fanin will be
// marked for removal from the original node.
template <typename GraphViewT>
inline bool RemoveControllingFanin(NodeViewDiff<GraphViewT>* diff,
                                   int control_index,
                                   absl::string_view fanin_node_name) {
   std::vector<std::string> mht_51_v;
   mht_51_v.push_back("fanin_node_name: \"" + std::string(fanin_node_name.data(), fanin_node_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_51(mht_51_v, 922, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "RemoveControllingFanin");

  if (control_index == kMissingIndex) {
    diff->controlling_inputs_to_add.erase(fanin_node_name);
  } else {
    diff->controlling_inputs_to_remove.emplace(control_index);
  }
  return true;
}

// Adds or updates an attribute by name. If an attribute exist in the original
// node or diff (including those marked for removal), this will overwrite it.
template <typename GraphViewT>
inline bool AddOrUpdateAttribute(NodeViewDiff<GraphViewT>* diff,
                                 absl::string_view attr_name,
                                 const AttrValue& attr_value) {
   std::vector<std::string> mht_52_v;
   mht_52_v.push_back("attr_name: \"" + std::string(attr_name.data(), attr_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_52(mht_52_v, 940, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "AddOrUpdateAttribute");

  diff->attrs_to_add.empty() ? 0 : diff->attrs_to_remove.erase(attr_name);
  gtl::InsertOrUpdate(&diff->attrs_to_add, string(attr_name), attr_value);
  return true;
}

// Removes an attribute by name. If an attribute exist in the original node or
// diff, this will remove it.
template <typename GraphViewT>
inline bool RemoveAttribute(NodeViewDiff<GraphViewT>* diff,
                            absl::string_view attr_name) {
   std::vector<std::string> mht_53_v;
   mht_53_v.push_back("attr_name: \"" + std::string(attr_name.data(), attr_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_53(mht_53_v, 954, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "RemoveAttribute");

  const size_t num_erased =
      diff->attrs_to_add.empty() ? 0 : diff->attrs_to_add.erase(attr_name);
  auto* node_view = diff->graph_view->GetNode(diff->node_index);
  if (node_view->HasAttr(attr_name)) {
    diff->attrs_to_remove.emplace(attr_name);
    return true;
  }
  return num_erased > 0;
}

// Removes trailing values in vector `v` for values equal to `value`.
template <typename T>
inline void ResizeByTrimmingEndForValue(std::vector<T>* v, const T& value) {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_54(mht_54_v, 970, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "ResizeByTrimmingEndForValue");

  int curr_index = v->size();
  const int last_index = v->size() - 1;
  for (int i = last_index; i >= 0; --i) {
    if ((*v)[i] == value) {
      curr_index = i;
    } else {
      break;
    }
  }
  if (curr_index <= last_index) {
    v->resize(curr_index);
  }
}

// Checks if any changes are set in the diff.
template <typename GraphViewT>
inline bool IsEmpty(NodeViewDiff<GraphViewT>* diff) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_55(mht_55_v, 990, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "IsEmpty");

  ResizeByTrimmingEndForValue(&diff->regular_inputs_to_remove, false);
  ResizeByTrimmingEndForValue(&diff->regular_inputs_to_add, EmptyTensorId());
  return !diff->update_name && !diff->update_op && !diff->update_device &&
         diff->regular_inputs_to_add.empty() &&
         diff->regular_inputs_to_update.empty() &&
         diff->regular_inputs_to_remove.empty() &&
         diff->controlling_inputs_to_add.empty() &&
         diff->controlling_inputs_to_remove.empty() &&
         diff->attrs_to_add.empty() && diff->attrs_to_remove.empty();
}

// Resets and clears existing diff.
template <typename GraphViewT>
inline void Reset(NodeViewDiff<GraphViewT>* diff) {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_56(mht_56_v, 1007, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "Reset");

  diff->name.clear();
  diff->update_name = false;
  diff->op.clear();
  diff->update_op = false;
  diff->device.clear();
  diff->update_device = false;
  std::vector<SafeTensorId>().swap(diff->regular_inputs_to_add);
  diff->num_regular_inputs_to_add = false;
  std::map<int, SafeTensorId>().swap(diff->regular_inputs_to_update);
  std::vector<bool>().swap(diff->regular_inputs_to_remove);
  diff->num_regular_inputs_to_remove = 0;
  absl::flat_hash_set<string>().swap(diff->controlling_inputs_to_add);
  std::set<int>().swap(diff->controlling_inputs_to_remove);
  absl::flat_hash_map<string, AttrValue>().swap(diff->attrs_to_add);
  absl::flat_hash_set<string>().swap(diff->attrs_to_remove);
}

// Checks if changes to node will result in a valid node.
template <typename GraphViewT>
inline bool IsWellFormed(
    NodeViewDiff<GraphViewT>* diff,
    const absl::flat_hash_map<absl::string_view, int>& updated_node_names) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_57(mht_57_v, 1032, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "IsWellFormed");

  ResizeByTrimmingEndForValue(&diff->regular_inputs_to_remove, false);
  ResizeByTrimmingEndForValue(&diff->regular_inputs_to_add, EmptyTensorId());
  int diff_regular_inputs_to_add_size = diff->regular_inputs_to_add.size();
  if (diff_regular_inputs_to_add_size != diff->num_regular_inputs_to_add) {
    // Missing regular fanins in between appended fanins.
    return false;
  } else if (diff->num_regular_inputs_to_add > 0 &&
             !diff->regular_inputs_to_remove.empty()) {
    // Appending new fanins while removing existing fanins, resulting in missing
    // regular fanins in between.
    return false;
  } else if (static_cast<int>(diff->regular_inputs_to_remove.size()) !=
             diff->num_regular_inputs_to_remove) {
    // Regular fanins exist in between removed fanins.
    return false;
  }
  auto* node_view = diff->graph_view->GetNode(diff->node_index);
  const string& node_name =
      diff->update_name ? diff->name : node_view->GetName();
  auto invalid_node_name = [&](absl::string_view fanin_node_name) -> bool {
    return fanin_node_name == node_name ||
           !CheckNodeNameExists(fanin_node_name, updated_node_names,
                                diff->graph_view);
  };

  // Check if nodes of all updated and new fanins exist (from name) and if such
  // fanins do not introduce self loops. Note, this will not check for if
  // unmodified fanins exist.
  if (diff->update_name) {
    // If name of node was changed in node, check all fanins. Updated fanins are
    // checked for existence and self loops. Unmodified fanins are checked for
    // self loops.
    // `regular_inputs_to_update`, `controlling_inputs_to_remove` are sorted,
    // so iterators from these maps/sets can be incremented alongside iteration
    // and be used for comparisons.
    const int last_index =
        node_view->NumRegularFanins() - diff->num_regular_inputs_to_remove - 1;
    auto regular_to_update_it = diff->regular_inputs_to_update.begin();
    for (int i = 0; i <= last_index; ++i) {
      if (regular_to_update_it != diff->regular_inputs_to_update.end() &&
          regular_to_update_it->first < i) {
        ++regular_to_update_it;
      }
      if (regular_to_update_it != diff->regular_inputs_to_update.end() &&
          regular_to_update_it->first == i) {
        if (invalid_node_name(regular_to_update_it->second.node())) {
          return false;
        }
      } else {
        const string& regular_name =
            node_view->GetRegularFanin(i).node_view()->GetName();
        if (regular_name == node_name) {
          return false;
        }
      }
    }

    auto& controls = node_view->GetControllingFanins();
    const int num_controls = controls.size();
    auto control_to_remove_it = diff->controlling_inputs_to_remove.begin();
    for (int i = 0; i < num_controls; ++i) {
      if (control_to_remove_it != diff->controlling_inputs_to_remove.end() &&
          *control_to_remove_it < i) {
        ++control_to_remove_it;
      }
      if (control_to_remove_it != diff->controlling_inputs_to_remove.end() &&
          *control_to_remove_it == i) {
        // Control dependency marked for removal, can be ignored.
        continue;
      } else if (controls[i].node_view()->GetName() == node_name) {
        return false;
      }
    }
  } else {
    // Name of node was not changed, check only updated fanins under the
    // assumption prior fanins were valid.
    for (const auto& updated : diff->regular_inputs_to_update) {
      const string& fanin_name = updated.second.node();
      if (invalid_node_name(fanin_name)) {
        return false;
      }
    }
  }
  // Check appended regular fanins.
  for (const auto& regular : diff->regular_inputs_to_add) {
    if (invalid_node_name(regular.node())) {
      return false;
    }
  }
  // Check new controlling fanins.
  for (const auto& control : diff->controlling_inputs_to_add) {
    if (invalid_node_name(control)) {
      return false;
    }
  }

  return true;
}

// NewNode is a helper struct holding a new node to be added to a GraphViewT.
// This should not be initialized or be used directly.
template <typename GraphViewT>
struct NewNode {
  explicit NewNode(GraphViewT* graph_view, NodeDef&& node)
      : graph_view(graph_view), node(std::move(node)) {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_58(mht_58_v, 1140, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "NewNode");
}

  GraphViewT* graph_view;
  NodeDef node;
  std::vector<SafeTensorId> regular_fanins;
  int num_regular_fanins = 0;
  absl::flat_hash_set<string> controlling_fanins;
};

// Updates new node name.
template <typename GraphViewT>
inline void UpdateName(NewNode<GraphViewT>* new_node, absl::string_view name) {
   std::vector<std::string> mht_59_v;
   mht_59_v.push_back("name: \"" + std::string(name.data(), name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_59(mht_59_v, 1155, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "UpdateName");

  if (name.empty()) {
    new_node->node.clear_name();
  } else {
    new_node->node.set_name(string(name));
  }
}

// Updates new node op.
template <typename GraphViewT>
inline void UpdateOp(NewNode<GraphViewT>* new_node, absl::string_view op) {
   std::vector<std::string> mht_60_v;
   mht_60_v.push_back("op: \"" + std::string(op.data(), op.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_60(mht_60_v, 1169, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "UpdateOp");

  if (op.empty()) {
    new_node->node.clear_op();
  } else {
    new_node->node.set_op(string(op));
  }
}

// Updates new node device.
template <typename GraphViewT>
inline void UpdateDevice(NewNode<GraphViewT>* new_node,
                         absl::string_view device) {
   std::vector<std::string> mht_61_v;
   mht_61_v.push_back("device: \"" + std::string(device.data(), device.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_61(mht_61_v, 1184, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "UpdateDevice");

  if (device.empty()) {
    new_node->node.clear_device();
  } else {
    new_node->node.set_device(string(device));
  }
}

// Adds or updates regular fanin at `index` of regular fanins in the new node.
// If another fanin already exists at `index`, it will be replaced with `fanin`.
template <typename GraphViewT>
inline void AddOrUpdateRegularFanin(NewNode<GraphViewT>* new_node, int index,
                                    const TensorId& fanin) {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_62(mht_62_v, 1199, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "AddOrUpdateRegularFanin");

  if (index < 0) {
    // Not a valid index for regular fanins.
    return;
  } else if (AddOrUpdateAtIndex(&new_node->regular_fanins, index, fanin,
                                EmptyTensorId())) {
    ++new_node->num_regular_fanins;
  }
}

// Remove regular fanin at `index` of regular fanins in the new node. This can
// remove existing fanins and updated/added fanins via AddOrUpdateRegularFanins.
template <typename GraphViewT>
inline void RemoveRegularFanin(NewNode<GraphViewT>* new_node, int index) {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_63(mht_63_v, 1215, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "RemoveRegularFanin");

  int new_node_regular_fanins_size = new_node->regular_fanins.size();
  if (index < 0 || index >= new_node_regular_fanins_size ||
      IsEmptyTensorId(new_node->regular_fanins[index])) {
    return;
  }
  new_node->regular_fanins[index] = EmptyTensorId();
  --new_node->num_regular_fanins;
}

// Adds controlling fanin to new node.
template <typename GraphViewT>
inline void AddControllingFanin(NewNode<GraphViewT>* new_node,
                                absl::string_view fanin_node_name) {
   std::vector<std::string> mht_64_v;
   mht_64_v.push_back("fanin_node_name: \"" + std::string(fanin_node_name.data(), fanin_node_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_64(mht_64_v, 1232, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "AddControllingFanin");

  new_node->controlling_fanins.emplace(fanin_node_name);
}

// Removes controlling fanin to new node.
template <typename GraphViewT>
inline void RemoveControllingFanin(NewNode<GraphViewT>* new_node,
                                   absl::string_view fanin_node_name) {
   std::vector<std::string> mht_65_v;
   mht_65_v.push_back("fanin_node_name: \"" + std::string(fanin_node_name.data(), fanin_node_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_65(mht_65_v, 1243, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "RemoveControllingFanin");

  new_node->controlling_fanins.erase(fanin_node_name);
}

// Adds or updates an attribute by name to a new node.
template <typename GraphViewT>
inline void AddOrUpdateAttribute(NewNode<GraphViewT>* new_node,
                                 absl::string_view attr_name,
                                 const AttrValue& attr_value) {
   std::vector<std::string> mht_66_v;
   mht_66_v.push_back("attr_name: \"" + std::string(attr_name.data(), attr_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_66(mht_66_v, 1255, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "AddOrUpdateAttribute");

  gtl::InsertOrUpdate(new_node->node.mutable_attr(), string(attr_name),
                      attr_value);
}

// Removes an attribute by name to a new node.
template <typename GraphViewT>
inline void RemoveAttribute(NewNode<GraphViewT>* new_node,
                            absl::string_view attr_name) {
   std::vector<std::string> mht_67_v;
   mht_67_v.push_back("attr_name: \"" + std::string(attr_name.data(), attr_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_67(mht_67_v, 1267, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "RemoveAttribute");

  new_node->node.mutable_attr()->erase(string(attr_name));
}

// Checks if current state of new node is a valid node.
template <typename GraphViewT>
inline bool IsWellFormed(
    NewNode<GraphViewT>* new_node,
    const absl::flat_hash_map<absl::string_view, int>& updated_node_names) {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_68(mht_68_v, 1278, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "IsWellFormed");

  ResizeByTrimmingEndForValue(&new_node->regular_fanins, EmptyTensorId());
  int new_node_regular_fanins_size = new_node->regular_fanins.size();
  if (new_node_regular_fanins_size != new_node->num_regular_fanins) {
    return false;
  }

  const string& node_name = new_node->node.name();
  auto invalid_node_name = [new_node, updated_node_names,
                            node_name](absl::string_view fanin_node_name) {
   std::vector<std::string> mht_69_v;
   mht_69_v.push_back("fanin_node_name: \"" + std::string(fanin_node_name.data(), fanin_node_name.size()) + "\"");
   MHTracer_DTPStensorflowPScorePSgrapplerPSutilsPSgraph_view_internalDTh mht_69(mht_69_v, 1291, "", "./tensorflow/core/grappler/utils/graph_view_internal.h", "lambda");

    return fanin_node_name == node_name ||
           !CheckNodeNameExists(fanin_node_name, updated_node_names,
                                new_node->graph_view);
  };
  // Check if nodes of all fanins exist (from name) and if fanins do not
  // introduce self loops.
  for (const auto& regular : new_node->regular_fanins) {
    if (invalid_node_name(regular.node())) {
      return false;
    }
  }
  for (const auto& control : new_node->controlling_fanins) {
    if (invalid_node_name(control)) {
      return false;
    }
  }

  return true;
}

}  // namespace internal
}  // namespace utils
}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_UTILS_GRAPH_VIEW_INTERNAL_H_
