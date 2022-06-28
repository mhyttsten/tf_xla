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

// A Graph describes a set of computations that are to be
// performed, as well as the dependencies between those
// computations. The basic model is a DAG (directed acyclic graph) with
// * internal nodes representing computational operations to be performed;
// * edges represent dependencies, indicating the target may only be
//   executed once the source has completed; and
// * predefined "source" (start) and "sink" (finish) nodes -- the source
//   should be the only node that doesn't depend on anything, and the sink
//   should be the only node that nothing depends on.
//
// Note: Node ids are intended to be relatively dense in the
// 0..max_id range, but there may be gaps since ids won't be reused.
//
// Note: Some dependencies between operations are due to one operation
// consuming the output of another. In fact operations can produce
// multiple outputs and consume multiple inputs, and some
// optimizations will care about which specific outputs are connected
// to which specific inputs.  We therefore represent data dependency
// between output O of layer A and input I of layer B using
// "input index" and "output index" labels per edge.

#ifndef TENSORFLOW_CORE_GRAPH_GRAPH_H_
#define TENSORFLOW_CORE_GRAPH_GRAPH_H_
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
class MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh {
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
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh() {
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


#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/edgeset.h"
#include "tensorflow/core/lib/core/arena.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/iterator_range.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class Edge;
class EdgeSetTest;
class Graph;
class GraphDef;
class Node;
struct OutputTensor;
class VersionDef;
class WhileContext;

class NeighborIter;     // Declared below
class NodeIter;         // Declared below

// Indicates where the graph instance is originated from.
enum class ConstructionContext {
  kNotTracked,     // Not tracked.
  kDirectSession,  // From `tensorflow::DirectSession`, TF1 session API.
  kEagerRuntime,   // Registered from TF2 eager runtime.
};

class Node {
 public:
  std::string DebugString() const;
  int id() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_0(mht_0_v, 255, "", "./tensorflow/core/graph/graph.h", "id");
 return id_; }
  int cost_id() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_1(mht_1_v, 259, "", "./tensorflow/core/graph/graph.h", "cost_id");
 return cost_id_; }
  const std::string& name() const;
  void set_name(std::string name);
  const std::string& type_string() const;

  // def() provides the NodeDef the user supplied, but the specifics
  // of this Node may have changed due to placement, optimization, etc.
  // In particular:
  // * def().name() will match name();
  // * def().op() will match type_string() and op_def().name();
  // * def().input() is not reliable, use "in_edges()" below instead;
  // * def().device() is the "user's requested device" and may not match
  //   the actual assigned device, see assigned_device_name() below;
  // * def().attr() is authoritative.
  // TODO(irving): Replace with NodeInfo.
  const NodeDef& def() const;
  const OpDef& op_def() const;

  // TODO(mdan): This is only used by control_flow_deps_o_chains. Remove?
  NodeDef* mutable_def();

  // input and output types
  int32 num_inputs() const;
  DataType input_type(int32_t i) const;
  const DataTypeVector& input_types() const;

  int32 num_outputs() const;
  DataType output_type(int32_t o) const;
  const DataTypeVector& output_types() const;

  // The device requested by the user.  For the actual assigned device,
  // use assigned_device_name() below.
  const std::string& requested_device() const;

  // This changes the user requested device but not necessarily the device that
  // on which the operation will run.
  void set_requested_device(const std::string& device);

  // This gives the device the runtime has assigned this node to.  If
  // you want the device the user requested, use def().device() instead.
  // TODO(josh11b): Validate that the assigned_device, if not empty:
  // fully specifies a device, and satisfies def().device().
  // TODO(josh11b): Move assigned_device_name outside of Node into a
  // NodeId->DeviceName map.
  const std::string& assigned_device_name() const;
  void set_assigned_device_name(const std::string& device_name);
  bool has_assigned_device_name() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_2(mht_2_v, 308, "", "./tensorflow/core/graph/graph.h", "has_assigned_device_name");

    return assigned_device_name_index_ > 0;
  }
  int assigned_device_name_index() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_3(mht_3_v, 314, "", "./tensorflow/core/graph/graph.h", "assigned_device_name_index");
 return assigned_device_name_index_; }
  void set_assigned_device_name_index(int index);

  // Sets 'original_node_names' field of this node's DebugInfo proto to
  // 'names'.
  void set_original_node_names(const std::vector<string>& names);
  void set_original_func_names(const std::vector<string>& names);

  // Read only access to attributes
  AttrSlice attrs() const;

  // Inputs requested by the NodeDef.  For the actual inputs, use in_edges.
  const protobuf::RepeatedPtrField<string>& requested_inputs() const;

  // Get the neighboring nodes via edges either in or out of this node.  This
  // includes control edges.
  gtl::iterator_range<NeighborIter> in_nodes() const;
  gtl::iterator_range<NeighborIter> out_nodes() const;
  const EdgeSet& in_edges() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_4(mht_4_v, 335, "", "./tensorflow/core/graph/graph.h", "in_edges");
 return in_edges_; }
  const EdgeSet& out_edges() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_5(mht_5_v, 339, "", "./tensorflow/core/graph/graph.h", "out_edges");
 return out_edges_; }

  // Node type helpers.
  bool IsSource() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_6(mht_6_v, 345, "", "./tensorflow/core/graph/graph.h", "IsSource");
 return id() == 0; }
  bool IsSink() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_7(mht_7_v, 349, "", "./tensorflow/core/graph/graph.h", "IsSink");
 return id() == 1; }
  // Anything other than the special Source & Sink nodes.
  bool IsOp() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_8(mht_8_v, 354, "", "./tensorflow/core/graph/graph.h", "IsOp");
 return id() > 1; }

  // Node class helpers
  bool IsSwitch() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_9(mht_9_v, 360, "", "./tensorflow/core/graph/graph.h", "IsSwitch");
 return class_ == NC_SWITCH; }
  bool IsMerge() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_10(mht_10_v, 364, "", "./tensorflow/core/graph/graph.h", "IsMerge");
 return class_ == NC_MERGE; }
  bool IsEnter() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_11(mht_11_v, 368, "", "./tensorflow/core/graph/graph.h", "IsEnter");
 return class_ == NC_ENTER; }
  bool IsExit() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_12(mht_12_v, 372, "", "./tensorflow/core/graph/graph.h", "IsExit");
 return class_ == NC_EXIT; }
  bool IsNextIteration() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_13(mht_13_v, 376, "", "./tensorflow/core/graph/graph.h", "IsNextIteration");
 return class_ == NC_NEXT_ITERATION; }
  bool IsLoopCond() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_14(mht_14_v, 380, "", "./tensorflow/core/graph/graph.h", "IsLoopCond");
 return class_ == NC_LOOP_COND; }
  bool IsControlTrigger() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_15(mht_15_v, 384, "", "./tensorflow/core/graph/graph.h", "IsControlTrigger");
 return class_ == NC_CONTROL_TRIGGER; }
  bool IsSend() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_16(mht_16_v, 388, "", "./tensorflow/core/graph/graph.h", "IsSend");
 return class_ == NC_SEND || class_ == NC_HOST_SEND; }
  bool IsRecv() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_17(mht_17_v, 392, "", "./tensorflow/core/graph/graph.h", "IsRecv");
 return class_ == NC_RECV || class_ == NC_HOST_RECV; }
  bool IsConstant() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_18(mht_18_v, 396, "", "./tensorflow/core/graph/graph.h", "IsConstant");
 return class_ == NC_CONSTANT; }
  bool IsVariable() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_19(mht_19_v, 400, "", "./tensorflow/core/graph/graph.h", "IsVariable");
 return class_ == NC_VARIABLE; }
  bool IsIdentity() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_20(mht_20_v, 404, "", "./tensorflow/core/graph/graph.h", "IsIdentity");
 return class_ == NC_IDENTITY; }
  bool IsGetSessionHandle() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_21(mht_21_v, 408, "", "./tensorflow/core/graph/graph.h", "IsGetSessionHandle");
 return class_ == NC_GET_SESSION_HANDLE; }
  bool IsGetSessionTensor() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_22(mht_22_v, 412, "", "./tensorflow/core/graph/graph.h", "IsGetSessionTensor");
 return class_ == NC_GET_SESSION_TENSOR; }
  bool IsDeleteSessionTensor() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_23(mht_23_v, 416, "", "./tensorflow/core/graph/graph.h", "IsDeleteSessionTensor");

    return class_ == NC_DELETE_SESSION_TENSOR;
  }
  bool IsControlFlow() const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_24(mht_24_v, 422, "", "./tensorflow/core/graph/graph.h", "IsControlFlow");

    return (class_ != NC_OTHER) &&  // Fast path
           (IsSwitch() || IsMerge() || IsEnter() || IsExit() ||
            IsNextIteration());
  }
  bool IsHostSend() const {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_25(mht_25_v, 430, "", "./tensorflow/core/graph/graph.h", "IsHostSend");
 return class_ == NC_HOST_SEND; }
  bool IsHostRecv() const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_26(mht_26_v, 434, "", "./tensorflow/core/graph/graph.h", "IsHostRecv");
 return class_ == NC_HOST_RECV; }
  bool IsScopedAllocator() const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_27(mht_27_v, 438, "", "./tensorflow/core/graph/graph.h", "IsScopedAllocator");
 return class_ == NC_SCOPED_ALLOCATOR; }
  bool IsCollective() const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_28(mht_28_v, 442, "", "./tensorflow/core/graph/graph.h", "IsCollective");
 return class_ == NC_COLLECTIVE; }

  bool IsMetadata() const {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_29(mht_29_v, 447, "", "./tensorflow/core/graph/graph.h", "IsMetadata");
 return class_ == NC_METADATA; }
  bool IsFakeParam() const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_30(mht_30_v, 451, "", "./tensorflow/core/graph/graph.h", "IsFakeParam");
 return class_ == NC_FAKE_PARAM; }
  bool IsPartitionedCall() const {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_31(mht_31_v, 455, "", "./tensorflow/core/graph/graph.h", "IsPartitionedCall");
 return class_ == NC_PARTITIONED_CALL; }

  // Returns true if this node is any kind of function call node.
  //
  // NOTE: "function call nodes" include partitioned call ops, symbolic gradient
  // ops, and ops whose type_string is the name of a function ("function ops").
  bool IsFunctionCall() const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_32(mht_32_v, 464, "", "./tensorflow/core/graph/graph.h", "IsFunctionCall");

    return class_ == NC_PARTITIONED_CALL || class_ == NC_FUNCTION_OP ||
           class_ == NC_SYMBOLIC_GRADIENT;
  }

  bool IsIfNode() const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_33(mht_33_v, 472, "", "./tensorflow/core/graph/graph.h", "IsIfNode");
 return class_ == NC_IF; }
  bool IsWhileNode() const {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_34(mht_34_v, 476, "", "./tensorflow/core/graph/graph.h", "IsWhileNode");
 return class_ == NC_WHILE; }
  bool IsCaseNode() const {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_35(mht_35_v, 480, "", "./tensorflow/core/graph/graph.h", "IsCaseNode");
 return class_ == NC_CASE; }
  // Is this node a function input
  bool IsArg() const {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_36(mht_36_v, 485, "", "./tensorflow/core/graph/graph.h", "IsArg");
 return class_ == NC_ARG; }
  // Is this node a function output
  bool IsRetval() const {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_37(mht_37_v, 490, "", "./tensorflow/core/graph/graph.h", "IsRetval");
 return class_ == NC_RETVAL; }

  bool IsDistributedCommunication() const {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_38(mht_38_v, 495, "", "./tensorflow/core/graph/graph.h", "IsDistributedCommunication");

    return op_def().is_distributed_communication();
  }

  template <typename T>
  void AddAttr(const std::string& name, const T& val) {
   std::vector<std::string> mht_39_v;
   mht_39_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_39(mht_39_v, 504, "", "./tensorflow/core/graph/graph.h", "AddAttr");

    SetAttrValue(val, AddAttrHelper(name));
    UpdateProperties();
  }

  void AddAttr(const std::string& name, std::vector<string>&& val) {
   std::vector<std::string> mht_40_v;
   mht_40_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_40(mht_40_v, 513, "", "./tensorflow/core/graph/graph.h", "AddAttr");

    MoveAttrValue(std::move(val), AddAttrHelper(name));
    UpdateProperties();
  }

  void ClearAttr(const std::string& name);

  // Returns into '*e' the edge connecting to the 'idx' input of this Node.
  Status input_edge(int idx, const Edge** e) const;

  // Returns into '*edges' the input data edges of this Node, indexed by input
  // number. Does not return control edges.
  Status input_edges(std::vector<const Edge*>* edges) const;

  // Returns into '*n' the node that has an output connected to the
  // 'idx' input of this Node.
  Status input_node(int idx, const Node** n) const;
  Status input_node(int idx, Node** n) const;

  // Returns into '*t' the idx-th input tensor of this node, represented as the
  // output tensor of input_node(idx).
  Status input_tensor(int idx, OutputTensor* t) const;

  WhileContext* while_ctx() const {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_41(mht_41_v, 539, "", "./tensorflow/core/graph/graph.h", "while_ctx");
 return while_ctx_; }
  void set_while_ctx(WhileContext* while_ctx) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_42(mht_42_v, 543, "", "./tensorflow/core/graph/graph.h", "set_while_ctx");

    DCHECK(IsExit());
    DCHECK(while_ctx_ == nullptr);
    while_ctx_ = while_ctx;
  }

  std::shared_ptr<NodeProperties> properties() const { return props_; }

  // Sets the stack trace for the node. Assumes that getting and setting the
  // stack trace for a given node will not race.
  void SetStackTrace(const std::shared_ptr<AbstractStackTrace>& stack_trace) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_43(mht_43_v, 556, "", "./tensorflow/core/graph/graph.h", "SetStackTrace");

    stack_trace_ = stack_trace;
  }

  // Get the stack trace for when the node was instantiated.
  const std::shared_ptr<AbstractStackTrace>& GetStackTrace() const {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_44(mht_44_v, 564, "", "./tensorflow/core/graph/graph.h", "GetStackTrace");

    return stack_trace_;
  }

  // Called after an attr has changed. Decides whether we need to update some
  // property of the node (stored in props_).
  void UpdateProperties();

  // Erases type information from the node.
  void ClearTypeInfo();

  // Called after an incident non-control edge has changed. Does nothing if not
  // all input edges are defined.
  void RunForwardTypeInference();

 private:
  // TODO(mdan): Drop this.
  friend class Graph;
  Node();

  // Stack trace for the user code for node instantiation. Can be shared across
  // multiple nodes (e.g. when inlining).
  std::shared_ptr<AbstractStackTrace> stack_trace_;

  // Releases memory from props_, in addition to restoring *this to its
  // uninitialized state.
  void Clear();

  // Make a copy of the Node's props_ if props_ is shared with
  // other nodes. This must be called before mutating properties,
  // e.g. in AddAttr.
  void MaybeCopyOnWrite();

  AttrValue* AddAttrHelper(const std::string& name);

  // A set of mutually exclusive classes for different kinds of nodes,
  // class_ is initialized in the Node::Initialize routine based on the
  // node's type_string().
  enum NodeClass {
    NC_UNINITIALIZED,
    NC_SWITCH,
    NC_MERGE,
    NC_ENTER,
    NC_EXIT,
    NC_NEXT_ITERATION,
    NC_LOOP_COND,
    NC_CONTROL_TRIGGER,
    NC_SEND,
    NC_HOST_SEND,
    NC_RECV,
    NC_HOST_RECV,
    NC_CONSTANT,
    NC_VARIABLE,
    NC_IDENTITY,
    NC_GET_SESSION_HANDLE,
    NC_GET_SESSION_TENSOR,
    NC_DELETE_SESSION_TENSOR,
    NC_METADATA,
    NC_SCOPED_ALLOCATOR,
    NC_COLLECTIVE,
    NC_FAKE_PARAM,
    NC_PARTITIONED_CALL,
    NC_FUNCTION_OP,
    NC_SYMBOLIC_GRADIENT,
    NC_IF,
    NC_WHILE,
    NC_CASE,
    NC_ARG,
    NC_RETVAL,
    NC_OTHER  // Not a special kind of node
  };

  void Initialize(int id, int cost_id, std::shared_ptr<NodeProperties> props,
                  NodeClass node_class);

  static NodeClass GetNodeClassForOp(const std::string& ts);

  int id_;       // -1 until Initialize() is called
  int cost_id_;  // -1 if there is no corresponding cost accounting node
  NodeClass class_;

  EdgeSet in_edges_;
  EdgeSet out_edges_;

  // NOTE(skyewm): inheriting from core::RefCounted may have a slight
  // performance benefit over using shared_ptr, at the cost of manual ref
  // counting
  std::shared_ptr<NodeProperties> props_;

  // Index within Graph::device_names_ of the name of device assigned
  // to perform this computation.
  int assigned_device_name_index_;

  // A back-pointer to the Graph that owns this node.  Currently, this exists
  // solely to allow Node::[set_]assigned_device_name() to work. However, if all
  // callers of Node::[set_]assigned_device_name() are modified to use the
  // equivalent methods defined directly on Graph, then we can remove this
  // field and reclaim that memory.
  Graph* graph_;

  // Set if this is an exit node of a while loop with an associated
  // WhileContext. Otherwise null. (This is only set for exit nodes because
  // they're the first nodes of a loop encountered while creating the gradient
  // graph. Exit nodes that are part of while loop gradient graphs will not have
  // this set.)
  WhileContext* while_ctx_;

  TF_DISALLOW_COPY_AND_ASSIGN(Node);
};

// Stores debug information associated with the Node.
struct NodeDebugInfo {
  const std::string name;
  std::vector<string> original_node_names;
  std::vector<string> original_func_names;

  NodeDebugInfo(const Node& n);
  NodeDebugInfo(const NodeDef& ndef);
  NodeDebugInfo(StringPiece node_name, bool has_experimental_debug_info,
                const NodeDef_ExperimentalDebugInfo& experimental_debug_info);
};

// Represents an input of a node, i.e., the `index`-th input to `node`.
struct InputTensor {
  Node* node;
  int index;

  InputTensor(Node* n, int i) : node(n), index(i) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_45(mht_45_v, 694, "", "./tensorflow/core/graph/graph.h", "InputTensor");
}
  InputTensor() : node(nullptr), index(0) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_46(mht_46_v, 698, "", "./tensorflow/core/graph/graph.h", "InputTensor");
}

  // Returns true if this InputTensor is identical to 'other'. Nodes are
  // compared using pointer equality.
  bool operator==(const InputTensor& other) const;

  // A hash function for InputTensors. Nodes are hashed based on their pointer
  // value.
  struct Hash {
    uint64 operator()(InputTensor const& s) const;
  };
};

// Represents an output of a node, i.e., the `index`-th output of `node`. Note
// that a single `OutputTensor` can correspond to multiple `Edge`s if the output
// is consumed by multiple destination nodes.
struct OutputTensor {
  Node* node;
  int index;

  OutputTensor(Node* n, int i) : node(n), index(i) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_47(mht_47_v, 721, "", "./tensorflow/core/graph/graph.h", "OutputTensor");
}
  OutputTensor() : node(nullptr), index(0) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_48(mht_48_v, 725, "", "./tensorflow/core/graph/graph.h", "OutputTensor");
}

  // Returns true if this OutputTensor is identical to 'other'. Nodes are
  // compared using pointer equality.
  bool operator==(const OutputTensor& other) const;

  // A hash function for OutputTensors. Nodes are hashed based on their pointer
  // value.
  struct Hash {
    uint64 operator()(OutputTensor const& s) const;
  };
};

class Edge {
 public:
  Node* src() const {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_49(mht_49_v, 743, "", "./tensorflow/core/graph/graph.h", "src");
 return src_; }
  Node* dst() const {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_50(mht_50_v, 747, "", "./tensorflow/core/graph/graph.h", "dst");
 return dst_; }
  int id() const {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_51(mht_51_v, 751, "", "./tensorflow/core/graph/graph.h", "id");
 return id_; }

  // Return the index of the source output that produces the data
  // carried by this edge.  The special value kControlSlot is used
  // for control dependencies.
  int src_output() const {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_52(mht_52_v, 759, "", "./tensorflow/core/graph/graph.h", "src_output");
 return src_output_; }

  // Return the index of the destination input that consumes the data
  // carried by this edge.  The special value kControlSlot is used
  // for control dependencies.
  int dst_input() const {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_53(mht_53_v, 767, "", "./tensorflow/core/graph/graph.h", "dst_input");
 return dst_input_; }

  // Return true iff this is an edge that indicates a control-flow
  // (as opposed to a data-flow) dependency.
  bool IsControlEdge() const;

  std::string DebugString() const;

 private:
  Edge() {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_54(mht_54_v, 779, "", "./tensorflow/core/graph/graph.h", "Edge");
}

  friend class EdgeSetTest;
  friend class Graph;
  Node* src_;
  Node* dst_;
  int id_;
  int src_output_;
  int dst_input_;
};

// Allows for iteration of the edges of a Graph, by iterating the underlying
// Graph.edges_ vector while skipping over null entries.
class GraphEdgesIterable {
 private:
  const std::vector<Edge*>& edges_;

 public:
  explicit GraphEdgesIterable(const std::vector<Edge*>& edges)
      : edges_(edges) {
   std::vector<std::string> mht_55_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_55(mht_55_v, 801, "", "./tensorflow/core/graph/graph.h", "GraphEdgesIterable");
}

  typedef Edge* value_type;

  class const_iterator {
   private:
    // The underlying iterator.
    std::vector<value_type>::const_iterator iter_;

    // The end of the underlying iterator.
    std::vector<value_type>::const_iterator end_;

    // Advances iter_ until it reaches a non-null item, or reaches the end.
    void apply_filter() {
   std::vector<std::string> mht_56_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_56(mht_56_v, 817, "", "./tensorflow/core/graph/graph.h", "apply_filter");

      while (iter_ != end_ && *iter_ == nullptr) {
        ++iter_;
      }
    }

   public:
    const_iterator(std::vector<value_type>::const_iterator iter,
                   std::vector<value_type>::const_iterator end)
        : iter_(iter), end_(end) {
   std::vector<std::string> mht_57_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_57(mht_57_v, 829, "", "./tensorflow/core/graph/graph.h", "const_iterator");

      apply_filter();
    }

    bool operator==(const const_iterator& other) const {
      return iter_ == other.iter_;
    }

    bool operator!=(const const_iterator& other) const {
      return iter_ != other.iter_;
    }

    // This is the prefix increment operator (++x), which is the operator
    // used by C++ range iteration (for (x : y) ...).  We intentionally do not
    // provide a postfix increment operator.
    const_iterator& operator++() {
      ++iter_;
      apply_filter();
      return *this;
    }

    value_type operator*() {
   std::vector<std::string> mht_58_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_58(mht_58_v, 853, "", "./tensorflow/core/graph/graph.h", "*");
 return *iter_; }
  };

  const_iterator begin() {
   std::vector<std::string> mht_59_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_59(mht_59_v, 859, "", "./tensorflow/core/graph/graph.h", "begin");

    return const_iterator(edges_.begin(), edges_.end());
  }
  const_iterator end() {
   std::vector<std::string> mht_60_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_60(mht_60_v, 865, "", "./tensorflow/core/graph/graph.h", "end");
 return const_iterator(edges_.end(), edges_.end()); }
};

// Thread compatible but not thread safe.
class Graph {
 public:
  // Constructs a graph with a single SOURCE (always id kSourceId) and a
  // single SINK (always id kSinkId) node, and an edge from SOURCE->SINK.
  //
  // The graph can hold ops found in the registry. `ops`s lifetime must be at
  // least that of the constructed graph's.
  explicit Graph(const OpRegistryInterface* ops);

  // Constructs a graph with a single SOURCE (always id kSourceId) and a
  // single SINK (always id kSinkId) node, and an edge from SOURCE->SINK.
  //
  // The graph can hold ops found in `flib_def`. Unlike the constructor taking
  // an OpRegistryInterface, this constructor copies the function definitions in
  // `flib_def` so its lifetime may be shorter than that of the graph's. The
  // OpRegistryInterface backing `flib_def` must still have the lifetime of the
  // graph though.
  explicit Graph(const FunctionLibraryDefinition& flib_def);

  ~Graph();

  // Clone the current graph into a new one.
  std::unique_ptr<Graph> Clone();

  static const int kControlSlot;

  // The GraphDef version range of this graph (see graph.proto).
  const VersionDef& versions() const;
  void set_versions(const VersionDef& versions);

  // Adds a new node to this graph, and returns it. Infers the Op and
  // input/output types for the node. *this owns the returned instance.
  // Returns nullptr and sets *status on error.
  Node* AddNode(NodeDef node_def, Status* status);

  // Same as above, but using StatusOr. This method is always preferred.
  StatusOr<Node*> AddNode(NodeDef node_def);

  // Copies *node, which may belong to another graph, to a new node,
  // which is returned.  Does not copy any edges.  *this owns the
  // returned instance.
  Node* CopyNode(const Node* node);

  // Removes a node from this graph, including all edges from or to it.
  // *node should not be accessed after calling this function.
  // REQUIRES: node->IsOp()
  void RemoveNode(Node* node);

  void Copy(const Graph& src);

  // Removes all nodes from this graph, including all edges from or to them.
  // No Node* references to the Graph are valid post.
  void Clear();

  // Adds an edge that connects the xth output of `source` to the yth input of
  // `dest` and returns it. Does not update dest's NodeDef.
  const Edge* AddEdge(Node* source, int x, Node* dest, int y);

  // Adds a control edge (no data flows along this edge) that connects `source`
  // to `dest`. If `dest`s NodeDef is missing the corresponding control input,
  // adds the control input.
  //
  // If such a control edge already exists and `allow_duplicates` is false, no
  // edge is added and the function returns nullptr. Otherwise the edge is
  // unconditionally created and returned. The NodeDef is not updated if
  // `allow_duplicates` is true.
  // TODO(skyewm): // TODO(skyewm): allow_duplicates is needed only by
  // graph_partition.cc. Figure out if we can do away with it.
  const Edge* AddControlEdge(Node* source, Node* dest,
                             bool allow_duplicates = false);

  // Removes edge from the graph. Does not update the destination node's
  // NodeDef.
  // REQUIRES: The edge must exist.
  void RemoveEdge(const Edge* edge);

  // Removes control edge `edge` from the graph. Note that this also updates
  // the corresponding NodeDef to reflect the change.
  // REQUIRES: The control edge must exist.
  void RemoveControlEdge(const Edge* e);

  // Updates the input to a node.  The existing edge to `dst` is removed and an
  // edge from `new_src` to `dst` is created. The NodeDef associated with `dst`
  // is also updated.
  Status UpdateEdge(Node* new_src, int new_src_index, Node* dst, int dst_index);

  // Like AddEdge but updates dst's NodeDef. Used to add an input edge to a
  // "While" op during gradient construction, see AddInputWhileHack in
  // python_api.h for more details.
  Status AddWhileInputHack(Node* new_src, int new_src_index, Node* dst);

  // Adds the function and gradient definitions in `fdef_lib` to this graph's op
  // registry. Ignores duplicate functions, and returns a bad status if an
  // imported function differs from an existing function or op with the same
  // name.
  Status AddFunctionLibrary(const FunctionDefLibrary& fdef_lib);

  // The number of live nodes in the graph.
  //
  // Because nodes can be removed from the graph, num_nodes() is often
  // smaller than num_node_ids(). If one needs to create an array of
  // nodes indexed by node ids, num_node_ids() should be used as the
  // array's size.
  int num_nodes() const {
   std::vector<std::string> mht_61_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_61(mht_61_v, 975, "", "./tensorflow/core/graph/graph.h", "num_nodes");
 return num_nodes_; }

  // The number of live nodes in the graph, excluding the Source and Sink nodes.
  int num_op_nodes() const {
   std::vector<std::string> mht_62_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_62(mht_62_v, 981, "", "./tensorflow/core/graph/graph.h", "num_op_nodes");

    DCHECK_GE(num_nodes_, 2);
    return num_nodes_ - 2;
  }

  // The number of live edges in the graph.
  //
  // Because edges can be removed from the graph, num_edges() is often
  // smaller than num_edge_ids(). If one needs to create an array of
  // edges indexed by edge ids, num_edge_ids() should be used as the
  // array's size.
  int num_edges() const {
   std::vector<std::string> mht_63_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_63(mht_63_v, 995, "", "./tensorflow/core/graph/graph.h", "num_edges");
 return num_edges_; }

  // Serialize the nodes starting at `from_node_id` to a GraphDef.
  void ToGraphDefSubRange(GraphDef* graph_def, int from_node_id) const;

  // Serialize to a GraphDef.
  void ToGraphDef(GraphDef* graph_def) const;

  // This version can be called from debugger to inspect the graph content.
  // Use the previous version outside debug context for efficiency reasons.
  //
  // Note: We do not expose a DebugString() API, since GraphDef.DebugString() is
  // not defined in some TensorFlow builds.
  GraphDef ToGraphDefDebug() const;

  // Generate new node name with the specified prefix that is unique
  // across this graph.
  std::string NewName(StringPiece prefix);

  // Access to the list of all nodes.  Example usage:
  //   for (Node* node : graph.nodes()) { ... }
  gtl::iterator_range<NodeIter> nodes() const;

  // Access to the list of all nodes, excluding the Source and Sink nodes.
  gtl::iterator_range<NodeIter> op_nodes() const;

  // Returns one more than the maximum id assigned to any node.
  int num_node_ids() const {
   std::vector<std::string> mht_64_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_64(mht_64_v, 1025, "", "./tensorflow/core/graph/graph.h", "num_node_ids");
 return nodes_.size(); }

  // Returns the node associated with an id, or nullptr if no node
  // with that id (the node with that id was removed and the id has
  // not yet been re-used). *this owns the returned instance.
  // REQUIRES: 0 <= id < num_node_ids().
  Node* FindNodeId(int id) const {
   std::vector<std::string> mht_65_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_65(mht_65_v, 1034, "", "./tensorflow/core/graph/graph.h", "FindNodeId");
 return nodes_[id]; }

  // Returns one more than the maximum id assigned to any edge.
  int num_edge_ids() const {
   std::vector<std::string> mht_66_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_66(mht_66_v, 1040, "", "./tensorflow/core/graph/graph.h", "num_edge_ids");
 return edges_.size(); }

  // Returns the Edge associated with an id, or nullptr if no edge
  // with that id (the edge with that id was removed and the id has
  // not yet been re-used). *this owns the returned instance.
  // REQUIRES: 0 <= id < num_edge_ids().
  const Edge* FindEdgeId(int id) const {
   std::vector<std::string> mht_67_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_67(mht_67_v, 1049, "", "./tensorflow/core/graph/graph.h", "FindEdgeId");
 return edges_[id]; }

  // Access to the set of all edges.  Example usage:
  //   for (const Edge* e : graph.edges()) { ... }
  GraphEdgesIterable edges() const {
   std::vector<std::string> mht_68_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_68(mht_68_v, 1056, "", "./tensorflow/core/graph/graph.h", "edges");
 return GraphEdgesIterable(edges_); }

  // The pre-defined nodes.
  enum { kSourceId = 0, kSinkId = 1 };
  Node* source_node() const {
   std::vector<std::string> mht_69_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_69(mht_69_v, 1063, "", "./tensorflow/core/graph/graph.h", "source_node");
 return FindNodeId(kSourceId); }
  Node* sink_node() const {
   std::vector<std::string> mht_70_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_70(mht_70_v, 1067, "", "./tensorflow/core/graph/graph.h", "sink_node");
 return FindNodeId(kSinkId); }

  const OpRegistryInterface* op_registry() const {
   std::vector<std::string> mht_71_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_71(mht_71_v, 1072, "", "./tensorflow/core/graph/graph.h", "op_registry");
 return &ops_; }
  const FunctionLibraryDefinition& flib_def() const {
   std::vector<std::string> mht_72_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_72(mht_72_v, 1076, "", "./tensorflow/core/graph/graph.h", "flib_def");
 return ops_; }

  // TODO(mdan): This is only used by control_flow_deps_o_chains. Remove?
  FunctionLibraryDefinition* mutable_flib_def() {
   std::vector<std::string> mht_73_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_73(mht_73_v, 1082, "", "./tensorflow/core/graph/graph.h", "mutable_flib_def");
 return &ops_; }

  void CheckDeviceNameIndex(int index) {
   std::vector<std::string> mht_74_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_74(mht_74_v, 1087, "", "./tensorflow/core/graph/graph.h", "CheckDeviceNameIndex");

    DCHECK_GE(index, 0);
    DCHECK_LT(index, static_cast<int>(device_names_.size()));
  }

  int InternDeviceName(const std::string& device_name);

  const std::string& get_assigned_device_name(const Node& node) const {
   std::vector<std::string> mht_75_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_75(mht_75_v, 1097, "", "./tensorflow/core/graph/graph.h", "get_assigned_device_name");

    return device_names_[node.assigned_device_name_index()];
  }

  void set_assigned_device_name_index(Node* node, int device_name_index) {
   std::vector<std::string> mht_76_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_76(mht_76_v, 1104, "", "./tensorflow/core/graph/graph.h", "set_assigned_device_name_index");

    CheckDeviceNameIndex(device_name_index);
    node->assigned_device_name_index_ = device_name_index;
  }

  void set_assigned_device_name(Node* node, const std::string& device_name) {
   std::vector<std::string> mht_77_v;
   mht_77_v.push_back("device_name: \"" + device_name + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_77(mht_77_v, 1113, "", "./tensorflow/core/graph/graph.h", "set_assigned_device_name");

    node->assigned_device_name_index_ = InternDeviceName(device_name);
  }

  // Returns OK if `node` is non-null and belongs to this graph
  Status IsValidNode(const Node* node) const;

  // Returns OK if IsValidNode(`node`) and `idx` is a valid output.  Does not
  // accept control outputs.
  Status IsValidOutputTensor(const Node* node, int idx) const;

  // Returns OK if IsValidNode(`node`) and `idx` a valid input.  Does not accept
  // control inputs.
  Status IsValidInputTensor(const Node* node, int idx) const;

  // Create and return a new WhileContext owned by this graph. This is called
  // when a new while loop is created. `frame_name` must be unique among
  // WhileContexts in this graph.
  Status AddWhileContext(StringPiece frame_name, std::vector<Node*> enter_nodes,
                         std::vector<Node*> exit_nodes,
                         OutputTensor cond_output,
                         std::vector<OutputTensor> body_inputs,
                         std::vector<OutputTensor> body_outputs,
                         WhileContext** result);

  // Builds a node name to node pointer index for all nodes in the graph.
  std::unordered_map<string, Node*> BuildNodeNameIndex() const;

  absl::optional<std::vector<bool>>& GetConstArgIndicesCache() const {
   std::vector<std::string> mht_78_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_78(mht_78_v, 1144, "", "./tensorflow/core/graph/graph.h", "GetConstArgIndicesCache");

    return const_arg_indices_cache_;
  }

  // TODO(kkb): Add to the constructor when it becomes managable.
  // Sets the graph construction context.
  void SetConstructionContext(ConstructionContext construction_context) {
   std::vector<std::string> mht_79_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_79(mht_79_v, 1153, "", "./tensorflow/core/graph/graph.h", "SetConstructionContext");

    construction_context_ = construction_context;
  }

  // TODO(kkb): Rename to `GetConstructionContext` once we're comfortable
  // making this stable and make it available widely.
  // Returns the graph construction context. It's `kUnknown` if not set.
  ConstructionContext GetConstructionContextInternal() const {
   std::vector<std::string> mht_80_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_80(mht_80_v, 1163, "", "./tensorflow/core/graph/graph.h", "GetConstructionContextInternal");

    return construction_context_;
  }

  // TODO(josh11b): uint64 hash() const;

 private:
  // If cost_node is non-null, then cost accounting (in CostModel)
  // will be associated with that node rather than the new one being
  // created.
  //
  // Ownership of the returned Node is not transferred to caller.
  Node* AllocateNode(std::shared_ptr<NodeProperties> props,
                     const Node* cost_node, Node::NodeClass node_class);
  void ReleaseNode(Node* node);
  // Insert edge in free_edges_ for possible reuse.
  void RecycleEdge(const Edge* edge);
  // Registry of all known ops, including functions.
  FunctionLibraryDefinition ops_;

  // GraphDef versions
  const std::unique_ptr<VersionDef> versions_;

  // Allocator which will give us good locality.
  core::Arena arena_;

  // Map from node ids to allocated nodes.  nodes_[id] may be nullptr if
  // the node with that id was removed from the graph.
  std::vector<Node*> nodes_;

  // Number of nodes alive.
  int64_t num_nodes_ = 0;

  // Map from edge ids to allocated edges.  edges_[id] may be nullptr if
  // the edge with that id was removed from the graph.
  std::vector<Edge*> edges_;

  // The number of entries in edges_ that are not nullptr.
  int num_edges_ = 0;

  // Allocated but free nodes and edges.
  std::vector<Node*> free_nodes_;
  std::vector<Edge*> free_edges_;

  // For generating unique names.
  int name_counter_ = 0;

  // In most graphs, the number of unique values used for the
  // Node::assigned_device_name() property is quite small.  If the graph is
  // large, then this duplication of values can consume a significant amount of
  // memory.  Instead, we represent the same information using an interning
  // table, which consists of a vector of unique strings (device_names_), as
  // well a map (device_names_map_) from unique strings to indices within the
  // unique string table.
  //
  // The InternDeviceName() method handles adding a new entry into the table,
  // or locating the index of an existing entry.
  //
  // The fact that Node::assigned_device_name() is implemented using an
  // interning table is intentionally public.  This allows algorithms that
  // frequently access this field to do so efficiently, especially for the case
  // where the assigned_device_name of one Node is copied directly from that
  // of another Node.

  // A table of the unique assigned device names.  Indices do NOT correspond
  // to node IDs.  Index 0 is always the empty string.
  std::vector<string> device_names_;

  // Maps unique device names to indices within device_names_[i].
  std::unordered_map<string, int> device_names_map_;

  // All the while contexts owned by this graph, keyed by frame name,
  // corresponding to all the while loops contained in this graph (including
  // nested loops). The stored contexts are usually accessed via
  // AddWhileContext() or Node::while_ctx(), but this manages the lifetime.
  std::map<string, WhileContext> while_ctxs_;

  // Cache of the indices of the arguments which need to be constant for the XLA
  // compilation.
  mutable absl::optional<std::vector<bool>> const_arg_indices_cache_;

  // Indicates the context that this Graph instance is constructed.
  ConstructionContext construction_context_ = ConstructionContext::kNotTracked;

  TF_DISALLOW_COPY_AND_ASSIGN(Graph);
};

// TODO(josh11b): We may want to support keeping an index on various
// node/edge attributes in a graph, particularly node names.

// Helper routines

inline bool IsSource(const Node* node) {
   std::vector<std::string> mht_81_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_81(mht_81_v, 1258, "", "./tensorflow/core/graph/graph.h", "IsSource");
 return node->IsSource(); }
inline bool IsSink(const Node* node) {
   std::vector<std::string> mht_82_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_82(mht_82_v, 1262, "", "./tensorflow/core/graph/graph.h", "IsSink");
 return node->IsSink(); }
inline bool IsSwitch(const Node* node) {
   std::vector<std::string> mht_83_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_83(mht_83_v, 1266, "", "./tensorflow/core/graph/graph.h", "IsSwitch");
 return node->IsSwitch(); }
inline bool IsMerge(const Node* node) {
   std::vector<std::string> mht_84_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_84(mht_84_v, 1270, "", "./tensorflow/core/graph/graph.h", "IsMerge");
 return node->IsMerge(); }
inline bool IsEnter(const Node* node) {
   std::vector<std::string> mht_85_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_85(mht_85_v, 1274, "", "./tensorflow/core/graph/graph.h", "IsEnter");
 return node->IsEnter(); }
inline bool IsExit(const Node* node) {
   std::vector<std::string> mht_86_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_86(mht_86_v, 1278, "", "./tensorflow/core/graph/graph.h", "IsExit");
 return node->IsExit(); }
inline bool IsNextIteration(const Node* n) {
   std::vector<std::string> mht_87_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_87(mht_87_v, 1282, "", "./tensorflow/core/graph/graph.h", "IsNextIteration");
 return n->IsNextIteration(); }
inline bool IsLoopCond(const Node* node) {
   std::vector<std::string> mht_88_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_88(mht_88_v, 1286, "", "./tensorflow/core/graph/graph.h", "IsLoopCond");
 return node->IsLoopCond(); }
inline bool IsControlTrigger(const Node* n) {
   std::vector<std::string> mht_89_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_89(mht_89_v, 1290, "", "./tensorflow/core/graph/graph.h", "IsControlTrigger");
 return n->IsControlTrigger(); }
inline bool IsSend(const Node* node) {
   std::vector<std::string> mht_90_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_90(mht_90_v, 1294, "", "./tensorflow/core/graph/graph.h", "IsSend");
 return node->IsSend(); }
inline bool IsRecv(const Node* node) {
   std::vector<std::string> mht_91_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_91(mht_91_v, 1298, "", "./tensorflow/core/graph/graph.h", "IsRecv");
 return node->IsRecv(); }
inline bool IsHostSend(const Node* node) {
   std::vector<std::string> mht_92_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_92(mht_92_v, 1302, "", "./tensorflow/core/graph/graph.h", "IsHostSend");
 return node->IsHostSend(); }
inline bool IsHostRecv(const Node* node) {
   std::vector<std::string> mht_93_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_93(mht_93_v, 1306, "", "./tensorflow/core/graph/graph.h", "IsHostRecv");
 return node->IsHostRecv(); }

// True for Nodes that mediate the transfer of values between processes.
inline bool IsTransferNode(const Node* n) {
   std::vector<std::string> mht_94_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_94(mht_94_v, 1312, "", "./tensorflow/core/graph/graph.h", "IsTransferNode");
 return IsSend(n) || IsRecv(n); }

inline bool IsConstant(const Node* node) {
   std::vector<std::string> mht_95_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_95(mht_95_v, 1317, "", "./tensorflow/core/graph/graph.h", "IsConstant");
 return node->IsConstant(); }
inline bool IsVariable(const Node* node) {
   std::vector<std::string> mht_96_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_96(mht_96_v, 1321, "", "./tensorflow/core/graph/graph.h", "IsVariable");
 return node->IsVariable(); }
inline bool IsIdentity(const Node* node) {
   std::vector<std::string> mht_97_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_97(mht_97_v, 1325, "", "./tensorflow/core/graph/graph.h", "IsIdentity");
 return node->IsIdentity(); }

// Returns true iff 'n' is a control flow node.
inline bool IsControlFlow(const Node* n) {
   std::vector<std::string> mht_98_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_98(mht_98_v, 1331, "", "./tensorflow/core/graph/graph.h", "IsControlFlow");
 return n->IsControlFlow(); }

// Returns true if the node only depends on its input's metadata
// (shape).  Specifically, returns true for "Size", "Shape" and "Rank" ops.
inline bool IsMetadata(const Node* n) {
   std::vector<std::string> mht_99_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_99(mht_99_v, 1338, "", "./tensorflow/core/graph/graph.h", "IsMetadata");
 return n->IsMetadata(); }

inline bool IsScopedAllocator(const Node* n) {
   std::vector<std::string> mht_100_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_100(mht_100_v, 1343, "", "./tensorflow/core/graph/graph.h", "IsScopedAllocator");
 return n->IsScopedAllocator(); }

inline bool IsHostMemoryPreserving(const Node* node) {
   std::vector<std::string> mht_101_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_101(mht_101_v, 1348, "", "./tensorflow/core/graph/graph.h", "IsHostMemoryPreserving");

  return IsIdentity(node) || IsControlFlow(node);
}

inline bool IsDistributedCommunication(const Node* n) {
   std::vector<std::string> mht_102_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_102(mht_102_v, 1355, "", "./tensorflow/core/graph/graph.h", "IsDistributedCommunication");

  return n->IsDistributedCommunication();
}

// NOTE: We declare Reference type of NodeIter and NeighborIter as Node* (see
// https://en.cppreference.com/w/cpp/iterator/iterator).

// Iterator for stepping through the nodes of a graph.
class NodeIter
    : public std::iterator<std::forward_iterator_tag, Node, std::ptrdiff_t,
                           /*Pointer*/ Node*, /*Reference*/ Node*> {
 public:
  NodeIter(const Graph* graph, int id);
  bool operator==(const NodeIter& rhs) const;
  bool operator!=(const NodeIter& rhs) const;
  void operator++();
  reference operator*() const;
  pointer operator->() const;

 private:
  // Invariant: id_ == graph_->num_node_ids() || graph_->FindId(id_) != nullptr
  const Graph* graph_;
  int id_;
};

// Iterator for stepping through the neighbors of a node.
class NeighborIter
    : public std::iterator<std::forward_iterator_tag, Node, std::ptrdiff_t,
                           /*Pointer*/ Node*, /*Reference*/ Node*> {
 public:
  NeighborIter(EdgeSet::const_iterator iter, bool incoming);
  bool operator==(const NeighborIter& rhs) const;
  bool operator!=(const NeighborIter& rhs) const;
  void operator++();
  reference operator*() const;
  pointer operator->() const;

 private:
  EdgeSet::const_iterator iter_;
  bool incoming_;
};

// IMPLEMENTATION DETAILS, PLEASE IGNORE

inline NodeIter::NodeIter(const Graph* graph, int id)
    : graph_(graph), id_(id) {
   std::vector<std::string> mht_103_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_103(mht_103_v, 1403, "", "./tensorflow/core/graph/graph.h", "NodeIter::NodeIter");
}

inline bool NodeIter::operator==(const NodeIter& rhs) const {
  DCHECK(graph_ == rhs.graph_);
  return id_ == rhs.id_;
}

inline bool NodeIter::operator!=(const NodeIter& rhs) const {
  return !(*this == rhs);
}

inline void NodeIter::operator++() {
  while (1) {
    DCHECK_LE(id_, graph_->num_node_ids());
    ++id_;
    if (id_ >= graph_->num_node_ids() || graph_->FindNodeId(id_) != nullptr) {
      return;
    }
  }
}

inline Node* NodeIter::operator*() const {
   std::vector<std::string> mht_104_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_104(mht_104_v, 1427, "", "./tensorflow/core/graph/graph.h", "*");
 return graph_->FindNodeId(id_); }

inline Node* NodeIter::operator->() const { return graph_->FindNodeId(id_); }

inline NeighborIter::NeighborIter(EdgeSet::const_iterator iter, bool incoming)
    : iter_(iter), incoming_(incoming) {
   std::vector<std::string> mht_105_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_105(mht_105_v, 1435, "", "./tensorflow/core/graph/graph.h", "NeighborIter::NeighborIter");
}

inline bool NeighborIter::operator==(const NeighborIter& rhs) const {
  return iter_ == rhs.iter_ && incoming_ == rhs.incoming_;
}

inline bool NeighborIter::operator!=(const NeighborIter& rhs) const {
  return !(*this == rhs);
}

inline void NeighborIter::operator++() { ++iter_; }

inline Node* NeighborIter::operator*() const {
   std::vector<std::string> mht_106_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_106(mht_106_v, 1450, "", "./tensorflow/core/graph/graph.h", "*");

  const Edge* e = *iter_;
  return incoming_ ? e->src() : e->dst();
}

inline Node* NeighborIter::operator->() const {
  const Edge* e = *iter_;
  return incoming_ ? e->src() : e->dst();
}

inline bool Edge::IsControlEdge() const {
   std::vector<std::string> mht_107_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_107(mht_107_v, 1463, "", "./tensorflow/core/graph/graph.h", "Edge::IsControlEdge");

  // Note that if either src_output_ or dst_input_ is kControlSlot,
  // so is the other one (AddEdge checks this).
  return src_output_ == Graph::kControlSlot;
}

inline gtl::iterator_range<NodeIter> Graph::nodes() const {
   std::vector<std::string> mht_108_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_108(mht_108_v, 1472, "", "./tensorflow/core/graph/graph.h", "Graph::nodes");

  // Note that NodeId 0 is always valid since we don't let the source
  // node be removed from the graph.
  return gtl::make_range(NodeIter(this, 0), NodeIter(this, num_node_ids()));
}

inline gtl::iterator_range<NodeIter> Graph::op_nodes() const {
   std::vector<std::string> mht_109_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_109(mht_109_v, 1481, "", "./tensorflow/core/graph/graph.h", "Graph::op_nodes");

  // Note that NodeId 0 is always valid since we don't let the source
  // node be removed from the graph.
  //
  // The current implementation of Graph maintains the invariant that the
  // first two nodes are the source and sink nodes, and all other nodes are op
  // nodes. This method (op_nodes()) relies on this invariant.
  NodeIter begin(this, 0);
  NodeIter end(this, num_node_ids());
  if (begin != end) {
    ++begin;
  }
  if (begin != end) {
    ++begin;
  }
  return gtl::make_range(begin, end);
}

inline void Node::set_assigned_device_name_index(int index) {
   std::vector<std::string> mht_110_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_110(mht_110_v, 1502, "", "./tensorflow/core/graph/graph.h", "Node::set_assigned_device_name_index");

  graph_->CheckDeviceNameIndex(index);
  assigned_device_name_index_ = index;
}

inline void Node::set_assigned_device_name(const std::string& device_name) {
   std::vector<std::string> mht_111_v;
   mht_111_v.push_back("device_name: \"" + device_name + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_111(mht_111_v, 1511, "", "./tensorflow/core/graph/graph.h", "Node::set_assigned_device_name");

  graph_->set_assigned_device_name(this, device_name);
}

inline const std::string& Node::assigned_device_name() const {
   std::vector<std::string> mht_112_v;
   MHTracer_DTPStensorflowPScorePSgraphPSgraphDTh mht_112(mht_112_v, 1518, "", "./tensorflow/core/graph/graph.h", "Node::assigned_device_name");

  return graph_->get_assigned_device_name(*this);
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPH_GRAPH_H_
