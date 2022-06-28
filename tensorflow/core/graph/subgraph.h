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

#ifndef TENSORFLOW_CORE_GRAPH_SUBGRAPH_H_
#define TENSORFLOW_CORE_GRAPH_SUBGRAPH_H_
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
class MHTracer_DTPStensorflowPScorePSgraphPSsubgraphDTh {
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
   MHTracer_DTPStensorflowPScorePSgraphPSsubgraphDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgraphPSsubgraphDTh() {
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


#include <string>

#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {
namespace subgraph {

// Information about a graph rewritten by `RewriteGraphForExecution()`.
struct RewriteGraphMetadata {
  // The element type of each tensor fed to this subgraph. The order
  // of types corresponds to the order of tensor names in
  // `fed_outputs` when calling `RewriteGraphForExecution()`.
  DataTypeVector feed_types;
  // The element type of each tensor fetched from this subgraph. The
  // order of types corresponds to the order of tensor names in
  // `fetch_outputs` when calling `RewriteGraphForExecution()`.
  DataTypeVector fetch_types;
};

// Describes the action to take on a particular tensor endpoint (described by
// a "<node_name>:<output_index>" pair) when pruning the graph.
//
// The `AddNode()` method must be overridden to describe this action. The method
// will be invoked once during `RewriteGraphForExecution()` with tensor endpoint
// named by `endpoint_name`, and it may either create a single new node, or fail
// with an error if the resulting graph would be invalid.
class PruneRewrite {
 public:
  // `endpoint_name` and `device_info` must outlive this object.
  PruneRewrite(const string* endpoint_name, const DeviceAttributes* device_info)
      : endpoint_name_(endpoint_name), device_info_(device_info) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgraphPSsubgraphDTh mht_0(mht_0_v, 223, "", "./tensorflow/core/graph/subgraph.h", "PruneRewrite");
}
  virtual ~PruneRewrite() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgraphPSsubgraphDTh mht_1(mht_1_v, 227, "", "./tensorflow/core/graph/subgraph.h", "~PruneRewrite");
}

  // Creates a new node whose output replaces the given `tensor` in graph `g`.
  // The node will be assigned to the device named in `device_info`.
  virtual Status AddNode(Graph* g, NodeBuilder::NodeOut tensor,
                         Node** out_node) = 0;

  // Returns the name of the tensor to which this rewrite applies.
  const string& endpoint_name() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSgraphPSsubgraphDTh mht_2(mht_2_v, 238, "", "./tensorflow/core/graph/subgraph.h", "endpoint_name");
 return *endpoint_name_; }

 protected:
  // The device on which the new node will be created.
  const DeviceAttributes& device_info() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgraphPSsubgraphDTh mht_3(mht_3_v, 245, "", "./tensorflow/core/graph/subgraph.h", "device_info");
 return *device_info_; }

 private:
  const string* const endpoint_name_;          // Not owned.
  const DeviceAttributes* const device_info_;  // Not owned.
};

// Rewrite the graph structure of "*g" to deal with feeding node
// outputs, fetching node outputs, and only running a subset of the
// graph.  "fed_outputs" and "fetch_outputs" are both lists of
// output tensor identifiers in the form of
// "<name>[:<optional_output_index>]", and "target_nodes_str" is a
// lists of target node names in "*g" "g".
//
// In the resulting graph "*g", output edges in "fed_outputs" have
// been redirected to special "_recv" nodes introduced into the graph.
// If these fed nodes are not needed in order to compute the effects
// of the nodes in "target_node_names" and "fetch_outputs", then these may
// be omitted from the graph.
//
// In the resulting graph "*g", additional "_send" nodes are connected
// to every output in "fetch_outputs".  These "_send" nodes are set up
// to execute on the device described by device_info.
//
// On success, returns OK, and sets "*g" to a version of "*g"
// that represents the portions of the graph necessary for producing
// the output of all nodes listed in "target_node_names" and fetching the
// specific node outputs specified in "fetch_outputs".
//
// On failure, returns the error status. Possible errors include:
//    - fed output "node:output_index" does not exist in "*g"
//    - fetch output "node:output_index" does not exist in "*g"
//    - target node "node" does not exist in "*g"
Status RewriteGraphForExecution(
    Graph* g, const gtl::ArraySlice<string>& fed_outputs,
    const gtl::ArraySlice<string>& fetch_outputs,
    const gtl::ArraySlice<string>& target_node_names,
    const DeviceAttributes& device_info, bool use_function_convention,
    RewriteGraphMetadata* out_metadata);

// A more general version of the above function that supports
// customizable rewriting actions for each fed and fetched tensor.
Status RewriteGraphForExecution(
    Graph* g, const std::vector<std::unique_ptr<PruneRewrite>>& feed_rewrites,
    const std::vector<std::unique_ptr<PruneRewrite>>& fetch_rewrites,
    const gtl::ArraySlice<string>& target_node_names,
    RewriteGraphMetadata* out_metadata);

/////////////////////////////////////////////////////////
// Custom rewrite actions for fed and fetched tensors. //
/////////////////////////////////////////////////////////

// A rewrite action that adds an _Arg node for a fed tensor.
class ArgFeedRewrite : public PruneRewrite {
 public:
  ArgFeedRewrite(const string* endpoint_name,
                 const DeviceAttributes* device_info, int32_t arg_index)
      : PruneRewrite(endpoint_name, device_info), arg_index_(arg_index) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgraphPSsubgraphDTh mht_4(mht_4_v, 305, "", "./tensorflow/core/graph/subgraph.h", "ArgFeedRewrite");
}
  Status AddNode(Graph* g, NodeBuilder::NodeOut feed_tensor,
                 Node** out_node) override;

 private:
  const int32 arg_index_;
};

// A rewrite action that adds a client-terminated _Recv node for a fed tensor.
class RecvFeedRewrite : public PruneRewrite {
 public:
  using PruneRewrite::PruneRewrite;
  Status AddNode(Graph* g, NodeBuilder::NodeOut feed_tensor,
                 Node** out_node) override;
};

// A rewrite action that adds a _Retval node for a fetched tensor.
class RetvalFetchRewrite : public PruneRewrite {
 public:
  RetvalFetchRewrite(const string* endpoint_name,
                     const DeviceAttributes* device_info, int32_t retval_index)
      : PruneRewrite(endpoint_name, device_info), retval_index_(retval_index) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgraphPSsubgraphDTh mht_5(mht_5_v, 329, "", "./tensorflow/core/graph/subgraph.h", "RetvalFetchRewrite");
}
  Status AddNode(Graph* g, NodeBuilder::NodeOut fetch_tensor,
                 Node** out_node) override;

 private:
  const int32 retval_index_;
};

// A rewrite action that adds a client-terminated _Send node for a
// fetched tensor.
class SendFetchRewrite : public PruneRewrite {
 public:
  using PruneRewrite::PruneRewrite;
  Status AddNode(Graph* g, NodeBuilder::NodeOut fetch_tensor,
                 Node** out_node) override;
};

}  // namespace subgraph
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPH_SUBGRAPH_H_
