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
class MHTracer_DTPStensorflowPScorePSgraphPSsubgraphDTcc {
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
   MHTracer_DTPStensorflowPScorePSgraphPSsubgraphDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSgraphPSsubgraphDTcc() {
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

#include "tensorflow/core/graph/subgraph.h"

#include <algorithm>
#include <deque>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/tensor_id.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace subgraph {

// ----------------------------------------------------------------------------
// Subgraph construction-related routines
// ----------------------------------------------------------------------------
// TODO(vrv): Profile the unordered_set and unordered_map use in this file to
// see if we should use an alternative implementation.

namespace {

typedef std::unordered_map<StringPiece, Node*, StringPieceHasher> NameIndex;

// Rewrite graph by replacing the output tensors specified in
// "fed_outputs" with special feed nodes for each specified output
// tensor, and removing any nodes that are now disconnected from the
// part of the graph that reaches the sink node.  The set of special
// feed nodes added to the graph are returned in "*feed_nodes".
//
// Return true on success.  On error, return false and sets *error to
// an appropriate error message (and *g is left in an indeterminate
// state).
Status FeedInputs(
    Graph* g, const std::vector<std::unique_ptr<PruneRewrite>>& feed_rewrites,
    NameIndex* name_index, DataTypeVector* out_feed_types) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSgraphPSsubgraphDTcc mht_0(mht_0_v, 228, "", "./tensorflow/core/graph/subgraph.cc", "FeedInputs");

  out_feed_types->clear();
  out_feed_types->reserve(feed_rewrites.size());
  for (size_t i = 0; i < feed_rewrites.size(); ++i) {
    const string& t = feed_rewrites[i]->endpoint_name();
    TensorId id(ParseTensorName(t));

    auto iter = name_index->find(id.first);
    if (iter == name_index->end()) {
      return errors::NotFound("FeedInputs: unable to find feed output ", t);
    }
    Node* n = iter->second;
    DCHECK_EQ(n->name(), id.first);
    if (id.second >= n->num_outputs()) {
      return errors::InvalidArgument(
          "FeedInputs: ", t, " should have output index < ", n->num_outputs());
    }

    Node* feed_node;
    TF_RETURN_IF_ERROR(
        feed_rewrites[i]->AddNode(g, {n, id.second}, &feed_node));

    // Update name_index
    (*name_index)[feed_node->name()] = feed_node;
    // Duplicate control edges aren't allowed, but feed_node was *just* created
    // so there's no need to check for a duplicate.
    g->AddControlEdge(g->source_node(), feed_node, true);

    // Look through edges coming out of "n" for edges whose src_output() index
    // matches "output_index".  If found, replace the edges with a connection
    // from the special feed node.
    std::vector<const Edge*> to_remove;
    for (const Edge* e : n->out_edges()) {
      if (e->src_output() == id.second) {
        to_remove.emplace_back(e);
      } else if (e->src_output() == Graph::kControlSlot &&
                 (n->type_string() == "Placeholder" ||
                  n->type_string() == "PlaceholderV2")) {
        // When feeding a Placeholder node, any outgoing control edges
        // will be replaced with a control edge from the replacement
        // feed_node.
        // TODO(josh11b,mrry): Come up with a more elegant way of addressing
        // the general version of this problem.
        to_remove.emplace_back(e);
      }
    }

    for (const Edge* e : to_remove) {
      if (e->src_output() == id.second) {
        g->AddEdge(feed_node, 0, e->dst(), e->dst_input());
      } else {
        CHECK_EQ(Graph::kControlSlot, e->src_output());
        // Duplicate control edges aren't allowed, but feed_node was *just*
        // created so there's no need to check for a duplicate.
        g->AddControlEdge(feed_node, e->dst(), true);
      }
      g->RemoveEdge(e);
    }
    out_feed_types->push_back(BaseType(n->output_type(id.second)));
  }
  return Status::OK();
}

Status FetchOutputs(
    Graph* g, const std::vector<std::unique_ptr<PruneRewrite>>& fetch_rewrites,
    NameIndex* name_index, std::vector<Node*>* out_fetch_nodes,
    DataTypeVector* out_fetch_types) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSgraphPSsubgraphDTcc mht_1(mht_1_v, 297, "", "./tensorflow/core/graph/subgraph.cc", "FetchOutputs");

  out_fetch_nodes->clear();
  out_fetch_nodes->reserve(fetch_rewrites.size());
  for (size_t i = 0; i < fetch_rewrites.size(); ++i) {
    const string& t = fetch_rewrites[i]->endpoint_name();

    // Parse t into node_name and output_index.
    TensorId id(ParseTensorName(t));

    // Find node in graph with that name.
    auto iter = name_index->find(id.first);
    if (iter == name_index->end()) {
      return errors::NotFound("FetchOutputs node ", t, ": not found");
    }
    Node* n = iter->second;
    DCHECK_EQ(n->name(), id.first);
    VLOG(2) << "Found fetch node for " << t;

    // Validate output_index
    if (n->num_outputs() == 0) {
      return errors::InvalidArgument(
          "Tried to fetch data for '", t,
          "', which produces no output.  To run to a node but not fetch any "
          "data, pass '",
          t,
          "' as an argument to the 'target_node_names' argument of the "
          "Session::Run API.");
    } else if (id.second >= n->num_outputs()) {
      return errors::InvalidArgument("FetchOutputs ", t,
                                     ": output index too large, must be < ",
                                     n->num_outputs());
    }

    // Create the fetch Node and connect it up
    Node* fetch_node;
    TF_RETURN_IF_ERROR(
        fetch_rewrites[i]->AddNode(g, {n, id.second}, &fetch_node));

    // Update the index.
    (*name_index)[fetch_node->name()] = fetch_node;

    // Duplicate control edges aren't allowed, but fetch_node was *just* created
    // so there's no need to check for a duplicate.
    g->AddControlEdge(fetch_node, g->sink_node(), true);
    out_fetch_nodes->push_back(fetch_node);
    out_fetch_types->push_back(BaseType(n->output_type(id.second)));
  }

  return Status::OK();
}

bool AddNodeToTargets(const string& node_or_tensor_name,
                      const NameIndex& name_index,
                      std::unordered_set<const Node*>* targets) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("node_or_tensor_name: \"" + node_or_tensor_name + "\"");
   MHTracer_DTPStensorflowPScorePSgraphPSsubgraphDTcc mht_2(mht_2_v, 354, "", "./tensorflow/core/graph/subgraph.cc", "AddNodeToTargets");

  TensorId id = ParseTensorName(node_or_tensor_name);
  auto iter = name_index.find(id.first);
  if (iter == name_index.end()) {
    return false;
  }
  const Node* n = iter->second;
  CHECK_EQ(n->name(), id.first);
  targets->insert(n);
  return true;
}

Status PruneForTargets(Graph* g, const NameIndex& name_index,
                       const std::vector<Node*>& fetch_nodes,
                       const gtl::ArraySlice<string>& target_nodes) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSgraphPSsubgraphDTcc mht_3(mht_3_v, 371, "", "./tensorflow/core/graph/subgraph.cc", "PruneForTargets");

  string not_found;
  std::unordered_set<const Node*> targets;
  for (Node* n : fetch_nodes) {
    if (!AddNodeToTargets(n->name(), name_index, &targets)) {
      strings::StrAppend(&not_found, n->name(), " ");
    }
  }
  for (const string& s : target_nodes) {
    if (!AddNodeToTargets(s, name_index, &targets)) {
      strings::StrAppend(&not_found, s, " ");
    }
  }
  if (!not_found.empty()) {
    return errors::NotFound("PruneForTargets: Some target nodes not found: ",
                            not_found);
  }
  PruneForReverseReachability(g, std::move(targets));

  // Reconnect nodes with no outgoing edges to the sink node
  FixupSourceAndSinkEdges(g);

  return Status::OK();
}

}  // namespace

Status ArgFeedRewrite::AddNode(Graph* g, NodeBuilder::NodeOut feed_tensor,
                               Node** out_node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSgraphPSsubgraphDTcc mht_4(mht_4_v, 402, "", "./tensorflow/core/graph/subgraph.cc", "ArgFeedRewrite::AddNode");

  // NOTE(mrry): We must include the index as part of the node
  // name, because _Arg is a "stateful" kernel and therefore
  // its name must uniquely identify a kernel instance across all
  // graphs in the same session.
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat("_arg_", feed_tensor.node->name(), "_",
                                  feed_tensor.index, "_", arg_index_),
                  "_Arg")
          .Attr("T", BaseType(feed_tensor.node->output_type(feed_tensor.index)))
          .Attr("index", arg_index_)
          .Finalize(g, out_node, /*consume=*/true));
  (*out_node)->set_assigned_device_name(device_info().name());
  return Status::OK();
}

Status RecvFeedRewrite::AddNode(Graph* g, NodeBuilder::NodeOut feed_tensor,
                                Node** out_node) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSgraphPSsubgraphDTcc mht_5(mht_5_v, 422, "", "./tensorflow/core/graph/subgraph.cc", "RecvFeedRewrite::AddNode");

  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat("_recv_", feed_tensor.node->name(), "_",
                                  feed_tensor.index),
                  "_Recv")
          .Attr("tensor_type",
                BaseType(feed_tensor.node->output_type(feed_tensor.index)))
          .Attr("tensor_name", endpoint_name())
          .Attr("send_device", device_info().name())
          .Attr("recv_device", device_info().name())
          .Attr("send_device_incarnation",
                static_cast<int64_t>(device_info().incarnation()))
          .Attr("client_terminated", true)
          .Finalize(g, out_node, /*consume=*/true));

  (*out_node)->set_assigned_device_name(device_info().name());
  return Status::OK();
}

Status RetvalFetchRewrite::AddNode(Graph* g, NodeBuilder::NodeOut fetch_tensor,
                                   Node** out_node) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSgraphPSsubgraphDTcc mht_6(mht_6_v, 445, "", "./tensorflow/core/graph/subgraph.cc", "RetvalFetchRewrite::AddNode");

  // NOTE(mrry): We must include the index as part of the node
  // name, because _Retval is a "stateful" kernel and therefore
  // its name must uniquely identify a kernel instance across all
  // graphs in the same session.
  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat("_retval_", fetch_tensor.node->name(), "_",
                                  fetch_tensor.index, "_", retval_index_),
                  "_Retval")
          .Input(fetch_tensor.node, fetch_tensor.index)
          .Attr("T",
                BaseType(fetch_tensor.node->output_type(fetch_tensor.index)))
          .Attr("index", retval_index_)
          .Finalize(g, out_node, /*consume=*/true));
  (*out_node)->set_assigned_device_name(device_info().name());
  return Status::OK();
}

Status SendFetchRewrite::AddNode(Graph* g, NodeBuilder::NodeOut fetch_tensor,
                                 Node** out_node) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSgraphPSsubgraphDTcc mht_7(mht_7_v, 467, "", "./tensorflow/core/graph/subgraph.cc", "SendFetchRewrite::AddNode");

  TF_RETURN_IF_ERROR(
      NodeBuilder(strings::StrCat("_send_", fetch_tensor.node->name(), "_",
                                  fetch_tensor.index),
                  "_Send")
          .Input(fetch_tensor.node, fetch_tensor.index)
          .Attr("tensor_name", endpoint_name())
          .Attr("send_device", device_info().name())
          .Attr("recv_device", device_info().name())
          .Attr("send_device_incarnation",
                static_cast<int64_t>(device_info().incarnation()))
          .Attr("client_terminated", true)
          .Finalize(g, out_node, /*consume=*/true));
  (*out_node)->set_assigned_device_name(device_info().name());
  return Status::OK();
}

Status RewriteGraphForExecution(
    Graph* g, const gtl::ArraySlice<string>& fed_outputs,
    const gtl::ArraySlice<string>& fetch_outputs,
    const gtl::ArraySlice<string>& target_node_names,
    const DeviceAttributes& device_info, bool use_function_convention,
    RewriteGraphMetadata* out_metadata) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSgraphPSsubgraphDTcc mht_8(mht_8_v, 492, "", "./tensorflow/core/graph/subgraph.cc", "RewriteGraphForExecution");

  std::vector<std::unique_ptr<PruneRewrite>> feed_rewrites;
  feed_rewrites.reserve(fed_outputs.size());
  if (use_function_convention) {
    for (size_t i = 0; i < fed_outputs.size(); ++i) {
      feed_rewrites.emplace_back(new ArgFeedRewrite(
          &fed_outputs[i], &device_info, static_cast<int32>(i)));
    }
  } else {
    for (const string& fed_output : fed_outputs) {
      feed_rewrites.emplace_back(
          new RecvFeedRewrite(&fed_output, &device_info));
    }
  }

  std::vector<std::unique_ptr<PruneRewrite>> fetch_rewrites;
  fetch_rewrites.reserve(fetch_outputs.size());
  if (use_function_convention) {
    for (size_t i = 0; i < fetch_outputs.size(); ++i) {
      fetch_rewrites.emplace_back(new RetvalFetchRewrite(
          &fetch_outputs[i], &device_info, static_cast<int32>(i)));
    }
  } else {
    for (const string& fetch_output : fetch_outputs) {
      fetch_rewrites.emplace_back(
          new SendFetchRewrite(&fetch_output, &device_info));
    }
  }

  return RewriteGraphForExecution(g, feed_rewrites, fetch_rewrites,
                                  target_node_names, out_metadata);
}

namespace {
template <typename StringContainer>
std::vector<string> ConvertToVector(StringContainer field) {
  return std::vector<string>(field.begin(), field.end());
}
}  // namespace

Status RewriteGraphForExecution(
    Graph* g, const std::vector<std::unique_ptr<PruneRewrite>>& feed_rewrites,
    const std::vector<std::unique_ptr<PruneRewrite>>& fetch_rewrites,
    const gtl::ArraySlice<string>& target_node_names,
    RewriteGraphMetadata* out_metadata) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSgraphPSsubgraphDTcc mht_9(mht_9_v, 539, "", "./tensorflow/core/graph/subgraph.cc", "RewriteGraphForExecution");

  if (fetch_rewrites.empty() && target_node_names.empty()) {
    return errors::InvalidArgument(
        "Must specify at least one target to fetch or execute.");
  }

  std::unordered_set<string> endpoints;
  for (const auto& feed_rewrite : feed_rewrites) {
    auto result = endpoints.insert(feed_rewrite->endpoint_name());
    if (!result.second) {
      return errors::InvalidArgument("Endpoint \"",
                                     feed_rewrite->endpoint_name(),
                                     "\" fed more than once.");
    }
  }

  for (const auto& fetch_rewrite : fetch_rewrites) {
    if (endpoints.count(fetch_rewrite->endpoint_name()) > 0) {
      return errors::InvalidArgument(fetch_rewrite->endpoint_name(),
                                     " is both fed and fetched.");
    }
  }

  // A separate index mapping name to Node*, for use by FeedInputs,
  // FetchOutputs, and PruneForTargets
  NameIndex name_index;
  name_index.reserve(g->num_nodes());
  for (Node* n : g->nodes()) {
    name_index[n->name()] = n;
  }

  // Add the feeds.  This may replace nodes in the graph, including the nodes
  // currently listed in "fetch_rewrites".  We pass "name_index" so the index is
  // kept up to date.
  if (!feed_rewrites.empty()) {
    TF_RETURN_IF_ERROR(
        FeedInputs(g, feed_rewrites, &name_index, &out_metadata->feed_types));
  }

  // Add the fetch nodes, also updating "name_index".
  std::vector<Node*> fetch_nodes;
  if (!fetch_rewrites.empty()) {
    TF_RETURN_IF_ERROR(FetchOutputs(g, fetch_rewrites, &name_index,
                                    &fetch_nodes, &out_metadata->fetch_types));
  }

  // Prune the graph to only compute what is needed for the fetch nodes and the
  // target nodes.
  if (!fetch_nodes.empty() || !target_node_names.empty()) {
    TF_RETURN_IF_ERROR(
        PruneForTargets(g, name_index, fetch_nodes, target_node_names));
  }

  return Status::OK();
}

}  // namespace subgraph

}  // namespace tensorflow
