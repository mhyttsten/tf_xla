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
class MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc {
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
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc() {
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

#include "tensorflow/tools/graph_transforms/transform_utils.h"

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/lib/strings/str_util.h"

namespace tensorflow {
namespace graph_transforms {

namespace {
inline bool IsMerge(const NodeDef& node_def) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_0(mht_0_v, 197, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "IsMerge");

  return node_def.op() == "Merge" || node_def.op() == "RefMerge" ||
         node_def.op() == "_XlaMerge";
}

void RecordMatchedNodes(const NodeMatch& match,
                        std::set<string>* matched_nodes) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_1(mht_1_v, 206, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "RecordMatchedNodes");

  matched_nodes->insert(match.node.name());
  for (const NodeMatch& input_match : match.inputs) {
    RecordMatchedNodes(input_match, matched_nodes);
  }
}

inline uint64 Hash64String(const string& input) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("input: \"" + input + "\"");
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_2(mht_2_v, 217, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "Hash64String");

  return Hash64(input.data(), input.size());
}
}  // namespace

void MatchedNodesAsArray(const NodeMatch& match, std::vector<NodeDef>* result) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_3(mht_3_v, 225, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "MatchedNodesAsArray");

  std::set<string> found_nodes;
  std::vector<NodeMatch> current_matches = {match};
  while (!current_matches.empty()) {
    std::vector<NodeMatch> next_matches;
    for (const NodeMatch& current_match : current_matches) {
      if (found_nodes.count(current_match.node.name())) {
        continue;
      }
      found_nodes.insert(current_match.node.name());
      result->push_back(current_match.node);
      for (const NodeMatch& input_match : current_match.inputs) {
        next_matches.push_back(input_match);
      }
    }
    current_matches = next_matches;
  }
}

void MapNamesToNodes(const GraphDef& graph_def,
                     std::map<string, const NodeDef*>* result) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_4(mht_4_v, 248, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "MapNamesToNodes");

  for (const NodeDef& node : graph_def.node()) {
    (*result)[node.name()] = &node;
  }
}

void MapNodesToOutputs(const GraphDef& graph_def,
                       std::map<string, std::vector<const NodeDef*>>* result) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_5(mht_5_v, 258, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "MapNodesToOutputs");

  std::map<string, const NodeDef*> node_map;
  MapNamesToNodes(graph_def, &node_map);
  for (const NodeDef& node : graph_def.node()) {
    for (const string& input : node.input()) {
      string input_node_name = NodeNameFromInput(input);
      (*result)[input_node_name].push_back(&node);
    }
  }
}

void NodeNamePartsFromInput(const string& input_name, string* prefix,
                            string* node_name, string* suffix) {
   std::vector<std::string> mht_6_v;
   mht_6_v.push_back("input_name: \"" + input_name + "\"");
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_6(mht_6_v, 274, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "NodeNamePartsFromInput");

  std::vector<string> input_parts = str_util::Split(input_name, ':');
  if (input_parts.size() < 2) {
    *suffix = "";
  } else {
    *suffix = ":" + input_parts[1];
  }
  StringPiece node_name_piece(input_parts[0]);
  if (absl::ConsumePrefix(&node_name_piece, "^")) {
    *prefix = "^";
  } else {
    *prefix = "";
  }
  *node_name = string(node_name_piece);
}

string NodeNameFromInput(const string& input_name) {
   std::vector<std::string> mht_7_v;
   mht_7_v.push_back("input_name: \"" + input_name + "\"");
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_7(mht_7_v, 294, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "NodeNameFromInput");

  string prefix;
  string node_name;
  string suffix;
  NodeNamePartsFromInput(input_name, &prefix, &node_name, &suffix);
  return node_name;
}

string CanonicalInputName(const string& input_name) {
   std::vector<std::string> mht_8_v;
   mht_8_v.push_back("input_name: \"" + input_name + "\"");
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_8(mht_8_v, 306, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "CanonicalInputName");

  string prefix;
  string node_name;
  string suffix;
  NodeNamePartsFromInput(input_name, &prefix, &node_name, &suffix);
  if (suffix.empty()) {
    suffix = ":0";
  }
  return prefix + node_name + suffix;
}

uint64 HashNodeDef(const NodeDef& node) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_9(mht_9_v, 320, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "HashNodeDef");

  uint64 hash = Hash64String(node.op());
  hash = Hash64Combine(hash, Hash64String(node.name()));
  for (const string& input : node.input()) {
    hash = Hash64Combine(hash, Hash64String(CanonicalInputName(input)));
  }
  hash = Hash64Combine(hash, Hash64String(node.device()));
  std::vector<string> attr_names;
  attr_names.reserve(node.attr().size());
  for (const auto& attr : node.attr()) {
    attr_names.push_back(attr.first);
  }
  std::sort(attr_names.begin(), attr_names.end());
  string attr_serialized;
  for (const string& attr_name : attr_names) {
    auto attr = node.attr().at(attr_name);
    attr.SerializeToString(&attr_serialized);
    hash = Hash64Combine(hash, Hash64String(attr_serialized));
  }
  return hash;
}

void AddNodeInput(const string& input_name, NodeDef* node) {
   std::vector<std::string> mht_10_v;
   mht_10_v.push_back("input_name: \"" + input_name + "\"");
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_10(mht_10_v, 346, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "AddNodeInput");

  *(node->mutable_input()->Add()) = input_name;
}

void CopyNodeAttr(const NodeDef& source, const string& source_key,
                  const string& dest_key, NodeDef* dest) {
   std::vector<std::string> mht_11_v;
   mht_11_v.push_back("source_key: \"" + source_key + "\"");
   mht_11_v.push_back("dest_key: \"" + dest_key + "\"");
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_11(mht_11_v, 356, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "CopyNodeAttr");

  CHECK_NE(0, source.attr().count(source_key))
      << "No key '" << source_key << "' found in " << source.DebugString();
  (*(dest->mutable_attr()))[dest_key] = source.attr().at(source_key);
}

Tensor GetNodeTensorAttr(const NodeDef& node, const string& key) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("key: \"" + key + "\"");
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_12(mht_12_v, 366, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "GetNodeTensorAttr");

  TensorProto tensor_proto = node.attr().at(key).tensor();
  Tensor tensor;
  CHECK(tensor.FromProto(tensor_proto));
  return tensor;
}

void FilterGraphDef(const GraphDef& input_graph_def,
                    std::function<bool(const NodeDef&)> selector,
                    GraphDef* output_graph_def) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_13(mht_13_v, 378, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "FilterGraphDef");

  output_graph_def->mutable_node()->Clear();
  for (const NodeDef& node : input_graph_def.node()) {
    if (selector(node)) {
      *output_graph_def->mutable_node()->Add() = node;
    }
  }
}

void RemoveAttributes(const GraphDef& input_graph_def,
                      const std::vector<string>& attributes,
                      GraphDef* output_graph_def) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_14(mht_14_v, 392, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "RemoveAttributes");

  output_graph_def->mutable_node()->Clear();
  for (const NodeDef& node : input_graph_def.node()) {
    NodeDef* new_node = output_graph_def->mutable_node()->Add();
    *new_node = node;
    for (const string& attribute : attributes) {
      new_node->mutable_attr()->erase(attribute);
    }
  }
}

Status SortByExecutionOrder(const GraphDef& input_graph_def,
                            GraphDef* output_graph_def) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_15(mht_15_v, 407, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "SortByExecutionOrder");

  const int num_nodes = input_graph_def.node_size();
  std::vector<int> ready;
  std::vector<int> pending_count;
  pending_count.reserve(num_nodes);
  std::vector<gtl::InlinedVector<int, 4>> outputs(num_nodes);

  std::map<string, int> name_index;
  for (int i = 0; i < input_graph_def.node_size(); ++i) {
    const NodeDef& node(input_graph_def.node(i));
    name_index[node.name()] = i;
  }

  // Parse the inputs for each node.
  for (int n = 0; n < num_nodes; ++n) {
    const NodeDef& node_def(input_graph_def.node(n));
    if (IsMerge(node_def)) {
      // for merge only wait for one non-control input.
      int32_t num_control_edges = 0;
      for (int i = 0; i < node_def.input_size(); ++i) {
        if (absl::StartsWith(node_def.input(i), "^")) {
          num_control_edges++;
        }
      }
      pending_count.push_back(num_control_edges + 1);
    } else {
      pending_count.push_back(node_def.input_size());
    }
    if (node_def.input_size() == 0) {
      ready.push_back(n);
      continue;
    }
    for (int i = 0; i < node_def.input_size(); ++i) {
      const string& input_name = node_def.input(i);
      const string& input_node_name = NodeNameFromInput(input_name);
      if (!name_index.count(input_node_name)) {
        return errors::InvalidArgument("Node '", node_def.name(),
                                       "': Unknown input node '",
                                       node_def.input(i), "'");
      }
      outputs[name_index[input_node_name]].push_back(n);
    }
  }

  int processed = 0;
  output_graph_def->Clear();
  // Process the NodeDefs in topological order.
  // Code above sets this up by filling in ready_ with nodes that have no
  // inputs, pending_counts_ with the number of inputs for each node and
  // outputs_ with the outputs of each node.
  while (!ready.empty()) {
    int o = ready.back();
    ready.pop_back();
    ++processed;
    const NodeDef& node_def(input_graph_def.node(o));
    *output_graph_def->mutable_node()->Add() = node_def;

    // Update pending_count for outputs.
    for (size_t i = 0; i < outputs[o].size(); ++i) {
      const int output = outputs[o][i];
      pending_count[output]--;
      if (pending_count[output] == 0) {
        ready.push_back(output);
      }
    }
  }

  if (processed < num_nodes) {
    LOG(WARNING) << "IN " << __func__ << (num_nodes - processed)
                 << " NODES IN A CYCLE";
    for (int64_t i = 0; i < num_nodes; i++) {
      if (pending_count[i] != 0) {
        LOG(WARNING) << "PENDING: " << SummarizeNodeDef(input_graph_def.node(i))
                     << "WITH PENDING COUNT = " << pending_count[i];
      }
    }
    return errors::InvalidArgument(num_nodes - processed, " nodes in a cycle");
  }
  return Status::OK();
}

string OpTypePattern::DebugString() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_16(mht_16_v, 491, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "OpTypePattern::DebugString");

  string result = "{" + op + ", {";
  for (const OpTypePattern& input : inputs) {
    result += input.DebugString() + ",";
  }
  result += "}}";
  return result;
}

string NodeMatch::DebugString() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_17(mht_17_v, 503, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "NodeMatch::DebugString");

  string result = "{";
  result += node.DebugString();
  result += ", {";
  for (const NodeMatch& input : inputs) {
    result += input.DebugString() + ",";
  }
  result += "}}";
  return result;
}

GraphMatcher::GraphMatcher(const GraphDef& graph_def) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_18(mht_18_v, 517, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "GraphMatcher::GraphMatcher");

  SortByExecutionOrder(graph_def, &graph_def_).IgnoreError();
  MapNamesToNodes(graph_def_, &node_map_);
}

Status GraphMatcher::GetOpTypeMatches(const OpTypePattern& pattern,
                                      std::vector<NodeMatch>* matches) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_19(mht_19_v, 526, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "GraphMatcher::GetOpTypeMatches");

  std::set<string> matched_nodes;
  for (const NodeDef& node : graph_def_.node()) {
    // Skip any nodes that are already part of a match.
    if (matched_nodes.count(node.name())) {
      continue;
    }
    NodeMatch match;
    if (DoesOpTypeMatch(node, pattern, matched_nodes, &match)) {
      RecordMatchedNodes(match, &matched_nodes);
      matches->push_back(match);
    }
  }
  return Status::OK();
}

bool GraphMatcher::DoesOpTypeMatch(
    const NodeDef& node, const OpTypePattern& pattern,
    const std::set<string>& previously_matched_nodes, NodeMatch* match) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_20(mht_20_v, 547, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "GraphMatcher::DoesOpTypeMatch");

  VLOG(1) << "Looking at node " << node.DebugString();
  VLOG(1) << "pattern=" << pattern.DebugString();
  VLOG(1) << "match=" << match->DebugString();
  if (previously_matched_nodes.count(node.name())) {
    VLOG(1) << "node " << node.name() << " has been previously matched";
    return false;
  }
  bool pattern_matched = false;
  if (pattern.op == "*") {
    pattern_matched = true;
  } else {
    std::vector<string> pattern_ops = str_util::Split(pattern.op, '|');
    for (const string& pattern_op : pattern_ops) {
      if (node.op() == pattern_op) {
        pattern_matched = true;
      }
    }
  }
  if (!pattern_matched) {
    VLOG(1) << "node.op() != pattern.op()";
    return false;
  }
  match->node = node;
  // Ignore any control inputs for pattern-matching purposes
  std::vector<string> non_control_inputs;
  for (const string& input : node.input()) {
    if (!input.empty() && (input[0] != '^')) {
      non_control_inputs.push_back(input);
    }
  }
  if (pattern.inputs.empty()) {
    // If there are no inputs, assume that's the end of the pattern.
    return true;
  }
  if (non_control_inputs.size() != pattern.inputs.size()) {
    VLOG(1) << "non_control_inputs.size() != pattern.inputs.size()";
    return false;
  }
  for (int i = 0; i < pattern.inputs.size(); ++i) {
    const string& input_node_name = NodeNameFromInput(non_control_inputs[i]);
    const NodeDef& input_node = *(node_map_[input_node_name]);
    const OpTypePattern& input_pattern = pattern.inputs[i];
    match->inputs.push_back(NodeMatch());
    NodeMatch* input_match = &(match->inputs.back());
    if (!DoesOpTypeMatch(input_node, input_pattern, previously_matched_nodes,
                         input_match)) {
      return false;
    }
  }
  return true;
}

Status ReplaceMatchingOpTypes(
    const GraphDef& input_graph_def, const OpTypePattern& pattern,
    const std::function<Status(const NodeMatch&, const std::set<string>&,
                               const std::set<string>&, std::vector<NodeDef>*)>&
        node_generator,
    const ReplaceMatchingOpTypesOptions& options, GraphDef* output_graph_def) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_21(mht_21_v, 608, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "ReplaceMatchingOpTypes");

  // Start off by retrieving all the matching subgraphs.
  GraphMatcher matcher(input_graph_def);
  std::vector<NodeMatch> matches;
  TF_RETURN_IF_ERROR(matcher.GetOpTypeMatches(pattern, &matches));

  // Do some housekeeping so we can easily look up the resulting matches given
  // a node name.
  std::set<string> matched_nodes;
  std::map<string, const NodeMatch*> matches_by_head_name;
  for (const NodeMatch& match : matches) {
    matches_by_head_name[match.node.name()] = &match;
    RecordMatchedNodes(match, &matched_nodes);
  }
  std::map<string, std::vector<const NodeDef*>> outputs_map;
  MapNodesToOutputs(input_graph_def, &outputs_map);

  // Go through all the nodes in the input graph, see if they are part of a
  // match or if they can be left untouched.
  output_graph_def->Clear();
  for (const NodeDef& input_node : input_graph_def.node()) {
    if (matches_by_head_name.count(input_node.name())) {
      // This node is the beginning of a match, so call the replacement function
      // after setting up some information it will need.
      const NodeMatch* match = matches_by_head_name[input_node.name()];
      std::vector<NodeDef> matched_nodes_array;
      MatchedNodesAsArray(*match, &matched_nodes_array);
      // This tells us whether a node is part of the current match.
      std::set<string> matched_nodes_lookup;
      for (const NodeDef& matched_node : matched_nodes_array) {
        matched_nodes_lookup.insert(matched_node.name());
      }
      // These are helper arrays that the replacement function can use to tell
      // whether it can safely remove an internal node (because nothing outside
      // of the match uses it) or whether external nodes depend on it.
      std::set<string> input_nodes;
      std::set<string> output_nodes;
      for (const NodeDef& matched_node : matched_nodes_array) {
        // Look through all of this node's inputs, and if any of them come from
        // outside the match, then this should be noted as one of the external
        // inputs of the subgraph.
        for (const string& input_name : matched_node.input()) {
          string input_node_name = NodeNameFromInput(input_name);
          if (!matched_nodes_lookup.count(input_node_name)) {
            input_nodes.insert(matched_node.name());
          }
        }
        // Do a reverse input lookup, to see which other nodes use the current
        // one as an input. If any of those nodes are outside the match
        // subgraph, then the current node is marked as an output node that
        // shouldn't be removed.
        if (outputs_map.count(matched_node.name())) {
          for (const NodeDef* dependent_node :
               outputs_map[matched_node.name()]) {
            if (!matched_nodes_lookup.count(dependent_node->name())) {
              output_nodes.insert(matched_node.name());
            }
          }
        }
      }
      // Call the generator function and add all the returned nodes to the
      // graph.
      std::vector<NodeDef> new_nodes;
      TF_RETURN_IF_ERROR(
          node_generator(*match, input_nodes, output_nodes, &new_nodes));
      std::set<string> new_node_names;
      for (const NodeDef& new_node : new_nodes) {
        new_node_names.insert(new_node.name());
      }
      // Check to make sure the generator function preserved all of the nodes
      // that are used elsewhere in the graph, and add them back in if not.
      bool abort_replacement = false;
      if (!options.allow_inconsistencies) {
        for (const string& expected_output : output_nodes) {
          if (!new_node_names.count(expected_output)) {
            LOG(WARNING) << "Expected " << expected_output
                         << " to be preserved.";
            abort_replacement = true;
          }
        }
      }
      if (abort_replacement) {
        LOG(WARNING) << "Generator function didn't preserve needed nodes, "
                     << "copying old replacements back in instead.";
        std::vector<NodeDef> old_nodes;
        MatchedNodesAsArray(*match, &old_nodes);
        for (const NodeDef& old_node : old_nodes) {
          NodeDef* added_node = output_graph_def->mutable_node()->Add();
          *added_node = old_node;
        }
      } else {
        for (const NodeDef& new_node : new_nodes) {
          NodeDef* added_node = output_graph_def->mutable_node()->Add();
          *added_node = new_node;
        }
      }
    } else if (!matched_nodes.count(input_node.name())) {
      // This node isn't part of any match, so just copy it over.
      NodeDef* added_node = output_graph_def->mutable_node()->Add();
      *added_node = input_node;
    } else {
      // Do nothing, because this is an internal part of a matching subgraph,
      // and so will have been replaced by a new replacement subgraph.
    }
  }

  return Status::OK();
}

Status RenameNodeInputs(const GraphDef& input_graph_def,
                        const std::map<string, string>& inputs_to_rename,
                        const std::unordered_set<string>& nodes_to_ignore,
                        GraphDef* output_graph_def) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_22(mht_22_v, 723, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "RenameNodeInputs");

  std::map<string, std::vector<std::pair<string, string>>>
      canonical_inputs_to_rename;
  for (const auto& input_to_rename : inputs_to_rename) {
    canonical_inputs_to_rename[NodeNameFromInput(input_to_rename.first)]
        .push_back({input_to_rename.first, input_to_rename.second});
  }

  output_graph_def->Clear();
  for (const NodeDef& node : input_graph_def.node()) {
    NodeDef* new_node = output_graph_def->mutable_node()->Add();
    *new_node = node;
    new_node->mutable_input()->Clear();
    for (const string& input_name : node.input()) {
      std::set<string> already_visited;
      string new_input_name = input_name;
      while (
          canonical_inputs_to_rename.count(NodeNameFromInput(new_input_name))) {
        string input_node_name = NodeNameFromInput(new_input_name);
        if (already_visited.count(input_node_name)) {
          return errors::InvalidArgument(
              "RenameNodeInputs argument contains a cycle for ",
              input_node_name);
        }
        already_visited.insert(input_node_name);
        if (nodes_to_ignore.count(node.name())) {
          break;
        }
        bool any_match_found = false;
        for (const std::pair<string, string>& input_to_rename :
             canonical_inputs_to_rename.at(input_node_name)) {
          const string& source_name = input_to_rename.first;
          const string& dest_name = input_to_rename.second;
          bool is_match;
          string match_name;
          if (str_util::EndsWith(source_name, ":*")) {
            is_match = true;
            string prefix;
            string unused_node_name;
            string suffix;
            NodeNamePartsFromInput(new_input_name, &prefix, &unused_node_name,
                                   &suffix);
            match_name = prefix + dest_name + suffix;
          } else {
            is_match = (CanonicalInputName(source_name) ==
                        CanonicalInputName(new_input_name));
            match_name = dest_name;
          }
          if (is_match) {
            new_input_name = match_name;
            any_match_found = true;
          }
        }
        if (!any_match_found) {
          break;
        }
      }
      *(new_node->mutable_input()->Add()) = new_input_name;
    }
  }
  return Status::OK();
}

void CopyOriginalMatch(const NodeMatch& match,
                       std::vector<NodeDef>* new_nodes) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_23(mht_23_v, 790, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "CopyOriginalMatch");

  std::vector<NodeDef> old_nodes;
  MatchedNodesAsArray(match, &old_nodes);
  for (const NodeDef& old_node : old_nodes) {
    new_nodes->push_back(old_node);
  }
}

TransformRegistry* GetTransformRegistry() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_24(mht_24_v, 801, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "GetTransformRegistry");

  static TransformRegistry transform_registry;
  return &transform_registry;
}

void FindInvalidInputs(const GraphDef& graph_def,
                       std::vector<std::pair<string, string>>* invalid_inputs) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_25(mht_25_v, 810, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "FindInvalidInputs");

  std::map<string, const NodeDef*> node_map;
  MapNamesToNodes(graph_def, &node_map);

  for (const NodeDef& node : graph_def.node()) {
    for (const string& input : node.input()) {
      string input_node = NodeNameFromInput(input);
      if (!node_map.count(input_node)) {
        invalid_inputs->push_back({node.name(), input_node});
      }
    }
  }
}

Status IsGraphValid(const GraphDef& graph_def) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_26(mht_26_v, 827, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "IsGraphValid");

  std::vector<std::pair<string, string>> invalid_inputs;
  FindInvalidInputs(graph_def, &invalid_inputs);
  if (!invalid_inputs.empty()) {
    std::map<string, const NodeDef*> node_map;
    MapNamesToNodes(graph_def, &node_map);
    for (const std::pair<string, string>& invalid_input : invalid_inputs) {
      LOG(ERROR) << "Invalid input " << invalid_input.second << " for node "
                 << invalid_input.first << " - "
                 << node_map[invalid_input.first]->DebugString();
    }
    return errors::Internal(
        "Invalid graph with inputs referring to nonexistent nodes");
  }
  return Status::OK();
}

Status GetInOutTypes(const NodeDef& node_def, DataTypeVector* inputs,
                     DataTypeVector* outputs) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_27(mht_27_v, 848, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "GetInOutTypes");

  const OpDef* op_def;
  TF_RETURN_IF_ERROR(OpRegistry::Global()->LookUpOpDef(node_def.op(), &op_def));
  TF_RETURN_IF_ERROR(InOutTypesForNode(node_def, *op_def, inputs, outputs));
  return Status::OK();
}

Status TensorShapeFromString(const string& shape_string, TensorShape* result) {
   std::vector<std::string> mht_28_v;
   mht_28_v.push_back("shape_string: \"" + shape_string + "\"");
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_28(mht_28_v, 859, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "TensorShapeFromString");

  if (shape_string.empty()) {
    return errors::InvalidArgument("Specified shape is empty.");
  }
  std::vector<string> dims_as_str = str_util::Split(shape_string, ",");
  std::vector<int64_t> dims;
  for (const string& dim : dims_as_str) {
    int64_t tmp;
    if (strings::safe_strto64(dim, &tmp)) {
      dims.push_back(tmp);
    } else {
      return errors::InvalidArgument("Could parse as shape: '", shape_string,
                                     "'");
    }
  }
  *result = TensorShape(dims);
  return Status::OK();
}

int TransformFuncContext::CountParameters(const string& name) const {
   std::vector<std::string> mht_29_v;
   mht_29_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_29(mht_29_v, 882, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "TransformFuncContext::CountParameters");

  if (params.count(name)) {
    return params.at(name).size();
  } else {
    return 0;
  }
}

Status TransformFuncContext::GetOneStringParameter(const string& name,
                                                   const string& default_value,
                                                   string* result) const {
   std::vector<std::string> mht_30_v;
   mht_30_v.push_back("name: \"" + name + "\"");
   mht_30_v.push_back("default_value: \"" + default_value + "\"");
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_30(mht_30_v, 897, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "TransformFuncContext::GetOneStringParameter");

  const int params_count = CountParameters(name);
  if (params_count == 0) {
    *result = default_value;
    return Status::OK();
  } else if (params_count == 1) {
    *result = params.at(name).at(0);
    return Status::OK();
  } else {
    return errors::InvalidArgument("Expected a single '", name,
                                   "' parameter, but found ", params_count,
                                   " occurrences");
  }
}

Status TransformFuncContext::GetOneInt32Parameter(const string& name,
                                                  int32_t default_value,
                                                  int32* result) const {
   std::vector<std::string> mht_31_v;
   mht_31_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_31(mht_31_v, 918, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "TransformFuncContext::GetOneInt32Parameter");

  const int params_count = CountParameters(name);
  if (params_count == 0) {
    *result = default_value;
    return Status::OK();
  }
  string string_value;
  TF_RETURN_IF_ERROR(GetOneStringParameter(name, "", &string_value));
  if (!strings::safe_strto32(StringPiece(string_value), result)) {
    return errors::InvalidArgument("Couldn't interpret the ", name,
                                   " argument as a number:", string_value);
  }
  return Status::OK();
}

Status TransformFuncContext::GetOneInt64Parameter(const string& name,
                                                  int64_t default_value,
                                                  int64_t* result) const {
   std::vector<std::string> mht_32_v;
   mht_32_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_32(mht_32_v, 939, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "TransformFuncContext::GetOneInt64Parameter");

  const int params_count = CountParameters(name);
  if (params_count == 0) {
    *result = default_value;
    return Status::OK();
  }
  string string_value;
  TF_RETURN_IF_ERROR(GetOneStringParameter(name, "", &string_value));
  if (!strings::safe_strto64(StringPiece(string_value), result)) {
    return errors::InvalidArgument("Couldn't interpret the ", name,
                                   " argument as a number:", string_value);
  }
  return Status::OK();
}

Status TransformFuncContext::GetOneFloatParameter(const string& name,
                                                  float default_value,
                                                  float* result) const {
   std::vector<std::string> mht_33_v;
   mht_33_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_33(mht_33_v, 960, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "TransformFuncContext::GetOneFloatParameter");

  const int params_count = CountParameters(name);
  if (params_count == 0) {
    *result = default_value;
    return Status::OK();
  }
  string string_value;
  TF_RETURN_IF_ERROR(GetOneStringParameter(name, "", &string_value));
  if (!strings::safe_strtof(string_value.c_str(), result)) {
    return errors::InvalidArgument(
        "Couldn't interpret the ", name,
        " argument as a float number:", string_value);
  }
  return Status::OK();
}

Status TransformFuncContext::GetOneBoolParameter(const string& name,
                                                 bool default_value,
                                                 bool* result) const {
   std::vector<std::string> mht_34_v;
   mht_34_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPStoolsPSgraph_transformsPStransform_utilsDTcc mht_34(mht_34_v, 982, "", "./tensorflow/tools/graph_transforms/transform_utils.cc", "TransformFuncContext::GetOneBoolParameter");

  const int params_count = CountParameters(name);
  if (params_count == 0) {
    *result = default_value;
    return Status::OK();
  }
  string string_value;
  TF_RETURN_IF_ERROR(GetOneStringParameter(name, "", &string_value));
  if (string_value == "true" || string_value == "1") {
    *result = true;
  } else if (string_value == "false" || string_value == "0") {
    *result = false;
  } else {
    return errors::InvalidArgument("Couldn't interpret the ", name,
                                   " argument as a boolean:", string_value,
                                   " (expected true, false, 0 or 1)");
  }
  return Status::OK();
}

}  // namespace graph_transforms
}  // namespace tensorflow
