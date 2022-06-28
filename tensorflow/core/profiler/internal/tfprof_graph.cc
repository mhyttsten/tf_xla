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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_graphDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_graphDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_graphDTcc() {
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

/* Copyright 2016 The TensorFlow Authors All Rights Reserved.

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

#include "tensorflow/core/profiler/internal/tfprof_graph.h"

#include <stdio.h>

#include <utility>

#include "absl/strings/str_format.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/profiler/internal/tfprof_constants.h"
#include "tensorflow/core/profiler/internal/tfprof_tensor.h"

namespace tensorflow {
namespace tfprof {
GraphNode* TFGraph::CreateParentNode(const string& name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_graphDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/profiler/internal/tfprof_graph.cc", "TFGraph::CreateParentNode");

  node_defs_.push_back(std::unique_ptr<NodeDef>(new NodeDef()));
  node_defs_.back()->set_name(name);
  node_defs_.back()->set_op(kTFGraphParent);
  parent_nodes_[name] = std::unique_ptr<TFGraphNode>(
      new TFGraphNode(node_defs_.back().get(), -1, nullptr));
  nodes_map_[name] =
      std::unique_ptr<GraphNode>(new GraphNode(parent_nodes_[name].get()));
  return nodes_map_[name].get();
}

void TFGraph::AddNode(TFGraphNode* node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_graphDTcc mht_1(mht_1_v, 213, "", "./tensorflow/core/profiler/internal/tfprof_graph.cc", "TFGraph::AddNode");

  string name = node->name();
  nodes_map_[name] = std::unique_ptr<GraphNode>(new GraphNode(node));
}

void TFGraph::Build() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_graphDTcc mht_2(mht_2_v, 221, "", "./tensorflow/core/profiler/internal/tfprof_graph.cc", "TFGraph::Build");

  if (root_) return;

  std::set<string> nonroots;
  // Filter out the root nodes (node not input of any other node).
  for (auto it = nodes_map_.begin(); it != nodes_map_.end(); it++) {
    GraphNode* node = it->second.get();
    const std::map<int, string>& inputs = node->node->inputs();
    for (auto inputs_it = inputs.cbegin(); inputs_it != inputs.cend();
         inputs_it++) {
      nonroots.insert(inputs_it->second);
      auto child_it = nodes_map_.find(inputs_it->second);
      if (child_it != nodes_map_.end()) {
        node->children.push_back(child_it->second.get());
      }
    }
  }
  std::vector<GraphNode*> roots;
  for (auto it = nodes_map_.begin(); it != nodes_map_.end(); it++) {
    if (nonroots.find(it->first) == nonroots.end()) {
      roots.push_back(it->second.get());
    }
  }
  root_ = CreateParentNode(kTFProfRoot);
  root_->children.insert(root_->children.end(), roots.begin(), roots.end());
}

const ShowNode* TFGraph::ShowInternal(const Options& opts, Timeline* timeline) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_graphDTcc mht_3(mht_3_v, 251, "", "./tensorflow/core/profiler/internal/tfprof_graph.cc", "TFGraph::ShowInternal");

  root_->ResetTotalStats();
  root_->show_children.clear();

  if (opts.output_type == kOutput[3]) {
    absl::FPrintF(stderr, "Only 'code' view supports pprof output now.\n");
    return root_;
  }
  if (timeline && timeline->step() < 0) {
    // TODO(xpan): Maybe pick a default step for users.
    absl::FPrintF(
        stderr,
        "Must specify -step option to generate timeline in graph view.\n");
    return root_;
  }
  // 1. Account and aggregate the stats based on the graph structure.
  // Returns a graph consists of accounted nodes.
  std::set<string> visits;
  std::vector<GraphNode*> roots = Account(root_->children, opts, &visits);
  for (GraphNode* n : roots) {
    root_->AggregateTotalStats(n);
  }

  // 2. Trim the nodes before start_name_regexes.
  if (opts.start_name_regexes.size() != 1 ||
      opts.start_name_regexes[0] != ".*") {
    visits.clear();
    roots = SearchRoot(roots, opts.start_name_regexes, &visits);
  }

  // 3. Trim the nodes not matching show/hide/trim_name_regexes.
  // If account_displayed_op_only=true, redo the accounting.
  visits.clear();
  root_->show_children.assign(roots.begin(), roots.end());
  GraphNode* root = PrintGraph({root_}, opts, 1, 0, &visits)[0];

  // 4. Prepare output based on the final graphs.
  root->formatted_str = FormatLegend(opts) + root->formatted_str;
  Format(root->show_children, &root->formatted_str, root->mutable_proto());

  if (timeline) {
    timeline->GenerateGraphTimeline(root->show_children);
  }
  return root;
}

std::vector<GraphNode*> TFGraph::SearchRoot(
    const std::vector<GraphNode*>& roots, const std::vector<string>& regexes,
    std::set<string>* visited) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_graphDTcc mht_4(mht_4_v, 302, "", "./tensorflow/core/profiler/internal/tfprof_graph.cc", "TFGraph::SearchRoot");

  std::vector<GraphNode*> res;
  if (roots.empty()) {
    return res;
  }
  for (GraphNode* root : roots) {
    if (visited->find(root->name()) != visited->end()) continue;
    visited->insert(root->name());
    // If the parent is a start point, don't search its children.
    // Note that its children can still be added as start node through
    // another route.
    bool match_start_node = false;
    for (const string& regex : regexes) {
      if (RE2::FullMatch(root->name(), regex)) {
        res.push_back(root);
        match_start_node = true;
        break;
      }
    }
    if (match_start_node) {
      continue;
    }
    std::vector<GraphNode*> nroot =
        SearchRoot(root->show_children, regexes, visited);
    res.insert(res.end(), nroot.begin(), nroot.end());
  }
  return res;
}

void TFGraph::Format(const std::vector<GraphNode*> roots, string* display_str,
                     GraphNodeProto* proto) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_graphDTcc mht_5(mht_5_v, 335, "", "./tensorflow/core/profiler/internal/tfprof_graph.cc", "TFGraph::Format");

  for (GraphNode* node : roots) {
    display_str->append(node->formatted_str);
    GraphNodeProto* child = proto->add_children();
    child->MergeFrom(node->proto());
    Format(node->show_children, display_str, child);
  }
}

std::vector<GraphNode*> TFGraph::PrintGraph(const std::vector<GraphNode*> roots,
                                            const Options& opts, int depth,
                                            int last_ident,
                                            std::set<string>* visits) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_graphDTcc mht_6(mht_6_v, 350, "", "./tensorflow/core/profiler/internal/tfprof_graph.cc", "TFGraph::PrintGraph");

  std::vector<GraphNode*> show_nodes;

  for (GraphNode* node : roots) {
    if (visits->find(node->name()) != visits->end()) continue;
    visits->insert(node->name());

    bool show = ShouldShow(node, opts, depth);
    int indent = last_ident;
    if (show) indent += 2;

    std::vector<GraphNode*> show_cnodes;
    if (!ShouldTrim(node, opts.trim_name_regexes) && depth <= opts.max_depth) {
      show_cnodes =
          PrintGraph(node->show_children, opts, depth + 1, indent, visits);
    }
    if (show) {
      node->show_children.clear();
      if (opts.account_displayed_op_only) {
        node->ResetTotalStats();
        node->AddSelfToTotalStats();
      }

      show_cnodes = SortNodes(show_cnodes, opts);
      for (GraphNode* sc : show_cnodes) {
        node->show_children.push_back(sc);
        if (opts.account_displayed_op_only) {
          node->AggregateTotalStats(sc);
        }
      }
      node->formatted_str = absl::StrFormat(
          "%s%s\n", std::string(last_ident, ' '), FormatNode(node, opts));

      if (opts.select.find(kShown[4]) != opts.select.end()) {
        std::unique_ptr<TFProfTensor> tfprof_tensor;
        if (LookUpCheckPoint(node->name(), &tfprof_tensor)) {
          string value_str;
          tfprof_tensor->Display(&value_str,
                                 node->mutable_proto()->mutable_tensor_value());
          node->formatted_str += value_str;
        }
      }
      show_nodes.push_back(node);
    } else {
      show_nodes.insert(show_nodes.end(), show_cnodes.begin(),
                        show_cnodes.end());
    }
  }
  return show_nodes;
}

std::vector<GraphNode*> TFGraph::Account(const std::vector<GraphNode*>& roots,
                                         const Options& opts,
                                         std::set<string>* visits) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_graphDTcc mht_7(mht_7_v, 406, "", "./tensorflow/core/profiler/internal/tfprof_graph.cc", "TFGraph::Account");

  std::vector<GraphNode*> act_nodes;
  for (GraphNode* node : roots) {
    if (visits->find(node->name()) != visits->end()) continue;
    visits->insert(node->name());
    // Depth-first.
    std::vector<GraphNode*> act_cnodes = Account(node->children, opts, visits);

    node->account = ReAccount(node, opts);
    if (node->account) {
      node->show_children.clear();
      node->ResetTotalStats();
      node->AddSelfToTotalStats();
      // Aggregate its accounted children stats.
      for (GraphNode* c : act_cnodes) {
        node->AggregateTotalStats(c);
        node->show_children.push_back(c);
      }
      act_nodes.push_back(node);
    } else {
      // If the current node is not accounted, pass the children to the
      // ancestor.
      act_nodes.insert(act_nodes.end(), act_cnodes.begin(), act_cnodes.end());
    }
  }
  return act_nodes;
}
}  // namespace tfprof
}  // namespace tensorflow
