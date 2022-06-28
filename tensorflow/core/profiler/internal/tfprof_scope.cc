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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_scopeDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_scopeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_scopeDTcc() {
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

#include "tensorflow/core/profiler/internal/tfprof_scope.h"

#include <stdio.h>

#include <utility>

#include "absl/strings/str_format.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/profiler/internal/tfprof_constants.h"
#include "tensorflow/core/profiler/internal/tfprof_tensor.h"

namespace tensorflow {
namespace tfprof {
ScopeNode* TFScope::CreateParentNode(const string& name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_scopeDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/profiler/internal/tfprof_scope.cc", "TFScope::CreateParentNode");

  if (nodes_map_.find(name) != nodes_map_.end()) {
    return nodes_map_[name].get();
  }
  node_defs_.push_back(std::unique_ptr<NodeDef>(new NodeDef()));
  node_defs_.back()->set_name(name);
  node_defs_.back()->set_op(kTFScopeParent);
  parent_nodes_[name] = std::unique_ptr<TFGraphNode>(
      new TFGraphNode(node_defs_.back().get(), -1, nullptr));
  nodes_map_[name] =
      std::unique_ptr<ScopeNode>(new ScopeNode(parent_nodes_[name].get()));
  return nodes_map_[name].get();
}

void TFScope::AddNode(TFGraphNode* node) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_scopeDTcc mht_1(mht_1_v, 218, "", "./tensorflow/core/profiler/internal/tfprof_scope.cc", "TFScope::AddNode");

  string name = node->name();
  if (nodes_map_.find(node->name()) == nodes_map_.end()) {
    nodes_map_[name] = std::unique_ptr<ScopeNode>(new ScopeNode(node));
  }

  auto last_slash = name.find_last_of('/');
  while (last_slash != name.npos) {
    name = name.substr(0, last_slash);
    if (nodes_map_.find(name) == nodes_map_.end()) {
      CHECK(CreateParentNode(name));
    }
    last_slash = name.find_last_of('/');
  }
}

void TFScope::Build() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_scopeDTcc mht_2(mht_2_v, 237, "", "./tensorflow/core/profiler/internal/tfprof_scope.cc", "TFScope::Build");

  if (root_) return;

  std::vector<ScopeNode*> roots;
  // Found roots, which are nodes without "/".
  for (auto it = nodes_map_.begin(); it != nodes_map_.end(); it++) {
    ScopeNode* node = it->second.get();
    auto last_slash = node->name().find_last_of('/');
    if (last_slash == string::npos) {
      roots.push_back(node);
    } else {
      const string prefix = node->name().substr(0, last_slash);
      nodes_map_[prefix]->children.push_back(node);
    }
  }

  root_ = CreateParentNode(kTFProfRoot);
  root_->children.assign(roots.begin(), roots.end());
}

const ShowNode* TFScope::ShowInternal(const Options& opts, Timeline* timeline) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_scopeDTcc mht_3(mht_3_v, 260, "", "./tensorflow/core/profiler/internal/tfprof_scope.cc", "TFScope::ShowInternal");

  root_->ResetTotalStats();
  if (opts.output_type == kOutput[3]) {
    absl::FPrintF(stderr, "Only 'code' view supports pprof output now.\n");
    return root_;
  }

  std::vector<ScopeNode*> roots = Account(root_->children, opts);
  root_->show_children.clear();
  for (ScopeNode* n : roots) {
    root_->AggregateTotalStats(n);
  }

  if (opts.start_name_regexes.size() != 1 ||
      opts.start_name_regexes[0] != ".*") {
    roots = SearchRoot(roots, opts.start_name_regexes);
  }

  root_->show_children.assign(roots.begin(), roots.end());
  ScopeNode* root = PrintScope({root_}, opts, 1, 0)[0];

  root->formatted_str = FormatLegend(opts) + root->formatted_str;
  Format(root->show_children, &root->formatted_str, root->mutable_proto());

  if (timeline) {
    timeline->GenerateScopeTimeline(root);
  }
  return root;
}

void TFScope::Format(const std::vector<ScopeNode*> roots, string* display_str,
                     GraphNodeProto* proto) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_scopeDTcc mht_4(mht_4_v, 294, "", "./tensorflow/core/profiler/internal/tfprof_scope.cc", "TFScope::Format");

  for (ScopeNode* node : roots) {
    display_str->append(node->formatted_str);
    GraphNodeProto* child = proto->add_children();
    child->MergeFrom(node->proto());
    Format(node->show_children, display_str, child);
  }
}

std::vector<ScopeNode*> TFScope::SearchRoot(
    std::vector<ScopeNode*> roots, const std::vector<string>& regexes) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_scopeDTcc mht_5(mht_5_v, 307, "", "./tensorflow/core/profiler/internal/tfprof_scope.cc", "TFScope::SearchRoot");

  std::vector<ScopeNode*> res;
  if (roots.empty()) {
    return res;
  }
  for (ScopeNode* root : roots) {
    bool match_start_node = false;
    for (const string& regex : regexes) {
      if (RE2::FullMatch(root->name(), regex)) {
        res.push_back(root);
        match_start_node = true;
        break;
      }
    }
    if (match_start_node) {
      // Found a start node at this branch, no need to continue.
      continue;
    }
    std::vector<ScopeNode*> nroots = SearchRoot(root->show_children, regexes);
    res.insert(res.end(), nroots.begin(), nroots.end());
  }
  return res;
}

std::vector<ScopeNode*> TFScope::PrintScope(const std::vector<ScopeNode*> roots,
                                            const Options& opts, int depth,
                                            int last_ident) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_scopeDTcc mht_6(mht_6_v, 336, "", "./tensorflow/core/profiler/internal/tfprof_scope.cc", "TFScope::PrintScope");

  std::vector<ScopeNode*> show_nodes;

  for (ScopeNode* node : roots) {
    int ident = last_ident;
    bool show = ShouldShow(node, opts, depth);
    if (show) ident += 2;

    std::vector<ScopeNode*> show_cnodes;
    if (!ShouldTrim(node, opts.trim_name_regexes) && depth <= opts.max_depth) {
      show_cnodes = PrintScope(node->show_children, opts, depth + 1, ident);
    }
    if (show) {
      node->show_children.clear();
      if (opts.account_displayed_op_only) {
        node->ResetTotalStats();
        node->AddSelfToTotalStats();
      }

      show_cnodes = SortNodes(show_cnodes, opts);
      for (ScopeNode* sc : show_cnodes) {
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

std::vector<ScopeNode*> TFScope::Account(const std::vector<ScopeNode*>& roots,
                                         const Options& opts) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_scopeDTcc mht_7(mht_7_v, 388, "", "./tensorflow/core/profiler/internal/tfprof_scope.cc", "TFScope::Account");

  std::vector<ScopeNode*> act_nodes;

  for (ScopeNode* node : roots) {
    node->ResetTotalStats();
    std::vector<ScopeNode*> act_cnodes = Account(node->children, opts);

    node->account = ReAccount(node, opts);
    if (node->account || !act_cnodes.empty()) {
      node->show_children.clear();
      node->ResetTotalStats();
      node->AddSelfToTotalStats();
      for (ScopeNode* c : act_cnodes) {
        node->AggregateTotalStats(c);
        node->show_children.push_back(c);
      }
      act_nodes.push_back(node);
    }
  }
  return act_nodes;
}
}  // namespace tfprof
}  // namespace tensorflow
