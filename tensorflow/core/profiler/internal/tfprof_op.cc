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
class MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_opDTcc() {
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

#include "tensorflow/core/profiler/internal/tfprof_op.h"

#include <stdio.h>

#include <utility>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/profiler/internal/tfprof_constants.h"
#include "tensorflow/core/profiler/internal/tfprof_tensor.h"

namespace tensorflow {
namespace tfprof {
namespace {
string FormatToalExecTime(const ShowMultiNode* node,
                          const ShowMultiNode* root) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_opDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/profiler/internal/tfprof_op.cc", "FormatToalExecTime");

  double accu_pct = 0.0;
  double pct = 0.0;
  if (node->proto().total_exec_micros() > 0) {
    accu_pct = 100.0 * node->proto().total_exec_micros() /
               root->proto().total_exec_micros();
    pct =
        100.0 * node->proto().exec_micros() / root->proto().total_exec_micros();
  }

  return absl::StrFormat(
      "%30s",
      absl::StrFormat("%s (%.2f%%, %.2f%%)",
                      FormatTime(node->proto().exec_micros()), accu_pct, pct));
}
string FormatCPUExecTime(const ShowMultiNode* node, const ShowMultiNode* root) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_opDTcc mht_1(mht_1_v, 219, "", "./tensorflow/core/profiler/internal/tfprof_op.cc", "FormatCPUExecTime");

  double accu_pct = 0.0;
  double pct = 0.0;
  if (node->proto().total_cpu_exec_micros() > 0) {
    accu_pct = 100.0 * node->proto().total_cpu_exec_micros() /
               root->proto().total_cpu_exec_micros();
    pct = 100.0 * node->proto().cpu_exec_micros() /
          root->proto().total_cpu_exec_micros();
  }

  return absl::StrFormat(
      "%30s", absl::StrFormat("%s (%.2f%%, %.2f%%)",
                              FormatTime(node->proto().cpu_exec_micros()),
                              accu_pct, pct));
}
string FormatAcceleratorExecTime(const ShowMultiNode* node,
                                 const ShowMultiNode* root) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_opDTcc mht_2(mht_2_v, 238, "", "./tensorflow/core/profiler/internal/tfprof_op.cc", "FormatAcceleratorExecTime");

  double accu_pct = 0.0;
  double pct = 0.0;
  if (node->proto().total_accelerator_exec_micros() > 0) {
    accu_pct = 100.0 * node->proto().total_accelerator_exec_micros() /
               root->proto().total_accelerator_exec_micros();
    pct = 100.0 * node->proto().accelerator_exec_micros() /
          root->proto().total_accelerator_exec_micros();
  }

  return absl::StrFormat(
      "%30s",
      absl::StrFormat("%s (%.2f%%, %.2f%%)",
                      FormatTime(node->proto().accelerator_exec_micros()),
                      accu_pct, pct));
}
}  // namespace

void TFOp::AddNode(TFGraphNode* node) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_opDTcc mht_3(mht_3_v, 259, "", "./tensorflow/core/profiler/internal/tfprof_op.cc", "TFOp::AddNode");

  const string& op = node->op();
  if (tfcnodes_map_.find(op) == tfcnodes_map_.end()) {
    tfcnodes_map_[op] =
        std::unique_ptr<TFMultiGraphNode>(new TFMultiGraphNode(op));
  }
  TFMultiGraphNode* tfcnode = tfcnodes_map_[op].get();
  tfcnode->AddGraphNode(node);
}

void TFOp::Build() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_opDTcc mht_4(mht_4_v, 272, "", "./tensorflow/core/profiler/internal/tfprof_op.cc", "TFOp::Build");

  for (auto& tn : tfcnodes_map_) {
    cnodes_map_[tn.first] =
        std::unique_ptr<OpNode>(new OpNode(tn.second.get()));
  }

  tfcnodes_map_[kTFProfRoot] =
      std::unique_ptr<TFMultiGraphNode>(new TFMultiGraphNode(kTFProfRoot));
  root_.reset(new OpNode(tfcnodes_map_[kTFProfRoot].get()));
}

const ShowMultiNode* TFOp::ShowInternal(const Options& opts,
                                        Timeline* timeline) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_opDTcc mht_5(mht_5_v, 287, "", "./tensorflow/core/profiler/internal/tfprof_op.cc", "TFOp::ShowInternal");

  root_->ResetTotalStats();
  if (opts.output_type == kOutput[3]) {
    absl::FPrintF(stderr, "Only 'code' view supports pprof output now.\n");
    return root_.get();
  }
  if (opts.output_type == kOutput[1] || opts.output_type == kOutput[2]) {
    root_->formatted_str = FormatNode(root_.get(), root_.get(), opts);
  }
  if (timeline) {
    absl::FPrintF(stderr,
                  "op view doesn't support timeline yet. "
                  "Consider graph/scope/code view.\n");
    return root_.get();
  }
  if (cnodes_map_.empty()) {
    return root_.get();
  }

  std::vector<OpNode*> nodes;
  for (auto& n : cnodes_map_) {
    n.second->account = ReAccount(n.second.get(), opts);
    n.second->ResetTotalStats();
    n.second->AddSelfToTotalStats();
    nodes.push_back(n.second.get());
  }
  nodes = SortNodes(nodes, opts);
  // pre keeps track of previous visited node.
  OpNode* pre = nullptr;
  std::vector<OpNode*> account_nodes;
  for (auto it = nodes.rbegin(); it != nodes.rend(); ++it) {
    if ((*it)->account) {
      if (pre) (*it)->AggregateTotalStats(pre);
      account_nodes.push_back(*it);
      pre = *it;
    }
  }
  std::reverse(std::begin(account_nodes), std::end(account_nodes));
  if (pre) {
    root_->AggregateTotalStats(pre);
  }

  // Perform the display and optionally redo accounting.
  int64_t depth = 0;
  std::vector<OpNode*> show_nodes;
  int64_t start = SearchRoot(account_nodes, opts.start_name_regexes);
  for (int64_t i = start, end = account_nodes.size(); i < end; ++i, ++depth) {
    OpNode* n = account_nodes[i];
    if (ShouldTrim(n, opts.trim_name_regexes) || depth > opts.max_depth) {
      break;
    }
    n->show = ShouldShow(n, opts, depth);
    if (n->show) show_nodes.push_back(n);
  }

  pre = nullptr;
  for (auto it = show_nodes.rbegin(); it != show_nodes.rend(); ++it) {
    if (opts.account_displayed_op_only) {
      (*it)->ResetTotalStats();
      (*it)->AddSelfToTotalStats();
      if (pre) (*it)->AggregateTotalStats(pre);
    }
    pre = *it;
  }
  if (opts.account_displayed_op_only) {
    root_->ResetTotalStats();
    if (pre) {
      root_->AggregateTotalStats(pre);
    }
  }
  if (opts.output_type == kOutput[1] || opts.output_type == kOutput[2]) {
    string display_str = FormatLegend(opts);
    for (OpNode* node : show_nodes) {
      display_str += FormatNode(node, root_.get(), opts);
    }
    // In op view, we don't show root (total). But it will still in proto.
    // TODO(xpan): Is it the right choice?
    root_->formatted_str = display_str;
  }
  // Populate the children field.
  auto* pre_pb = root_->mutable_proto();
  for (auto& show_node : show_nodes) {
    pre_pb->clear_children();
    pre_pb->add_children()->Swap(show_node->mutable_proto());
    pre_pb = pre_pb->mutable_children(0);
  }
  return root_.get();
}

int64_t TFOp::SearchRoot(const std::vector<OpNode*> nodes,
                         const std::vector<string>& regexes) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_opDTcc mht_6(mht_6_v, 380, "", "./tensorflow/core/profiler/internal/tfprof_op.cc", "TFOp::SearchRoot");

  if (regexes.empty() || (regexes.size() == 1 && regexes[0] == ".*")) {
    return 0;
  }
  int64_t i = 0;
  const int64_t nodes_size = nodes.size();
  for (; i < nodes_size; ++i) {
    for (const string& regex : regexes) {
      if (RE2::FullMatch(nodes[i]->name(), regex)) {
        return i;
      }
    }
  }
  return i;
}

string TFOp::FormatMemoryNode(int64_t node_total_bytes,
                              int64_t root_total_bytes,
                              int64_t node_bytes) const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_opDTcc mht_7(mht_7_v, 401, "", "./tensorflow/core/profiler/internal/tfprof_op.cc", "TFOp::FormatMemoryNode");

  double accu_pct = 0.0;
  double pct = 0.0;
  if (node_bytes > 0) {
    accu_pct = 100.0 * node_total_bytes / root_total_bytes;
    pct = 100.0 * node_bytes / root_total_bytes;
  }
  return absl::StrFormat(
      "%30s", absl::StrFormat("%s (%.2f%%, %.2f%%)", FormatMemory(node_bytes),
                              accu_pct, pct));
}

string TFOp::FormatNode(OpNode* node, OpNode* root, const Options& opts) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSprofilerPSinternalPStfprof_opDTcc mht_8(mht_8_v, 416, "", "./tensorflow/core/profiler/internal/tfprof_op.cc", "TFOp::FormatNode");

  std::vector<string> attrs;

  if (opts.select.find(kShown[0]) != opts.select.end()) {
    attrs.push_back(FormatMemoryNode(node->proto().total_requested_bytes(),
                                     root->proto().total_requested_bytes(),
                                     node->proto().requested_bytes()));
  }

  if (opts.select.find(kShown[11]) != opts.select.end()) {
    attrs.push_back(FormatMemoryNode(node->proto().total_peak_bytes(),
                                     root->proto().total_peak_bytes(),
                                     node->proto().peak_bytes()));
  }

  if (opts.select.find(kShown[12]) != opts.select.end()) {
    attrs.push_back(FormatMemoryNode(node->proto().total_residual_bytes(),
                                     root->proto().total_residual_bytes(),
                                     node->proto().residual_bytes()));
  }
  if (opts.select.find(kShown[13]) != opts.select.end()) {
    attrs.push_back(FormatMemoryNode(node->proto().total_output_bytes(),
                                     root->proto().total_output_bytes(),
                                     node->proto().output_bytes()));
  }

  if (opts.select.find(kShown[1]) != opts.select.end()) {
    attrs.push_back(FormatToalExecTime(node, root));
    attrs.push_back(FormatAcceleratorExecTime(node, root));
    attrs.push_back(FormatCPUExecTime(node, root));
  }
  if (opts.select.find(kShown[9]) != opts.select.end() &&
      opts.select.find(kShown[1]) == opts.select.end()) {
    attrs.push_back(FormatAcceleratorExecTime(node, root));
  }
  if (opts.select.find(kShown[10]) != opts.select.end() &&
      opts.select.find(kShown[1]) == opts.select.end()) {
    attrs.push_back(FormatCPUExecTime(node, root));
  }
  if (opts.select.find(kShown[2]) != opts.select.end()) {
    double accu_pct = 0.0;
    double pct = 0.0;
    if (node->proto().total_parameters() > 0) {
      accu_pct = 100.0 * node->proto().total_parameters() /
                 root->proto().total_parameters();
      pct =
          100.0 * node->proto().parameters() / root->proto().total_parameters();
    }
    attrs.push_back(absl::StrFormat(
        "%30s", absl::StrFormat("%s params (%.2f%%, %.2f%%)",
                                FormatNumber(node->proto().parameters()),
                                accu_pct, pct)));
  }

  if (opts.select.find(kShown[3]) != opts.select.end()) {
    double accu_pct = 0.0;
    double pct = 0.0;
    if (node->proto().total_float_ops() > 0) {
      accu_pct = 100.0 * node->proto().total_float_ops() /
                 root->proto().total_float_ops();
      pct = 100.0 * node->proto().float_ops() / root->proto().total_float_ops();
    }

    attrs.push_back(absl::StrFormat(
        "%30s", absl::StrFormat("%s float_ops (%.2f%%, %.2f%%)",
                                FormatNumber(node->proto().float_ops()),
                                accu_pct, pct)));
  }

  if (opts.select.find(kShown[5]) != opts.select.end()) {
    attrs.push_back(absl::StrJoin(node->node->devices(), "|"));
  }

  if (opts.select.find(kShown[6]) != opts.select.end()) {
    std::set<string> op_types = node->node->op_types();
    attrs.push_back(absl::StrJoin(op_types, "|"));
  }

  if (opts.select.find(kShown[7]) != opts.select.end()) {
    int64_t total_runs = 0;
    for (const auto& gnode : node->proto().graph_nodes()) {
      total_runs += gnode.run_count();
    }
    attrs.push_back(absl::StrFormat(
        "%10s", absl::StrFormat("%d|%d", total_runs,
                                node->proto().graph_nodes_size())));
  }

  string node_str =
      absl::StrFormat("%-25s%s\n", node->name(), absl::StrJoin(attrs, ", "));

  if (opts.select.find(kShown[8]) != opts.select.end()) {
    string input_shape_str = FormatInputShapes(node->proto());
    if (!input_shape_str.empty()) {
      node_str = absl::StrFormat("%s\n%s\n\n", node_str, input_shape_str);
    }
  }
  return node_str;
}
}  // namespace tfprof
}  // namespace tensorflow
