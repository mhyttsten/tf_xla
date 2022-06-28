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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSforward_type_inferenceDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSforward_type_inferenceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSforward_type_inferenceDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/forward_type_inference.h"

#include <functional>
#include <queue>
#include <string>
#include <string_view>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/full_type_util.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {

namespace {

int MAX_VISITS_PER_NODE = 2;

bool all_inputs_closed(const Node& n, const absl::flat_hash_set<int>& closed) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSforward_type_inferenceDTcc mht_0(mht_0_v, 205, "", "./tensorflow/core/common_runtime/forward_type_inference.cc", "all_inputs_closed");

  for (const auto& e : n.in_edges()) {
    if (e->IsControlEdge()) {
      continue;
    }
    if (!closed.contains(e->src()->id())) {
      return false;
    }
  }
  return true;
}

}  // namespace

Status ForwardTypeInferencePass::Run(
    const GraphOptimizationPassOptions& options) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSforward_type_inferenceDTcc mht_1(mht_1_v, 223, "", "./tensorflow/core/common_runtime/forward_type_inference.cc", "ForwardTypeInferencePass::Run");

  VLOG(1) << "ForwardTypeInferencePass::Run";

  DCHECK(options.graph != nullptr);
  Graph* g = options.graph->get();
  DCHECK(g != nullptr);
  FunctionLibraryDefinition* flib_def = options.flib_def;
  DCHECK(flib_def != nullptr);

  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("forward_type_inference_before", *g, flib_def);
  }

  for (Node* n : g->nodes()) {
    // TODO(mdan): Needed?
    n->UpdateProperties();
  }

  static FullTypeDef* no_type = new FullTypeDef();

  auto process_node = [&flib_def](Node* n, bool& updated) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSforward_type_inferenceDTcc mht_2(mht_2_v, 246, "", "./tensorflow/core/common_runtime/forward_type_inference.cc", "lambda");

    VLOG(3) << "  processing " << n->name();
    VLOG(4) << "\n  node: " << n->def().DebugString()
            << "\n  op def: " << n->op_def().DebugString();
    const OpRegistrationData* reg;
    TF_RETURN_IF_ERROR(flib_def->LookUp(n->op_def().name(), &reg));

    if (reg->fwd_type_fn == nullptr) {
      VLOG(4) << "  " << n->name() << " no type inference function";
      return Status::OK();
    }

    std::vector<std::reference_wrapper<const FullTypeDef>> input_types;
    for (const auto& in_edge : n->in_edges()) {
      if (in_edge->IsControlEdge()) {
        continue;
      }
      input_types.push_back(*no_type);
    }
    for (const auto& in_edge : n->in_edges()) {
      if (in_edge->IsControlEdge()) {
        continue;
      }
      VLOG(5) << "  in edge: " << in_edge->DebugString();
      NodeDef* ndef = in_edge->src()->mutable_def();
      if (ndef->has_experimental_type()) {
        const auto& t = ndef->experimental_type();
        if (t.type_id() != TFT_UNSET) {
          DCHECK(t.type_id() == TFT_PRODUCT) << ndef->DebugString();
          DCHECK(t.args_size() > in_edge->src_output()) << ndef->DebugString();
          input_types.at(in_edge->dst_input()) = t.args(in_edge->src_output());
        }
      }
    }

    // TODO(b/224775462): Populate with types from function references.
    TypeRefMap type_vars;

    const auto& infer_ret = reg->fwd_type_fn(input_types, type_vars);
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        infer_ret.status(), "while inferring type of node '", n->name(), "'");
    const auto& infer_type = *infer_ret;

    if (infer_type.type_id() == TFT_UNSET) {
      VLOG(3) << "  " << n->name() << " no new type information";
      return Status::OK();
    }

    if (!n->def().has_experimental_type() ||
        !full_type::IsEqual(n->def().experimental_type(), infer_type)) {
      *(n->mutable_def()->mutable_experimental_type()) = infer_type;
      updated = true;
      VLOG(3) << "  " << n->name() << " updated";
    } else {
      VLOG(3) << "  " << n->name() << " same type after inference";
    }

    return Status::OK();
  };

  std::list<int> queue;
  absl::flat_hash_set<int> in_queue;
  absl::flat_hash_map<int, int> visit_count;
  // Open nodes. A node is open if it has never been visited.
  absl::flat_hash_set<int> open;
  // Closed nodes. A closed node will never be visited again.
  absl::flat_hash_set<int> closed;

  // Upper bound. Worst-case is a cycle in which no nodes have type info,
  // case in which there will be max_passes iterations, each visiting one node.
  int max_passes = g->num_nodes();

  int visits = 0;

  // Start with niladic nodes. If none exist, a random one will be selected at
  // the end of first iteration.
  for (Node* n : g->nodes()) {
    const int nid = n->id();
    bool niladic = true;
    for (const auto& e : n->in_edges()) {
      if (!e->IsControlEdge()) {
        niladic = false;
        break;
      }
    }
    if (niladic) {
      queue.emplace_back(nid);
      in_queue.emplace(nid);
    }
    open.emplace(nid);
    visit_count.emplace(nid, 0);
  }

  for (int i = 0; i < max_passes; i++) {
    VLOG(2) << "Iteration " << i << ", " << queue.size() << " nodes in queue";

    while (!queue.empty()) {
      int nid = queue.front();
      Node* n = g->FindNodeId(nid);
      bool updated = false;
      VLOG(3) << "  visiting " << n->name();
      visits++;
      visit_count[nid]++;

      TF_RETURN_IF_ERROR(process_node(n, updated));
      VLOG(4) << "  done " << n->def().DebugString();

      queue.pop_front();
      in_queue.erase(nid);
      open.erase(nid);

      if (all_inputs_closed(*n, closed)) {
        VLOG(3) << "  closing " << n->name();
        closed.emplace(nid);
      }

      for (const auto& out_edge : n->out_edges()) {
        if (out_edge->IsControlEdge()) {
          continue;
        }
        Node* c = out_edge->dst();
        int cid = c->id();
        // Update the graph to fixed point, with iterations limited
        // by MAX_VISITS_PER_NODE.
        if (closed.contains(cid) || in_queue.contains(cid) ||
            visit_count.at(cid) >= MAX_VISITS_PER_NODE) {
          continue;
        }
        if (all_inputs_closed(*c, closed) || updated) {
          queue.emplace_back(cid);
          in_queue.emplace(cid);
        }
      }
    }

    VLOG(2) << "Done iteration " << i << ", " << closed.size()
            << " nodes closed";

    if (open.empty()) {
      VLOG(1) << "Finished after " << i << " iterations; done " << closed.size()
              << " of " << g->num_nodes() << " nodes in " << visits
              << " visits";
      break;
    } else {
      queue.emplace_back(*(open.begin()));
    }
  }

  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("forward_type_inference_after", *g, flib_def);
  }

  return Status::OK();
}

Status WeakForwardTypeInferencePass::Run(
    const GraphOptimizationPassOptions& options) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSforward_type_inferenceDTcc mht_3(mht_3_v, 405, "", "./tensorflow/core/common_runtime/forward_type_inference.cc", "WeakForwardTypeInferencePass::Run");

  ForwardTypeInferencePass pass;
  const auto& pass_status = pass.Run(options);
  if (!pass_status.ok()) {
    LOG_FIRST_N(WARNING, 1)
        << "Type inference failed. This indicates an "
           "invalid graph that escaped type checking. Error message: "
        << pass_status.ToString();
  }
  return Status::OK();
}

// Note: This needs to run last because Placer needs it.
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 99999,
                      WeakForwardTypeInferencePass);

}  // namespace tensorflow
