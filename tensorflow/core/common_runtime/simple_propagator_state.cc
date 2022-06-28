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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSsimple_propagator_stateDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsimple_propagator_stateDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSsimple_propagator_stateDTcc() {
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
#include "tensorflow/core/common_runtime/simple_propagator_state.h"

#include <atomic>

#include "tensorflow/core/common_runtime/propagator_debug_utils.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace tensorflow {

SimplePropagatorState::SimplePropagatorState(
    const ImmutableExecutorState& immutable_state, int64_t step_id, bool vlog)
    : SimplePropagatorState(immutable_state, step_id,
                            immutable_state.get_root_frame_info(), vlog) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsimple_propagator_stateDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/common_runtime/simple_propagator_state.cc", "SimplePropagatorState::SimplePropagatorState");
}

SimplePropagatorState::SimplePropagatorState(
    const ImmutableExecutorState& immutable_state, int64_t step_id,
    const ImmutableExecutorState::FrameInfo& finfo, bool vlog)
    : immutable_state_(immutable_state),
      step_id_(step_id),
      vlog_(vlog || VLOG_IS_ON(1)),
      input_tensors_(finfo.total_inputs),
      pending_(
          new std::atomic<int32>[immutable_state.graph_view().num_nodes()]),
      active_(vlog_ ? new std::vector<bool>(
                          immutable_state.graph_view().num_nodes())
                    : nullptr),
      nodes_(finfo.nodes.get()) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsimple_propagator_stateDTcc mht_1(mht_1_v, 214, "", "./tensorflow/core/common_runtime/simple_propagator_state.cc", "SimplePropagatorState::SimplePropagatorState");

  immutable_state_.copy_pending_counts(pending_.get());
}

SimplePropagatorState::~SimplePropagatorState() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsimple_propagator_stateDTcc mht_2(mht_2_v, 221, "", "./tensorflow/core/common_runtime/simple_propagator_state.cc", "SimplePropagatorState::~SimplePropagatorState");
}

void SimplePropagatorState::ActivateRoots(
    gtl::ArraySlice<const NodeItem*> roots, TaggedNodeSeq* ready) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsimple_propagator_stateDTcc mht_3(mht_3_v, 227, "", "./tensorflow/core/common_runtime/simple_propagator_state.cc", "SimplePropagatorState::ActivateRoots");

  for (const NodeItem* item : roots) {
    DCHECK_EQ(item->num_inputs, 0);
    ready->push_back(TaggedNode{item});
  }
}

void SimplePropagatorState::PropagateOutputs(const TaggedNode& tagged_node,
                                             EntryVector* outputs,
                                             TaggedNodeSeq* ready) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsimple_propagator_stateDTcc mht_4(mht_4_v, 239, "", "./tensorflow/core/common_runtime/simple_propagator_state.cc", "SimplePropagatorState::PropagateOutputs");

  profiler::TraceMe activity(
      [&]() {
        return strings::StrCat(
            "ExecutorPropagateOutputs#", "id=", step_id_,
            ",kernel_name=", tagged_node.node_item->kernel->name_view(),
            ",num_output_edges=", tagged_node.node_item->num_output_edges,
            ",num_output_control_edges=",
            tagged_node.node_item->num_output_control_edges, "#");
      },
      profiler::GetTFTraceMeLevel(/*is_expensive=*/false));

  // Propagates outputs along out edges, and puts newly ready nodes
  // into the ready queue.
  DCHECK(ready->empty());

  const GraphView& gview = immutable_state_.graph_view();
  const NodeItem* item = tagged_node.node_item;

  for (const EdgeInfo& e : item->output_edges()) {
    const int dst_id = e.dst_id;
    const int src_slot = e.output_slot;
    const int dst_loc = e.input_slot;

    // NOTE(mrry): The write to `input_tensors_[dst_loc]` must happen before
    // the pending count update, or else one thread might conclude that the
    // count has dropped to zero before another thread finishes updating the
    // input.
    if (e.is_last) {
      input_tensors_[dst_loc] = std::move((*outputs)[src_slot]);
    } else {
      input_tensors_[dst_loc] = (*outputs)[src_slot];
    }

    int32_t previous_num_pending =
        pending_[dst_id].fetch_sub(1, std::memory_order_release);
    if (previous_num_pending == 1) ready->emplace_back(&gview.node_ref(dst_id));
  }

  for (const ControlEdgeInfo& e : item->output_control_edges()) {
    const int dst_id = e.dst_id;

    int32_t previous_num_pending =
        pending_[dst_id].fetch_sub(1, std::memory_order_release);
    if (previous_num_pending == 1) ready->emplace_back(&gview.node_ref(dst_id));
  }
}

void SimplePropagatorState::DumpState() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsimple_propagator_stateDTcc mht_5(mht_5_v, 290, "", "./tensorflow/core/common_runtime/simple_propagator_state.cc", "SimplePropagatorState::DumpState");

  mutex_lock l(mu_);
  // Dump any waiting nodes that are holding on to tensors.
  for (const NodeItem* node : *nodes_) {
    if (pending_[node->node_id]) {
      DumpPendingNodeState(*node, input_tensors_.data(), false);
    }
  }
  // Then the active nodes.
  for (const NodeItem* node : *nodes_) {
    if ((*active_)[node->node_id]) {
      DumpActiveNodeState(*node, input_tensors_.data());
    }
  }
  // Show all input tensors in use.
  size_t total_bytes = 0;
  for (size_t i = 0; i < input_tensors_.size(); ++i) {
    const Entry& input = input_tensors_[i];
    const Tensor* tensor = GetTensorValueForDump(input);
    if (tensor && tensor->IsInitialized()) {
      LOG(WARNING) << "    Input " << i << ": "
                   << strings::StrCat(
                          "Tensor<type: ", DataTypeString(tensor->dtype()),
                          " shape: ", tensor->shape().DebugString(),
                          ", bytes: ", tensor->TotalBytes(), ">");
      total_bytes += tensor->TotalBytes();
    }
  }
  LOG(WARNING) << "    Total bytes " << total_bytes;
}

}  // namespace tensorflow
