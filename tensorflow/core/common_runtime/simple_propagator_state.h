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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_SIMPLE_PROPAGATOR_STATE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_SIMPLE_PROPAGATOR_STATE_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSsimple_propagator_stateDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsimple_propagator_stateDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSsimple_propagator_stateDTh() {
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


#include <vector>

#include "tensorflow/core/common_runtime/entry.h"
#include "tensorflow/core/common_runtime/immutable_executor_state.h"
#include "tensorflow/core/common_runtime/pending_counts.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// Represents the ephemeral "edge state" associated with one invocation of
// `Executor::Run()`.
//
// NOTE: `SimplePropagatorState` does not support "v1-style" control flow,
// including "dead tensors", "Switch" and "Merge" nodes, and cycles in the
// graph. Use `PropagatorState` for graphs with those features.
// `SimplePropagatorState` *does* support "v2-style" or "functional" control
// flow.
//
// `SimplePropagatorState` is responsible for propagating values along dataflow
// edges in a TensorFlow graph and determining which nodes are runnable. The
// executor primarily updates `SimplePropagatorState` by calling
// `PropagateOutputs()` after processing a node, and `SimplePropagatorState`
// dispatches `TaggedNode`s by adding them to a `TaggedNodeSeq`.
class SimplePropagatorState {
 public:
  SimplePropagatorState(const ImmutableExecutorState& immutable_state,
                        int64_t step_id, bool vlog);
  ~SimplePropagatorState();

  // A `TaggedNode` corresponds to a single invocation of a node's kernel,
  // and it is created when the kernel becomes runnable.
  struct TaggedNode {
    const NodeItem* node_item;

    explicit TaggedNode(const NodeItem* node_item) : node_item(node_item) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsimple_propagator_stateDTh mht_0(mht_0_v, 227, "", "./tensorflow/core/common_runtime/simple_propagator_state.h", "TaggedNode");
}

    const NodeItem& get_node_item() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsimple_propagator_stateDTh mht_1(mht_1_v, 232, "", "./tensorflow/core/common_runtime/simple_propagator_state.h", "get_node_item");
 return *node_item; }

    bool get_is_dead() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsimple_propagator_stateDTh mht_2(mht_2_v, 237, "", "./tensorflow/core/common_runtime/simple_propagator_state.h", "get_is_dead");
 return false; }
    int64_t get_iter_num() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsimple_propagator_stateDTh mht_3(mht_3_v, 241, "", "./tensorflow/core/common_runtime/simple_propagator_state.h", "get_iter_num");
 return 0; }
  };

  // A drop-in replacement for std::deque<TaggedNode>.  We typically don't
  // have that many nodes in the ready queue, so we just use a vector and
  // don't free up memory from the queue as we consume nodes.
  // TODO(mrry): Extract this and share it with the version in
  // `PropagatorState`. The correct constants might be different, since
  // sizeof(TaggedNode) is smaller in this version.
  class TaggedNodeReadyQueue {
   public:
    TaggedNodeReadyQueue() : front_index_(0) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsimple_propagator_stateDTh mht_4(mht_4_v, 255, "", "./tensorflow/core/common_runtime/simple_propagator_state.h", "TaggedNodeReadyQueue");
}

    void push_back(const TaggedNode& node) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsimple_propagator_stateDTh mht_5(mht_5_v, 260, "", "./tensorflow/core/common_runtime/simple_propagator_state.h", "push_back");
 ready_.push_back(node); }
    TaggedNode front() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsimple_propagator_stateDTh mht_6(mht_6_v, 264, "", "./tensorflow/core/common_runtime/simple_propagator_state.h", "front");

      DCHECK_LT(front_index_, ready_.size());
      return ready_[front_index_];
    }
    void pop_front() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsimple_propagator_stateDTh mht_7(mht_7_v, 271, "", "./tensorflow/core/common_runtime/simple_propagator_state.h", "pop_front");

      DCHECK_LT(front_index_, ready_.size());
      front_index_++;
      if ((front_index_ == ready_.size()) || (front_index_ > kSpillThreshold)) {
        if (front_index_ == ready_.size()) {
          ready_.clear();
        } else {
          // Lots of unused entries at beginning of vector: move everything
          // down to start of vector.
          ready_.erase(ready_.begin(), ready_.begin() + front_index_);
        }
        front_index_ = 0;
      }
    }
    bool empty() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsimple_propagator_stateDTh mht_8(mht_8_v, 288, "", "./tensorflow/core/common_runtime/simple_propagator_state.h", "empty");
 return ready_.empty(); }

   private:
    // TODO(b/152925936): Re-evaluate these constants with current usage
    // patterns.
    static constexpr int kSpillThreshold = 16384;
    gtl::InlinedVector<TaggedNode, 16> ready_;
    int front_index_;
  };

  // TODO(b/152925936): Re-evaluate this constant with current usage patterns.
  typedef gtl::InlinedVector<TaggedNode, 8> TaggedNodeSeq;

  // Creates and adds a `TaggedNode` for each node in `roots` to `*ready`.
  void ActivateRoots(gtl::ArraySlice<const NodeItem*> roots,
                     TaggedNodeSeq* ready);

  // After processing the outputs, propagates the outputs to their dsts.
  // Contents of *outputs are left in an indeterminate state after
  // returning from this method.
  void PropagateOutputs(const TaggedNode& tagged_node, EntryVector* outputs,
                        TaggedNodeSeq* ready);

  // Returns an array of `Entry` objects corresponding to the inputs of
  // `tagged_node`.
  Entry* GetInputTensors(const TaggedNode& tagged_node) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsimple_propagator_stateDTh mht_9(mht_9_v, 316, "", "./tensorflow/core/common_runtime/simple_propagator_state.h", "GetInputTensors");

#if defined(THREAD_SANITIZER) || defined(DEBUG)
    // NOTE: This read of `pending_[...]` works around a limitation in TSAN.
    // To avoid false positive data race reports, we need to perform an atomic
    // object access that will establish the happens-before relation between
    // the write to input_tensors_ in `PropagateOutputs()` and the read in
    // `PrepareInputs()`.
    CHECK_EQ(pending_[tagged_node.node_item->node_id], 0);
#endif  // defined(THREAD_SANITIZER) || defined(DEBUG)
    return input_tensors_.data() + tagged_node.node_item->input_start;
  }

  FrameAndIter GetFrameAndIter(const TaggedNode& tagged_node) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsimple_propagator_stateDTh mht_10(mht_10_v, 331, "", "./tensorflow/core/common_runtime/simple_propagator_state.h", "GetFrameAndIter");

    return {0, 0};
  }

  // Provide debugging output of the state of the executor.
  void DumpState();

  // For debugging/logging only.
  void MaybeMarkStarted(const TaggedNode& tagged_node) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsimple_propagator_stateDTh mht_11(mht_11_v, 342, "", "./tensorflow/core/common_runtime/simple_propagator_state.h", "MaybeMarkStarted");

    // TODO(misard) Replace with a finer-grain enabling flag once we add better
    // optional debugging support.
    if (TF_PREDICT_FALSE(vlog_) && VLOG_IS_ON(1)) {
      mutex_lock l(mu_);
      (*active_)[tagged_node.node_item->node_id] = true;
    }
  }
  void MaybeMarkCompleted(const TaggedNode& tagged_node) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSsimple_propagator_stateDTh mht_12(mht_12_v, 353, "", "./tensorflow/core/common_runtime/simple_propagator_state.h", "MaybeMarkCompleted");

    // TODO(misard) Replace with a finer-grain enabling flag once we add better
    // optional debugging support.
    if (TF_PREDICT_FALSE(vlog_) && VLOG_IS_ON(1)) {
      mutex_lock l(mu_);
      (*active_)[tagged_node.node_item->node_id] = false;
    }
  }

 private:
  SimplePropagatorState(const ImmutableExecutorState& immutable_state_,
                        int64_t step_id,
                        const ImmutableExecutorState::FrameInfo& finfo,
                        bool vlog);

  const ImmutableExecutorState& immutable_state_;
  const int64_t step_id_;
  const bool vlog_;

  // The i-th node's j-th input is stored at
  // `input_tensors[impl_->nodes[i].input_start + j]`.
  //
  // NOTE: No need to protect input_tensors[i] by any locks because it
  // is resized once. Each element of input_tensors is written once by the
  // source node of an edge and is cleared by the destination of the same
  // edge. The destination node always runs after the source node, so there
  // is never concurrent access to the same entry.
  std::vector<Entry> input_tensors_;

  std::unique_ptr<std::atomic<int32>[]> pending_;

  // If `vlog_` is true, this stores a bit vector of active nodes, indexed by
  // node ID.
  mutex mu_;
  std::unique_ptr<std::vector<bool>> active_ TF_GUARDED_BY(mu_);

  const std::vector<const NodeItem*>* const nodes_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_SIMPLE_PROPAGATOR_STATE_H_
