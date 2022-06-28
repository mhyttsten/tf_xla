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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_PROPAGATOR_STATE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_PROPAGATOR_STATE_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh() {
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


#include <queue>
#include <vector>

#include "tensorflow/core/common_runtime/entry.h"
#include "tensorflow/core/common_runtime/immutable_executor_state.h"
#include "tensorflow/core/common_runtime/pending_counts.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/control_flow.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef gtl::InlinedVector<AllocatorAttributes, 4> AllocatorAttributeVec;

// Represents the ephemeral "edge state" associated with one invocation of
// `Executor::Run()`.
//
// `PropagatorState` is responsible for propagating values along dataflow
// edges in a TensorFlow graph and determining which nodes are runnable. The
// executor primarily updates `PropagatorState` by calling `PropagateOutputs()`
// after processing a node, and `PropagatorState` dispatches `TaggedNode`s by
// adding them to a `TaggedNodeSeq`.
class PropagatorState {
 public:
  PropagatorState(const ImmutableExecutorState& immutable_state,
                  int64_t step_id, bool vlog);
  ~PropagatorState();

 private:
  // Forward declaration so that `TaggedNode` can include a `FrameState*` and an
  // `IterationState*`.
  struct FrameState;
  struct IterationState;

 public:
  // A `TaggedNode` corresponds to a single invocation of a node's kernel,
  // and it is created when the kernel becomes runnable (in a particular
  // iteration of a particular frame).
  struct TaggedNode {
    const NodeItem* node_item;
    FrameState* input_frame;
    IterationState* input_iter;
    bool is_dead;

    TaggedNode() = default;
    TaggedNode(const NodeItem* node_item, FrameState* in_frame,
               IterationState* in_iter, bool dead)
        : node_item(node_item),
          input_frame(in_frame),
          input_iter(in_iter),
          is_dead(dead) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_0(mht_0_v, 244, "", "./tensorflow/core/common_runtime/propagator_state.h", "TaggedNode");
}

    const NodeItem& get_node_item() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_1(mht_1_v, 249, "", "./tensorflow/core/common_runtime/propagator_state.h", "get_node_item");
 return *node_item; }

    bool get_is_dead() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_2(mht_2_v, 254, "", "./tensorflow/core/common_runtime/propagator_state.h", "get_is_dead");
 return is_dead; }
    int64_t get_iter_num() const;
  };

  // A drop-in replacement for std::deque<TaggedNode>.  We typically don't
  // have that many nodes in the ready queue, so we just use a vector and
  // don't free up memory from the queue as we consume nodes.
  class TaggedNodeReadyQueue {
   public:
    TaggedNodeReadyQueue() : front_index_(0) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_3(mht_3_v, 266, "", "./tensorflow/core/common_runtime/propagator_state.h", "TaggedNodeReadyQueue");
}

    void push_back(const TaggedNode& node) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_4(mht_4_v, 271, "", "./tensorflow/core/common_runtime/propagator_state.h", "push_back");
 ready_.push_back(node); }
    TaggedNode front() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_5(mht_5_v, 275, "", "./tensorflow/core/common_runtime/propagator_state.h", "front");

      DCHECK_LT(front_index_, ready_.size());
      return ready_[front_index_];
    }
    void pop_front() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_6(mht_6_v, 282, "", "./tensorflow/core/common_runtime/propagator_state.h", "pop_front");

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
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_7(mht_7_v, 299, "", "./tensorflow/core/common_runtime/propagator_state.h", "empty");
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

 private:
  // The state of an iteration in a particular frame.
  struct IterationState {
    explicit IterationState(int64_t iter_num,
                            const PendingCounts* pending_counts,
                            int total_input_tensors)
        : iter_num(iter_num),
          input_tensors(new Entry[total_input_tensors]),
          outstanding_ops(0),
          outstanding_frame_count(0),
          counts(*pending_counts) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_8(mht_8_v, 325, "", "./tensorflow/core/common_runtime/propagator_state.h", "IterationState");
  // Initialize with copy of *pending_counts
    }

    const int64_t
        iter_num;  // The index of this iteration in the enclosing loop.

    // One copy per iteration. For iteration k, i-th node's j-th input is in
    // input_tensors[k][immutable_state_.nodes[i].input_start + j]. An entry is
    // either a tensor pointer (pass-by-reference) or a tensor (pass-by-value).
    //
    // NOTE: No need to protect input_tensors[i] by any locks because it
    // is resized once. Each element of tensors_ is written once by the
    // source node of an edge and is cleared by the destination of the same
    // edge. The latter node is never run concurrently with the former node.
    Entry* input_tensors;

    // The number of outstanding ops for each iteration.
    std::atomic<size_t> outstanding_ops;

    // The number of outstanding frames for each iteration.
    int outstanding_frame_count;
    int pending(PendingCounts::Handle h) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_9(mht_9_v, 349, "", "./tensorflow/core/common_runtime/propagator_state.h", "pending");
 return counts.pending(h); }
    int decrement_pending(PendingCounts::Handle h, int v) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_10(mht_10_v, 353, "", "./tensorflow/core/common_runtime/propagator_state.h", "decrement_pending");

      return counts.decrement_pending(h, v);
    }
    // Mark a merge node as live
    // REQUIRES: Node corresponding to "h" is a merge node
    void mark_live(PendingCounts::Handle h) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_11(mht_11_v, 361, "", "./tensorflow/core/common_runtime/propagator_state.h", "mark_live");
 counts.mark_live(h); }
    // Mark a node to show that processing has started.
    void mark_started(PendingCounts::Handle h) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_12(mht_12_v, 366, "", "./tensorflow/core/common_runtime/propagator_state.h", "mark_started");
 counts.mark_started(h); }
    // Mark a node to show that processing has completed.
    void mark_completed(PendingCounts::Handle h) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_13(mht_13_v, 371, "", "./tensorflow/core/common_runtime/propagator_state.h", "mark_completed");
 counts.mark_completed(h); }
    PendingCounts::NodeState node_state(PendingCounts::Handle h) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_14(mht_14_v, 375, "", "./tensorflow/core/common_runtime/propagator_state.h", "node_state");

      return counts.node_state(h);
    }

    int dead_count(PendingCounts::Handle h) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_15(mht_15_v, 382, "", "./tensorflow/core/common_runtime/propagator_state.h", "dead_count");
 return counts.dead_count(h); }
    void increment_dead_count(PendingCounts::Handle h) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_16(mht_16_v, 386, "", "./tensorflow/core/common_runtime/propagator_state.h", "increment_dead_count");

      counts.increment_dead_count(h);
    }
    PendingCounts::AdjustResult adjust_for_activation(PendingCounts::Handle h,
                                                      bool increment_dead) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_17(mht_17_v, 393, "", "./tensorflow/core/common_runtime/propagator_state.h", "adjust_for_activation");

      return counts.adjust_for_activation(h, increment_dead);
    }
    PendingCounts::AdjustResult adjust_for_activation_atomic(
        PendingCounts::Handle h, bool increment_dead) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_18(mht_18_v, 400, "", "./tensorflow/core/common_runtime/propagator_state.h", "adjust_for_activation_atomic");

      return counts.adjust_for_activation_atomic(h, increment_dead);
    }

    ~IterationState() {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_19(mht_19_v, 407, "", "./tensorflow/core/common_runtime/propagator_state.h", "~IterationState");
 delete[] input_tensors; }

   private:
    PendingCounts counts;
  };

  struct FrameState {
    explicit FrameState(const ImmutableExecutorState& immutable_state,
                        int parallel_iters)
        : immutable_state(immutable_state),
          max_parallel_iterations(parallel_iters),
          num_outstanding_iterations(1),
          iterations(parallel_iters + 1),
          iterations_raw(iterations.data()) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_20(mht_20_v, 423, "", "./tensorflow/core/common_runtime/propagator_state.h", "FrameState");
}

    // A new frame is created for each loop. Execution starts at iteration 0.
    // When a value at iteration 0 passes through a NextIteration node,
    // iteration 1 is created and starts running. Note that iteration 0 may
    // still be running so multiple iterations may run in parallel. The
    // frame maintains the state of iterations in several data structures
    // such as pending_count and input_tensors. When iteration 0 completes,
    // we garbage collect the state of iteration 0.
    //
    // A frame instance is considered "done" and can be garbage collected
    // if all its inputs have entered and all its iterations are "done".
    //
    // A frame manages the live iterations of an iterative computation.
    // Iteration i is considered "done" when there are no outstanding ops,
    // frames at iteration i are done, all recvs for this iteration are
    // completed, and iteration i-1 is done. For iteration 0, we instead
    // wait for there to be no more pending inputs of the frame.
    //
    // Frames and iterations are garbage collected once they are done.
    // The state we need to keep around is highly dependent on the
    // parallelism enabled by the scheduler. We may want to have the
    // scheduler dynamically control the outstanding number of live
    // parallel frames and iterations. To reduce the state space, the
    // scheduler might want to schedule ops in inner frames first and
    // lower iterations first.
    //
    // This frame state is mostly initialized lazily on demand so we
    // don't introduce unnecessary overhead.

    // The immutable state of the executor the frame is in.
    const ImmutableExecutorState& immutable_state;

    // The name of this frame, which is the concatenation of its parent
    // frame name, the iteration of the parent frame when this frame was
    // created, and the value of the attr 'frame_name'.
    string frame_name;

    // The unique id for this frame. Generated by fingerprinting
    // frame_name.
    uint64 frame_id;

    // The iteration state of its parent frame when this frame is created.
    // nullptr if there is no parent frame. The frame_name/parent_iter pair
    // uniquely identifies this FrameState.
    IterationState* parent_iter = nullptr;

    // The FrameState of its parent frame.
    FrameState* parent_frame = nullptr;

    // The maximum allowed number of parallel iterations.
    const int max_parallel_iterations;

    // The number of inputs this frame is still waiting.
    int num_pending_inputs = 0;

    // The highest iteration number we have reached so far in this frame.
    int64_t iteration_count TF_GUARDED_BY(mu) = 0;

    // The number of outstanding iterations.
    int num_outstanding_iterations TF_GUARDED_BY(mu) = 1;

   private:
    // The active iteration states of this frame.
    gtl::InlinedVector<IterationState*, 12> iterations;
    IterationState** const iterations_raw TF_GUARDED_BY(mu);
    IterationState* iterations_first TF_GUARDED_BY(mu);

   public:
    // The NextIteration nodes to enter a new iteration. If the number of
    // outstanding iterations reaches the limit, we will defer the start of
    // the next iteration until the number of outstanding iterations falls
    // below the limit.
    std::vector<std::pair<const NodeItem*, Entry>> next_iter_roots
        TF_GUARDED_BY(mu);

    // The values of the loop invariants for this loop. They are added into
    // this list as they "enter" the frame. When a loop invariant enters,
    // we make it available to all active iterations. When the frame starts
    // a new iteration, we make all the current loop invariants available
    // to the new iteration.
    std::vector<std::pair<const NodeItem*, Entry>> inv_values TF_GUARDED_BY(mu);

    // The list of dead exit node items for the current highest iteration. We
    // will only "execute" the dead exits of the final iteration.
    std::vector<const NodeItem*> dead_exits TF_GUARDED_BY(mu);

    // Static information specific to this frame.
    PendingCounts* pending_counts = nullptr;
    int total_input_tensors = 0;
    std::vector<const NodeItem*>* nodes = nullptr;

    // Lock ordering: ExecutorState.mu_ < mu;
    // during structured traversal: parent_frame->mu < mu.
    mutex mu;

    void InitializeFrameInfo(const ImmutableExecutorState::FrameInfo& finfo);

    inline IterationState* GetIteration(int64_t iter)
        TF_SHARED_LOCKS_REQUIRED(mu) {
      if (TF_PREDICT_TRUE(iter == 0)) {
        return iterations_first;
      } else {
        size_t index = iter % (max_parallel_iterations + 1);
        return iterations_raw[index];
      }
    }

    void SetIteration(int64_t iter, IterationState* state);

    // Adjust the outstanding op count by 'delta' and clean up the iterations in
    // the frame if no more ops are oustanding. Return true iff the execution of
    // the frame is done.
    //
    // Avoids acquiring the lock in the common case that the frame is not done.
    bool AdjustOutstandingOps(IterationState* iter_state, int delta,
                              TaggedNodeSeq* ready);

    bool AdjustOutstandingOpsLocked(IterationState* iter_state, int delta,
                                    TaggedNodeSeq* ready)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu);

    bool AdjustOutstandingOpsFastPath(IterationState* iter_state, int delta)
        TF_SHARED_LOCKS_REQUIRED(mu);

    // Convenience methods for the above 'Adjust' calls where delta takes the
    // common value of -1.
    bool DecrementOutstandingOps(IterationState* iter_state,
                                 TaggedNodeSeq* ready);

    bool DecrementOutstandingOpsLocked(IterationState* iter_state,
                                       TaggedNodeSeq* ready);

    // Returns true if the computation in the frame is completed.
    bool IsFrameDone();

    // Returns true if the iteration of the frame is completed.
    bool IsIterationDone(IterationState* iter_state)
        TF_SHARED_LOCKS_REQUIRED(mu);

    // Increments the iteration id. If this is a new iteration, initialize it.
    //
    // Returns a pointer to the new iteration.
    IterationState* IncrementIteration(TaggedNodeSeq* ready)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu);

    // Activate all the deferred NextIteration nodes in a new iteration.
    void ActivateNexts(IterationState* iter_state, TaggedNodeSeq* ready)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu);

    // Activate all the current loop invariants in a new iteration.
    void ActivateLoopInvs(IterationState* iter_state, TaggedNodeSeq* ready)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu);

    // Add a new loop invariant and make it available to all active
    // iterations.
    void AddLoopInv(const NodeItem* item, const Entry& entry,
                    TaggedNodeSeq* ready) TF_EXCLUSIVE_LOCKS_REQUIRED(mu);

    // Activate the successors of a node. Contents of *outputs are left in an
    // indeterminate state after returning from this method.
    //
    // In the case that 'item' is a simple node (no merge/control outputs) this
    // will acquire a shared lock and can run concurrently with other
    // invocations.
    //
    // Return true if the frame is done after activation.
    bool ActivateNodesAndAdjustOutstanding(const NodeItem* item,
                                           const bool is_dead,
                                           IterationState* iter_state,
                                           EntryVector* outputs,
                                           TaggedNodeSeq* ready);

    // Same as the above, but requires 'mu' already held in exclusive mode.
    int ActivateNodesLocked(const NodeItem* item, const bool is_dead,
                            IterationState* iter_state, EntryVector* outputs,
                            TaggedNodeSeq* ready)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu);

    // Cleanup iterations of this frame starting from the given iteration.
    bool CleanupIterations(IterationState* iter_state, TaggedNodeSeq* ready)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu);

    void DumpIterationState(PropagatorState* parent) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_21(mht_21_v, 609, "", "./tensorflow/core/common_runtime/propagator_state.h", "DumpIterationState");

      mutex_lock l(mu);
      for (IterationState* iteration : iterations) {
        if (iteration) {
          LOG(WARNING) << "  Iteration:";
          parent->DumpIterationState(this, iteration);
        }
      }
    }

    ~FrameState() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_22(mht_22_v, 622, "", "./tensorflow/core/common_runtime/propagator_state.h", "~FrameState");

      for (size_t i = 0; i < iterations.size(); ++i) {
        delete iterations[i];
        iterations[i] = nullptr;
      }
    }

   private:
    // REQUIRES: `!item->is_any_consumer_merge_or_control_trigger`.
    // This variant does not use atomic operations to modify the pending counts
    // and thus must hold the exclusive lock.
    int ActivateNodesFastPathLocked(const NodeItem* item, const bool is_dead,
                                    IterationState* iter_state,
                                    EntryVector* outputs, TaggedNodeSeq* ready)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_23(mht_23_v, 639, "", "./tensorflow/core/common_runtime/propagator_state.h", "ActivateNodesFastPathLocked");

      return ActivateNodesFastPathInternal<false>(item, is_dead, iter_state,
                                                  outputs, ready);
    }

    // REQUIRES: `!item->is_any_consumer_merge_or_control_trigger`.
    // This variant uses atomic operations to modify the pending counts.
    int ActivateNodesFastPathShared(const NodeItem* item, const bool is_dead,
                                    IterationState* iter_state,
                                    EntryVector* outputs, TaggedNodeSeq* ready)
        TF_SHARED_LOCKS_REQUIRED(mu) {
      return ActivateNodesFastPathInternal<true>(item, is_dead, iter_state,
                                                 outputs, ready);
    }

    template <bool atomic>
    int ActivateNodesFastPathInternal(const NodeItem* item, const bool is_dead,
                                      IterationState* iter_state,
                                      EntryVector* outputs,
                                      TaggedNodeSeq* ready);

    int ActivateNodesSlowPath(const NodeItem* item, const bool is_dead,
                              IterationState* iter_state, EntryVector* outputs,
                              TaggedNodeSeq* ready)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu);
  };

 public:
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
  //
  // NOTE: Thread safety analysis is disabled on this method, because the
  // underlying `IterationState` and its array of `input_tensors` retain the
  // same address while the iteration is live.
  Entry* GetInputTensors(const TaggedNode& tagged_node) const
      TF_NO_THREAD_SAFETY_ANALYSIS {
    return tagged_node.input_iter->input_tensors +
           tagged_node.node_item->input_start;
  }

  FrameAndIter GetFrameAndIter(const TaggedNode& tagged_node) const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_24(mht_24_v, 692, "", "./tensorflow/core/common_runtime/propagator_state.h", "GetFrameAndIter");

    return {tagged_node.input_frame->frame_id,
            tagged_node.input_iter->iter_num};
  }

  // Provide debugging output of the state of the executor.
  void DumpState();

  // For debugging/logging only.
  void MaybeMarkStarted(const TaggedNode& tagged_node) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_25(mht_25_v, 704, "", "./tensorflow/core/common_runtime/propagator_state.h", "MaybeMarkStarted");

    // TODO(misard) Replace with a finer-grain enabling flag once we add better
    // optional debugging support.
    if (TF_PREDICT_FALSE(vlog_) && VLOG_IS_ON(1)) {
      mutex_lock l(tagged_node.input_frame->mu);
      tagged_node.input_iter->mark_started(
          immutable_state_.pending_ids()[tagged_node.node_item->node_id]);
    }
  }

  void MaybeMarkCompleted(const TaggedNode& tagged_node) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_26(mht_26_v, 717, "", "./tensorflow/core/common_runtime/propagator_state.h", "MaybeMarkCompleted");

    // TODO(misard) Replace with a finer-grain enabling flag once we add better
    // optional debugging support.
    if (TF_PREDICT_FALSE(vlog_) && VLOG_IS_ON(1)) {
      mutex_lock l(tagged_node.input_frame->mu);
      tagged_node.input_iter->mark_completed(
          immutable_state_.pending_ids()[tagged_node.node_item->node_id]);
    }
  }

 private:
  // Find an existing or create a new child frame in the frame 'frame' at
  // iteration 'iter'.
  void FindOrCreateChildFrame(FrameState* frame, IterationState* iter_state,
                              const NodeItem& node_item, FrameState** child);

  // Delete a frame. Called when the frame is done.
  void DeleteFrame(FrameState* frame, TaggedNodeSeq* ready);

  // Cleanup frames and iterations starting from frame/iter. Called when
  // a child frame is done.
  void CleanupFramesIterations(FrameState* frame, IterationState* iter_state,
                               TaggedNodeSeq* ready);

  // Provide debugging output about an outstanding iteration in the executor.
  void DumpIterationState(const FrameState* frame, IterationState* iteration);

  const ImmutableExecutorState& immutable_state_;
  const int64_t step_id_;
  const bool vlog_;

  mutex mu_;

  // The root frame in which the execution of this step is started.
  FrameState* root_frame_;

  // Mapping from frame ID to outstanding frames. A new frame is created
  // at some iteration of an active frame. So the unique key for the new
  // child frame is a hash composed of the ID of the parent frame, the iteration
  // number at which the parent frame is creating the new frame, and the
  // name of the new frame from nodedef.
  absl::flat_hash_map<uint64, FrameState*> outstanding_frames_
      TF_GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(PropagatorState);
};

inline int64_t PropagatorState::TaggedNode::get_iter_num() const {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_27(mht_27_v, 767, "", "./tensorflow/core/common_runtime/propagator_state.h", "PropagatorState::TaggedNode::get_iter_num");

  return input_iter->iter_num;
}

// `OrderedPropagatorState` replaces `PropagatorState`s `TaggedNodeReadyQueue`
// with a priority queue. This ensures that the order in which we dequeue
// `TaggedNode&`s is stable with respect to ASLR.
//
// This is not always needed, as in a multithreaded environment, executions are
// expected to happen nondeterministically, but this nondeteminism can be a
// problem: For example, In usecases that are running close to the RAM limit of
// a device, reordering ops can cause an increase in memory fragmenenation,
// causing an OOM.
// This codepath is enabled using TF_DETERMINISTIC_ORDER=1 in executor.cc
class OrderedPropagatorState : public PropagatorState {
  using PropagatorState::PropagatorState;

 public:
  class TaggedNodeReadyQueue : PropagatorState::TaggedNodeReadyQueue {
   public:
    TaggedNodeReadyQueue() : readyp_(compare) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_28(mht_28_v, 790, "", "./tensorflow/core/common_runtime/propagator_state.h", "TaggedNodeReadyQueue");
}
    void push_back(const TaggedNode& node) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_29(mht_29_v, 794, "", "./tensorflow/core/common_runtime/propagator_state.h", "push_back");
 readyp_.push(node); }
    TaggedNode front() const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_30(mht_30_v, 798, "", "./tensorflow/core/common_runtime/propagator_state.h", "front");
 return readyp_.top(); }
    void pop_front() {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_31(mht_31_v, 802, "", "./tensorflow/core/common_runtime/propagator_state.h", "pop_front");
 readyp_.pop(); }
    bool empty() const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_32(mht_32_v, 806, "", "./tensorflow/core/common_runtime/propagator_state.h", "empty");
 return readyp_.empty(); }

   private:
    static bool compare(TaggedNode const& lhs, TaggedNode const& rhs) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpropagator_stateDTh mht_33(mht_33_v, 812, "", "./tensorflow/core/common_runtime/propagator_state.h", "compare");

      std::tuple<int, uint64, int64_t> lhs_prio{lhs.node_item->node_id,
                                                lhs.input_frame->frame_id,
                                                lhs.input_iter->iter_num};
      std::tuple<int, uint64, int64_t> rhs_prio{rhs.node_item->node_id,
                                                rhs.input_frame->frame_id,
                                                rhs.input_iter->iter_num};
      return lhs_prio < rhs_prio;
    }

    std::priority_queue<TaggedNode, std::vector<TaggedNode>, decltype(&compare)>
        readyp_;
  };
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_PROPAGATOR_STATE_H_
