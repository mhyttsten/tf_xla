#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_PENDING_COUNTS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_PENDING_COUNTS_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSpending_countsDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpending_countsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSpending_countsDTh() {
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

#include <atomic>

#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/port.h"

namespace tensorflow {

// PendingCounts is an internal helper class to keep track of pending and
// dead counts for nodes, for use in the ExecutorState module.  It
// holds a map from Handles to various counts for that handle.  This
// information is needed per frame iteration. The amount of memory
// needed for an iteration is the same across all executions of the
// iteration. The memory amount and handles are precomputed at startup
// using a Layout object.
//
//    PendingCounts::Layout layout;
//    std::vector<PendingCounts::Handle> h(C);
//    for (int id = 0; id < C; id++) {
//      h[id] = r.AddHandle(max_pending[id], max_dead[id]);
//    }
//
// When we actually want to start an iteration we first create a
// PendingCounts object and then index into it using the precomputed
// handles:

//    PendingCounts counts(layout);
//    ...
//    counts.decrement_pending(h[id], 1);
class PendingCounts {
 public:
  // The state machine for a node's execution.
  enum NodeState {
    // The pending count for the node > 0.
    PENDING_NOTREADY,
    // The pending count for the node == 0, but the node has not
    // started executing.
    PENDING_READY,
    // The node has started executing.
    STARTED,
    // The node has finished executing.
    COMPLETED
  };

  // An opaque handle indicating where in the PendingCounts data structure
  // the appropriate count information can be found.
  class Handle;
  // Given a node that needs to represent counts no larger than the
  // specified "max_pending_count" and "max_dead_count", create a
  // handle that can be passed to various PendingCounts routines
  // to retrieve the count data for this node.
  class Layout {
   public:
    Handle CreateHandle(size_t max_pending_count, size_t max_dead_count);

   private:
    friend class PendingCounts;
    int next_offset_ = 0;  // Next byte offset to allocate
  };

  // Create a new PendingCounts object that can hold the state of
  // all the Handles allocated from "final_allocator".
  explicit PendingCounts(Layout layout)
      : num_bytes_(layout.next_offset_), bytes_(new char[num_bytes_]) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpending_countsDTh mht_0(mht_0_v, 253, "", "./tensorflow/core/common_runtime/pending_counts.h", "PendingCounts");

    if (num_bytes_ >= sizeof(LargeCounts)) {
      CHECK_EQ(uintptr_t(bytes_) % alignof(LargeCounts), 0);
    }
  }

  // Create a new PendingCounts object with the same layout and counts
  // as "other".
  explicit PendingCounts(const PendingCounts& other)
      : num_bytes_(other.num_bytes_), bytes_(new char[num_bytes_]) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpending_countsDTh mht_1(mht_1_v, 265, "", "./tensorflow/core/common_runtime/pending_counts.h", "PendingCounts");

    if (num_bytes_ >= sizeof(LargeCounts)) {
      CHECK_EQ(uintptr_t(bytes_) % alignof(LargeCounts), 0);
    }
    memcpy(bytes_, other.bytes_, other.num_bytes_);
  }

  ~PendingCounts() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpending_countsDTh mht_2(mht_2_v, 275, "", "./tensorflow/core/common_runtime/pending_counts.h", "~PendingCounts");
 delete[] bytes_; }

  void set_initial_count(Handle h, size_t pending_count) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpending_countsDTh mht_3(mht_3_v, 280, "", "./tensorflow/core/common_runtime/pending_counts.h", "set_initial_count");

    if (h.is_large_) {
      std::atomic<LargeCounts>* c_ptr = Large(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      c.pending = pending_count;
      c.dead_count = 0;
      c.has_started = 0;
      c_ptr->store(c, std::memory_order_relaxed);
    } else {
      DCHECK_LE(pending_count, kMaxCountForPackedCounts);
      std::atomic<PackedCounts>* c_ptr = Packed(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      c.pending = pending_count;
      c.dead_count = 0;
      c.has_started = 0;
      c_ptr->store(c, std::memory_order_relaxed);
    }
  }

  NodeState node_state(Handle h) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpending_countsDTh mht_4(mht_4_v, 302, "", "./tensorflow/core/common_runtime/pending_counts.h", "node_state");

    if (h.is_large_) {
      return NodeStateForStruct(Large(h)->load(std::memory_order_relaxed));
    } else {
      return NodeStateForStruct(Packed(h)->load(std::memory_order_relaxed));
    }
  }
  void mark_started(Handle h) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpending_countsDTh mht_5(mht_5_v, 312, "", "./tensorflow/core/common_runtime/pending_counts.h", "mark_started");

    DCHECK_EQ(pending(h), 0);
    if (h.is_large_) {
      std::atomic<LargeCounts>* c_ptr = Large(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      DCHECK_EQ(c.has_started, 0);
      c.has_started = 1;
      c_ptr->store(c, std::memory_order_relaxed);
    } else {
      std::atomic<PackedCounts>* c_ptr = Packed(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      DCHECK_EQ(c.has_started, 0);
      c.has_started = 1;
      c_ptr->store(c, std::memory_order_relaxed);
    }
  }
  void mark_completed(Handle h) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpending_countsDTh mht_6(mht_6_v, 331, "", "./tensorflow/core/common_runtime/pending_counts.h", "mark_completed");

    if (h.is_large_) {
      std::atomic<LargeCounts>* c_ptr = Large(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      DCHECK_EQ(c.has_started, 1);
      c.pending = 1;
      c_ptr->store(c, std::memory_order_relaxed);
    } else {
      std::atomic<PackedCounts>* c_ptr = Packed(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      DCHECK_EQ(c.has_started, 1);
      c.pending = 1;
      c_ptr->store(c, std::memory_order_relaxed);
    }
  }
  int pending(Handle h) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpending_countsDTh mht_7(mht_7_v, 349, "", "./tensorflow/core/common_runtime/pending_counts.h", "pending");

    if (h.is_large_) {
      LargeCounts c = Large(h)->load(std::memory_order_relaxed);
      if (PENDING_NOTREADY == NodeStateForStruct(c)) {
        return c.pending;
      } else {
        // The pending count encodes the state once the node has
        // started, so just return 0.
        return 0;
      }
    } else {
      PackedCounts c = Packed(h)->load(std::memory_order_relaxed);
      if (PENDING_NOTREADY == NodeStateForStruct(c)) {
        return c.pending;
      } else {
        // The pending count encodes the state once the node has
        // started, so just return 0.
        return 0;
      }
    }
  }
  int decrement_pending(Handle h, int v) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpending_countsDTh mht_8(mht_8_v, 373, "", "./tensorflow/core/common_runtime/pending_counts.h", "decrement_pending");

    DCHECK_GE(pending(h), v);
    if (h.is_large_) {
      std::atomic<LargeCounts>* c_ptr = Large(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      c.pending -= v;
      c_ptr->store(c, std::memory_order_relaxed);
      return c.pending;
    } else {
      std::atomic<PackedCounts>* c_ptr = Packed(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      c.pending -= v;
      c_ptr->store(c, std::memory_order_relaxed);
      return c.pending;
    }
  }
  // Mark a merge node as live
  // REQUIRES: Node corresponding to "h" is a merge node
  void mark_live(Handle h) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpending_countsDTh mht_9(mht_9_v, 394, "", "./tensorflow/core/common_runtime/pending_counts.h", "mark_live");

    if (h.is_large_) {
      std::atomic<LargeCounts>* c_ptr = Large(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      // Only do anything if the node hasn't already started executing.
      if (PENDING_NOTREADY == NodeStateForStruct(c)) {
        c.pending &= ~static_cast<int>(0x1);
        c_ptr->store(c, std::memory_order_relaxed);
      }
    } else {
      std::atomic<PackedCounts>* c_ptr = Packed(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      // Only do anything if the node hasn't already started executing.
      if (PENDING_NOTREADY == NodeStateForStruct(c)) {
        static_assert(7 == kMaxCountForPackedCounts,
                      "Live flag incorrect for max packed count");
        c.pending &= 0x6;
        c_ptr->store(c, std::memory_order_relaxed);
      }
    }
  }

  int dead_count(Handle h) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpending_countsDTh mht_10(mht_10_v, 419, "", "./tensorflow/core/common_runtime/pending_counts.h", "dead_count");

    int r = h.is_large_ ? Large(h)->load(std::memory_order_relaxed).dead_count
                        : Packed(h)->load(std::memory_order_relaxed).dead_count;
    return r;
  }
  void increment_dead_count(Handle h) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpending_countsDTh mht_11(mht_11_v, 427, "", "./tensorflow/core/common_runtime/pending_counts.h", "increment_dead_count");

    if (h.is_large_) {
      std::atomic<LargeCounts>* c_ptr = Large(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      if (PENDING_NOTREADY == NodeStateForStruct(c)) {
        c.dead_count++;
        c_ptr->store(c, std::memory_order_relaxed);
      }
    } else {
      std::atomic<PackedCounts>* c_ptr = Packed(h);
      auto c = c_ptr->load(std::memory_order_relaxed);
      if (PENDING_NOTREADY == NodeStateForStruct(c)) {
        DCHECK_LT(c.dead_count, kMaxCountForPackedCounts);
        c.dead_count++;
        c_ptr->store(c, std::memory_order_relaxed);
      }
    }
  }

  struct AdjustResult {
    bool any_dead;
    bool any_pending;

    AdjustResult(bool any_dead, bool any_pending)
        : any_dead(any_dead), any_pending(any_pending) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpending_countsDTh mht_12(mht_12_v, 454, "", "./tensorflow/core/common_runtime/pending_counts.h", "AdjustResult");
}
  };

  // A streamlined routine that does several pieces of bookkeeping at
  // once.  Equivalent to:
  //    if (increment_dead) increment_dead_count(h);
  //    decrement_pending(h, 1);
  //    return {dead_count(h) > 0, pending(h) > 0};
  AdjustResult adjust_for_activation(Handle h, bool increment_dead) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpending_countsDTh mht_13(mht_13_v, 465, "", "./tensorflow/core/common_runtime/pending_counts.h", "adjust_for_activation");

    DCHECK_GE(pending(h), 1);
    if (h.is_large_) {
      return adjust_for_activation_shared(Large(h), increment_dead);
    } else {
      return adjust_for_activation_shared(Packed(h), increment_dead);
    }
  }

  // The same as the above, but performs the operation atomically. This
  // is thread-safe to run concurrently with other threads.
  AdjustResult adjust_for_activation_atomic(Handle h, bool increment_dead) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpending_countsDTh mht_14(mht_14_v, 479, "", "./tensorflow/core/common_runtime/pending_counts.h", "adjust_for_activation_atomic");

    DCHECK_GE(pending(h), 1);
    if (h.is_large_) {
      return adjust_for_activation_shared_atomic(Large(h), increment_dead);
    } else {
      return adjust_for_activation_shared_atomic(Packed(h), increment_dead);
    }
  }

  class Handle {
   public:
    Handle() : byte_offset_(0), is_large_(0) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpending_countsDTh mht_15(mht_15_v, 493, "", "./tensorflow/core/common_runtime/pending_counts.h", "Handle");
}

   private:
    friend class PendingCounts;
    int byte_offset_ : 31;  // Byte offset of the rep in PendingCounts object
    bool is_large_ : 1;  // If true, rep is LargeCounts; otherwise PackedCounts
  };

 private:
  template <typename T>
  inline AdjustResult adjust_for_activation_shared(std::atomic<T>* c,
                                                   bool increment_dead) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpending_countsDTh mht_16(mht_16_v, 507, "", "./tensorflow/core/common_runtime/pending_counts.h", "adjust_for_activation_shared");

    T val = c->load(std::memory_order_relaxed);
    if (increment_dead && PENDING_NOTREADY == NodeStateForStruct(val)) {
      val.dead_count++;
    }
    val.pending--;
    c->store(val, std::memory_order_relaxed);
    return AdjustResult(val.dead_count, val.pending);
  }

  template <typename T>
  inline AdjustResult adjust_for_activation_shared_atomic(std::atomic<T>* c,
                                                          bool increment_dead) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpending_countsDTh mht_17(mht_17_v, 522, "", "./tensorflow/core/common_runtime/pending_counts.h", "adjust_for_activation_shared_atomic");

    T old_val = c->load(std::memory_order_relaxed);
    while (true) {
      T new_val = old_val;
      if (increment_dead && PENDING_NOTREADY == NodeStateForStruct(new_val)) {
        new_val.dead_count++;
      }
      new_val.pending--;
      AdjustResult ret(new_val.dead_count, new_val.pending);
      if (TF_PREDICT_TRUE(c->compare_exchange_weak(old_val, new_val)))
        return ret;
    }
  }

  // We keep track of the pending count and dead input count for each
  // graph node.  The representation used here is designed to be cache
  // efficient for graphs with large numbers of nodes, where most
  // nodes have relatively small maximum pending counts (e.g. for one
  // LSTM model, 99% of 5000+ nodes had in-degrees of 3 or less).  We
  // use one byte to hold both the pending and dead count for a node
  // where these together can fit in one byte, and we use a hash table
  // to handle the rare node ids that need larger counts than this.
  // Each frame in this subgraph has its own PendingCounts.

  // We use 3 bits each for dead_count and pending.
  static constexpr int kMaxCountForPackedCounts = 7;

  // Most counts are small, so we pack a pending count and a dead
  // count into 3 bits each, use 1 bit to indicate that the node has
  // started computing.
  struct PackedCounts {
    uint8 pending : 3;
    uint8 dead_count : 3;
    uint8 has_started : 1;
  };

  // NOTE: alignas(8) is critical to implement efficient atomic<LargeCounts>
  // on MSVC.
  struct alignas(8) LargeCounts {
    uint32 pending;
    uint32 dead_count : 31;
    // NOTE(tlipcon): MSVC won't pack this struct into 8 bytes unless
    // all of the member types are uint32.
    uint32 has_started : 1;
  };

  template <typename T>
  NodeState NodeStateForStruct(const T& c) const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpending_countsDTh mht_18(mht_18_v, 572, "", "./tensorflow/core/common_runtime/pending_counts.h", "NodeStateForStruct");

    if (c.has_started) {
      return (c.pending == 0) ? STARTED : COMPLETED;
    } else {
      return (c.pending == 0) ? PENDING_READY : PENDING_NOTREADY;
    }
  }
  inline std::atomic<LargeCounts>* Large(Handle h) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpending_countsDTh mht_19(mht_19_v, 582, "", "./tensorflow/core/common_runtime/pending_counts.h", "Large");

    DCHECK(h.is_large_);
    DCHECK_LE(h.byte_offset_ + sizeof(std::atomic<LargeCounts>), num_bytes_);
    DCHECK_EQ(h.byte_offset_ % alignof(std::atomic<LargeCounts>), 0);
    return reinterpret_cast<std::atomic<LargeCounts>*>(bytes_ + h.byte_offset_);
  }
  inline std::atomic<PackedCounts>* Packed(Handle h) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpending_countsDTh mht_20(mht_20_v, 591, "", "./tensorflow/core/common_runtime/pending_counts.h", "Packed");

    DCHECK(!h.is_large_);
    DCHECK_LE(h.byte_offset_ + sizeof(PackedCounts), num_bytes_);
    return reinterpret_cast<std::atomic<PackedCounts>*>(bytes_ +
                                                        h.byte_offset_);
  }

  const int num_bytes_;  // Just for bounds checking in debug mode
  char* bytes_;          // Array of num_bytes_ bytes

  void operator=(const PendingCounts&) = delete;
};

inline PendingCounts::Handle PendingCounts::Layout::CreateHandle(
    size_t max_pending_count, size_t max_dead_count) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpending_countsDTh mht_21(mht_21_v, 608, "", "./tensorflow/core/common_runtime/pending_counts.h", "PendingCounts::Layout::CreateHandle");

  Handle result;
  if ((max_pending_count > kMaxCountForPackedCounts) ||
      (max_dead_count > kMaxCountForPackedCounts)) {
    constexpr int B = sizeof(std::atomic<LargeCounts>);
    // Round byte offset to proper alignment
    static_assert(
        sizeof(std::atomic<LargeCounts>) >= alignof(std::atomic<LargeCounts>),
        "std::atomic<LargeCounts> must be packed");
    int64_t offset = ((static_cast<int64_t>(next_offset_) + B - 1) / B) * B;
    result.byte_offset_ = offset;
    result.is_large_ = true;
    next_offset_ = result.byte_offset_ + B;
  } else {
    result.byte_offset_ = next_offset_;
    result.is_large_ = false;
    static_assert(sizeof(std::atomic<PackedCounts>) == 1,
                  "std::atomic<PackedCounts> should be a single byte");
    next_offset_ += sizeof(std::atomic<PackedCounts>);
  }
  return result;
}

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_PENDING_COUNTS_H_
