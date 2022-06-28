/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EAGER_EXECUTOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EAGER_EXECUTOR_H_
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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTh {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTh() {
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


#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

class AsyncEagerNode;
class AsyncRemoteExecuteNode;
namespace eager {
class EagerClient;
}

// A unit of execution for the EagerExecutor class below. Example subclasses
// encapsulate execution of a TFE_Op, or copying a TFE_TensorHandle from one
// device to another.
class EagerNode {
 public:
  EagerNode() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTh mht_0(mht_0_v, 221, "", "./tensorflow/core/common_runtime/eager/eager_executor.h", "EagerNode");
}

  virtual ~EagerNode() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTh mht_1(mht_1_v, 226, "", "./tensorflow/core/common_runtime/eager/eager_executor.h", "~EagerNode");
}

  // Prepares the node when adding it into EagerExecutor. If any errors happens,
  // EagerExecutor will abort the node immediately.
  virtual Status Prepare() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTh mht_2(mht_2_v, 233, "", "./tensorflow/core/common_runtime/eager/eager_executor.h", "Prepare");
 return Status::OK(); }

  // Runs the computation corresponding to this node and blocks till the
  // execution is done.
  virtual Status Run() = 0;

  // Called when this node will not be run due to some error contained in
  // `status`. `status` must not be OK.
  // For example, if the node would have computed some tensors in the Run(),
  // it should poison the corresponding tensor handles in this method.
  virtual void Abort(Status status) = 0;

  // Returns nullptr iff this Eager node is synchronous.
  virtual AsyncEagerNode* AsAsync() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTh mht_3(mht_3_v, 249, "", "./tensorflow/core/common_runtime/eager/eager_executor.h", "AsAsync");
 return nullptr; }
  virtual AsyncRemoteExecuteNode* AsAsyncRemoteExecuteNode() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTh mht_4(mht_4_v, 253, "", "./tensorflow/core/common_runtime/eager/eager_executor.h", "AsAsyncRemoteExecuteNode");
 return nullptr; }

  virtual string DebugString() const = 0;

  // Indicates whether a node failure should make the executor unusable.
  virtual bool Fatal() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTh mht_5(mht_5_v, 261, "", "./tensorflow/core/common_runtime/eager/eager_executor.h", "Fatal");
 return true; }
};

class AsyncEagerNode : public EagerNode {
 public:
  using EagerNode::EagerNode;  // Lift EagerNode constructors.

  // This node will be cleaned up once the done callback is called.
  virtual void RunAsync(StatusCallback done) = 0;

  AsyncEagerNode* AsAsync() final {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTh mht_6(mht_6_v, 274, "", "./tensorflow/core/common_runtime/eager/eager_executor.h", "AsAsync");
 return this; }

  Status Run() final {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTh mht_7(mht_7_v, 279, "", "./tensorflow/core/common_runtime/eager/eager_executor.h", "Run");

    return errors::Unimplemented("Don't call AsyncEagerNode::Run().");
  }
};

class AsyncRemoteExecuteNode : public AsyncEagerNode {
 public:
  AsyncRemoteExecuteNode* AsAsyncRemoteExecuteNode() final {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTh mht_8(mht_8_v, 289, "", "./tensorflow/core/common_runtime/eager/eager_executor.h", "AsAsyncRemoteExecuteNode");
 return this; }

  virtual const eager::EagerClient* eager_client() const = 0;
  virtual bool needs_remote_inputs() const = 0;
  virtual bool allow_multiple_pending_requests() const = 0;
  virtual Status SyncExecutors() = 0;
};

// A class for handling async execution (see TFE_ContextSetAsync).
// Note that this class is thread-safe.
// TODO(agarwal): TFE_OpAddInput may currently block if it tries to access the
// device of the input handle. Fix that.
// TODO(agarwal): Implement support for control dependencies.
// TODO(agarwal): Support out-of-order execution and dispatching multiple
// EagerNode in parallel.
// TODO(agarwal): Implement optimizations over EagerNode traces.
class EagerExecutor {
 public:
  explicit EagerExecutor(bool async);

  ~EagerExecutor();

  // Puts this in a shutdown state. In this state, AddOrExecute() will return an
  // error and not add new EagerNodes. After putting this in the shutdown state,
  // blocks until all pendings nodes have finished running.
  // Returns the status of executing pending nodes.
  // If async was not enabled, aborts and destroys all pending nodes.
  Status ShutDown();

  bool Async() const;

  // Inline execute node if executor is in sync mode.
  Status SyncExecute(EagerNode* node);

  // - Async Mode: schedules `node` for execution.
  // - Sync Mode: inline execute the 'node' directly.
  // If an error occurs (e.g. EagerExecutor has already been shut down), the
  // `node` is not added to this executor and its Abort() method is called.
  Status AddOrExecute(std::unique_ptr<EagerNode> node);

  // Blocks till all currently pending ops are done.
  // In particular, if EnableAsync() has not beed called, it will not return
  // until that happens (and pendings, at the time of call, nodes finish
  // running). If this executor has already been shut down, its final status is
  // returned.
  Status WaitForAllPendingNodes();

  // Clears all currently set errors which re-enables async execution.
  void ClearError();

  // Returns Status based on any errors that occurred during async execution.
  Status status() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTh mht_9(mht_9_v, 343, "", "./tensorflow/core/common_runtime/eager/eager_executor.h", "status");

    if (ok()) return Status::OK();

    tf_shared_lock l(node_queue_mutex_);
    return status_;
  }

  bool ok() const TF_NO_THREAD_SAFETY_ANALYSIS { return ok_; }

  // On destruction, runs `callback`. Used by the EagerContext for clearing
  // thread-local executors.
  void AddCleanup(intptr_t key, std::function<void()> callback);
  // If `key` (e.g. a context) is destroyed before the executor, the associated
  // callbacks are no longer safe to run.
  void RemoveCleanups(intptr_t key);

 private:
  // Possible states for this executor.
  // Executor starts in kActive state. When Shutdown() is called, Executor
  // is put in the kShuttingDown state. In this state, the executor thread
  // continues to run, but no new nodes are accepted. Finally, when all nodes
  // are drained, the executor is put in the kShutDown state, which causes the
  // thread to exit.
  // If this executor is destroyed without calling shutdown first, it
  // transitions to kShutDown state immediately which causes the thread to exit
  // without running pending nodes.
  enum class ExecutorState {
    kActive,
    kShuttingDown,
    kShutDown,
  };

  enum class NodeState {
    kPENDING,
    kSCHEDULED,
    kDONE,
  };

  struct NodeItem : core::RefCounted {
    // Unique id generated in EagerExecutor::Add(). If item1.id < item2.id, it
    // means item1.node is added before item2.node.
    uint64 id;
    std::unique_ptr<EagerNode> node;
    NodeState state;
  };

  const char* StateStringLocked()
      TF_EXCLUSIVE_LOCKS_REQUIRED(node_queue_mutex_);

  void NodeDone(const core::RefCountPtr<NodeItem>& item, const Status& status,
                bool from_queue);
  void NotifyWaiters(uint64 id) TF_EXCLUSIVE_LOCKS_REQUIRED(node_queue_mutex_);

  // Starts execution of pending EagerNodes. This function loops till executor
  // state_ is set to kShutDown. If any errors are encountered, these are set
  // inside `status_`. The loop blocks anytime there are no pending nodes, or if
  // `status_` is not ok.
  void Run();

  Status RunItem(core::RefCountPtr<NodeItem> item, bool from_queue);
  Status MoveToUnfinished(core::RefCountPtr<NodeItem> item, bool from_queue);

  // The impl of WaitForAllPendingNodes
  // `lock` is the lock that holds node_queue_mutex_.
  Status WaitForAllPendingNodesLocked(mutex_lock* lock)
      TF_EXCLUSIVE_LOCKS_REQUIRED(node_queue_mutex_);

  Status WaitImpl(bool wait_all, uint64 node_id);

  std::atomic<uint64> next_node_id_;

  mutable mutex node_queue_mutex_;

  // Used to signal that some EagerNodes are pending execution.
  condition_variable nodes_pending_ TF_GUARDED_BY(node_queue_mutex_);

  // Queue of pending NodeItems. Ordered by NodeItem::id.
  std::queue<core::RefCountPtr<NodeItem>> node_queue_
      TF_GUARDED_BY(node_queue_mutex_);

  // Ordered by NodeItem::id.
  std::map<uint64, core::RefCountPtr<NodeItem>, std::less<uint64>>
      unfinished_nodes_ TF_GUARDED_BY(node_queue_mutex_);

  // `status_` is set based on any errors raised during execution of a
  // EagerNode.  It remains set until ClearError is called.
  Status status_ TF_GUARDED_BY(node_queue_mutex_);
  std::atomic<bool> ok_ TF_GUARDED_BY(node_queue_mutex_);

  // Map from id of a EagerNode to condition_variables (not owned by the map).
  // These condition_variables are notified and removed when that EagerNode is
  // done executing, or if an error is found in execution of any EagerNode.
  // The map is ordered by id.
  std::multimap<uint64, condition_variable*, std::less<uint64>>
      node_done_notifications_ TF_GUARDED_BY(node_queue_mutex_);

  // thread_exited_notification_ is notified by the `thread_` right before it
  // exits.
  Notification thread_exited_notification_;

  // When state_ is set to kShutDown, it indicates that `thread_` should stop as
  // soon as it is done executing the current EagerNode.
  ExecutorState state_ TF_GUARDED_BY(node_queue_mutex_) =
      ExecutorState::kActive;

  // Thread object that calls the `Run` method in async mode.This thread runs
  // until state_ is set to kShuttingDown. It is `nullptr` in sync mode.
  const std::unique_ptr<Thread> thread_;

  // Last device where remote function with remote inputs was executed.
  const eager::EagerClient* last_eager_client_;

  const bool enable_async_wait_for_remote_function_;

  // Callbacks to run on destruction.
  std::unordered_map<intptr_t, std::vector<std::function<void()>>> cleanups_;
};

inline bool EagerExecutor::Async() const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTh mht_10(mht_10_v, 464, "", "./tensorflow/core/common_runtime/eager/eager_executor.h", "EagerExecutor::Async");
 return thread_ != nullptr; }

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_EAGER_EXECUTOR_H_
