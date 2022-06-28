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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTcc() {
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

#include "tensorflow/core/common_runtime/eager/eager_executor.h"

#include <forward_list>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace {
bool IsAsyncWaitForRemoteFunctionEnabled() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTcc mht_0(mht_0_v, 195, "", "./tensorflow/core/common_runtime/eager/eager_executor.cc", "IsAsyncWaitForRemoteFunctionEnabled");

  bool enabled = true;
  TF_CHECK_OK(ReadBoolFromEnvVar("TF_ENABLE_ASYNC_WAIT_FOR_REMOTE_FUNCTION",
                                 true, &enabled));
  return enabled;
}
}  // namespace

EagerExecutor::EagerExecutor(bool async)
    : next_node_id_(0),
      ok_(true),
      thread_(async ? tensorflow::Env::Default()->StartThread(
                          tensorflow::ThreadOptions(), "eager_async_executor",
                          std::bind(&EagerExecutor::Run, this))
                    : nullptr),
      last_eager_client_(nullptr),
      enable_async_wait_for_remote_function_(
          IsAsyncWaitForRemoteFunctionEnabled()) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTcc mht_1(mht_1_v, 215, "", "./tensorflow/core/common_runtime/eager/eager_executor.cc", "EagerExecutor::EagerExecutor");
}

EagerExecutor::~EagerExecutor() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTcc mht_2(mht_2_v, 220, "", "./tensorflow/core/common_runtime/eager/eager_executor.cc", "EagerExecutor::~EagerExecutor");

  tensorflow::mutex_lock l(node_queue_mutex_);
  state_ = ExecutorState::kShutDown;
  nodes_pending_.notify_all();
  for (const auto& cleanups_for_key : cleanups_) {
    for (const std::function<void()>& cleanup : cleanups_for_key.second) {
      cleanup();
    }
  }
}

Status EagerExecutor::ShutDown() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTcc mht_3(mht_3_v, 234, "", "./tensorflow/core/common_runtime/eager/eager_executor.cc", "EagerExecutor::ShutDown");

  {
    bool has_thread;
    Status status;
    {
      tensorflow::mutex_lock l(node_queue_mutex_);
      if (state_ != ExecutorState::kShutDown) {
        // if the state is kShutDown, we don't return here because we want to
        // make sure the executor thread has ended (if there is one).
        // So, we fall through to
        // thread_exited_notification_.WaitForNotification() below.
        state_ = ExecutorState::kShuttingDown;
      }
      // It is OK to ignore the returned status here because it will be saved
      // as the final status_.
      WaitForAllPendingNodesLocked(&l).IgnoreError();
      state_ = ExecutorState::kShutDown;
      has_thread = thread_ != nullptr;
      status = status_;
      if (has_thread) {
        nodes_pending_.notify_all();
      }
    }
    if (!has_thread) {
      return status;
    }
  }

  thread_exited_notification_.WaitForNotification();

  return status();
}

const char* EagerExecutor::StateStringLocked() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTcc mht_4(mht_4_v, 270, "", "./tensorflow/core/common_runtime/eager/eager_executor.cc", "EagerExecutor::StateStringLocked");

  switch (state_) {
    case ExecutorState::kActive:
      return "Active";
    case ExecutorState::kShuttingDown:
      return "ShuttingDown";
    case ExecutorState::kShutDown:
      return "ShutDown";
  }
}

Status EagerExecutor::SyncExecute(EagerNode* node) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTcc mht_5(mht_5_v, 284, "", "./tensorflow/core/common_runtime/eager/eager_executor.cc", "EagerExecutor::SyncExecute");

  if (Async()) {
    return errors::Internal("Executor does not support async execution");
  }
  if (node->AsAsync() != nullptr) {
    return errors::Internal("Executor does not support executing async nodes");
  }
  // NOTE: SyncExecute runs every node regardless of error status in executor.

  uint64 id = next_node_id_++;

  Status s = node->Prepare();
  if (!s.ok()) {
    return s;
  }

  // Inline execution in sync mode.
  s = node->Run();
  tensorflow::mutex_lock l(node_queue_mutex_);
  NotifyWaiters(id);
  return s;
}

Status EagerExecutor::AddOrExecute(std::unique_ptr<EagerNode> node) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTcc mht_6(mht_6_v, 310, "", "./tensorflow/core/common_runtime/eager/eager_executor.cc", "EagerExecutor::AddOrExecute");

  Status status;
  core::RefCountPtr<NodeItem> item(new NodeItem);
  item->id = next_node_id_++;
  item->node = std::move(node);
  item->state = NodeState::kPENDING;

  status = item->node->Prepare();
  if (!status.ok()) {
    item->node->Abort(status);
    return status;
  }

  // Inline execution in sync mode.
  if (!Async()) {
    // In sync mode, run the node item regardless of executor status.
    return RunItem(std::move(item), /*from_queue=*/false);
  } else {
    tensorflow::mutex_lock l(node_queue_mutex_);
    DVLOG(3) << "Add node [id " << item->id << "]" << item->node->DebugString()
             << " with status: " << status_.ToString();
    if (state_ != ExecutorState::kActive) {
      status = errors::FailedPrecondition(
          "EagerExecutor accepts new EagerNodes to run only in Active state. "
          "Current state is '",
          StateStringLocked(), "'");
    } else {
      status = status_;
      if (status.ok()) {
        node_queue_.push(std::move(item));
        // If there were no previous nodes pending, wake the run thread to
        // start processing requests again.
        if (node_queue_.size() == 1) {
          nodes_pending_.notify_all();
        }

        return Status::OK();
      }
    }
  }

  // If we are unable to add the node to the queue, we must call Abort. However,
  // we want to do that outside of the scope of the lock since the Abort may
  // try to call EagerExecutor::AddOrExecute()
  item->node->Abort(status);

  return status;
}

tensorflow::Status EagerExecutor::WaitForAllPendingNodes() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTcc mht_7(mht_7_v, 362, "", "./tensorflow/core/common_runtime/eager/eager_executor.cc", "EagerExecutor::WaitForAllPendingNodes");

  tensorflow::mutex_lock l(node_queue_mutex_);
  return WaitForAllPendingNodesLocked(&l);
}

tensorflow::Status EagerExecutor::WaitForAllPendingNodesLocked(
    mutex_lock* lock) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTcc mht_8(mht_8_v, 371, "", "./tensorflow/core/common_runtime/eager/eager_executor.cc", "EagerExecutor::WaitForAllPendingNodesLocked");

  tensorflow::condition_variable cond;
  // Don't wait if an error is already set.
  if (!status_.ok()) return status_;
  if (node_queue_.empty() && unfinished_nodes_.empty())
    return tensorflow::Status::OK();
  // node_queue_ must be empty in sync mode.
  DCHECK(Async() || node_queue_.empty());
  auto last_id = next_node_id_ - 1;
  DVLOG(3) << "Wait for Node: [id " << last_id << "] ";
  node_done_notifications_.insert(std::make_pair(last_id, &cond));
  cond.wait(*lock);
  // Note that we could be woken up if an error occurs, even though the node has
  // not actually executed.
  return status_;
}

void EagerExecutor::ClearError() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTcc mht_9(mht_9_v, 391, "", "./tensorflow/core/common_runtime/eager/eager_executor.cc", "EagerExecutor::ClearError");

  // TODO(iga): Check state_ and return an error if it is not kActive.
  if (ok()) return;

  tensorflow::mutex_lock l(node_queue_mutex_);
  // If an error was set, node_done_notifications_ and node_queue_ should have
  // been cleared, and no new entries should have been added since.
  DCHECK(node_done_notifications_.empty());
  DCHECK(node_queue_.empty());
  status_ = tensorflow::Status::OK();
  ok_ = true;
  last_eager_client_ = nullptr;
  nodes_pending_.notify_all();
}

void EagerExecutor::NodeDone(const core::RefCountPtr<NodeItem>& item,
                             const Status& status, bool from_queue) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTcc mht_10(mht_10_v, 410, "", "./tensorflow/core/common_runtime/eager/eager_executor.cc", "EagerExecutor::NodeDone");

  DVLOG(3) << "Node Done: [id " << item->id << "] " << item->node->DebugString()
           << " with status: " << status.ToString();
  DCHECK(item->state != NodeState::kDONE);
  item->state = NodeState::kDONE;

  bool async = item->node->AsAsync() != nullptr;
  // If executing synchronously we don't need to notify if status is OK since
  // the node  was never added to the unfinished_nodes_ list and nobody should
  // ever be waiting for it.
  if (status.ok() && !from_queue && !async) {
    return;
  }

  std::forward_list<core::RefCountPtr<NodeItem>> items_to_destroy;
  {
    mutex_lock l(node_queue_mutex_);
    if (!status_.ok()) return;

    bool need_notification = from_queue;
    if (from_queue) {
      // Since this was from the async queue, pop it from the front of the queue
      DCHECK(!node_queue_.empty() && item.get() == node_queue_.front().get());
      node_queue_.pop();
    } else if (async) {
      // If it is an Async node then we will find the node in the unfinished
      // nodes list. However we only notify if we are at the front of the list
      // since we don't want to notify any waiters of earlier nodes.
      need_notification = item->id == unfinished_nodes_.begin()->first;
      // Remove item if it exists in unfinished_nodes_.
      // With async execution, if two separate nodes failed and enter this
      // callback, then the second node might not find itself in
      // unfinished_nodes_ in the following senario:
      //   1) Callback of the first failed node clears unfinished_nodes_
      //   2) ClearError is called and executor status_ is set to OK
      //   3) Callback of the second failed node is triggered
      // In this case, do not taint the executor status or other note items
      // because they are inserted after the ClearError.
      auto result = unfinished_nodes_.erase(item->id);
      if (result == 0) return;
    }

    if (!status.ok() && item->node->Fatal()) {
      // Since we received an error, broadcast to any waiters.
      need_notification = true;
      status_ = status;
      ok_ = false;
      if (Async()) {
        // We remove any pending ops so that we don't try to execute them if
        // ClearError is called.
        errors::AppendToMessage(&status_,
                                "Encountered when executing an operation using "
                                "EagerExecutor. This error cancels all future "
                                "operations and poisons their output tensors.");
      }
      while (!node_queue_.empty()) {
        items_to_destroy.push_front(std::move(node_queue_.front()));
        node_queue_.pop();
      }
      for (auto& it : unfinished_nodes_) {
        items_to_destroy.push_front(std::move(it.second));
      }
      unfinished_nodes_.clear();
    }
    if (need_notification) {
      NotifyWaiters(item->id);
    }
  }

  for (auto& item : items_to_destroy) {
    item->node->Abort(status);
  }
  // nodes_to_destroy will be destructed here, while not holding
  // node_queue_mutex_. This is important because, unfortunately, some nodes'
  // destructors can enqueue more operations onto this executor and cause
  // a deadlock.
}

void EagerExecutor::NotifyWaiters(uint64 id) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTcc mht_11(mht_11_v, 491, "", "./tensorflow/core/common_runtime/eager/eager_executor.cc", "EagerExecutor::NotifyWaiters");

  if (!node_done_notifications_.empty()) {
    uint64 upperbound_id = 0;
    if (!unfinished_nodes_.empty()) {
      upperbound_id = unfinished_nodes_.begin()->first - 1;
    } else if (!node_queue_.empty()) {
      upperbound_id = node_queue_.front()->id - 1;
    } else {
      upperbound_id = next_node_id_ - 1;
    }
    if (upperbound_id < id) {
      return;
    }
    DVLOG(3) << "Notify node done: [id " << id << " to " << upperbound_id
             << "] ";
    // Note that we notify all waiting threads in case an error has
    // occurred. These calling threads are responsible for checking status_
    // before proceeding.
    const auto range =
        status_.ok()
            ? make_pair(node_done_notifications_.lower_bound(id),
                        node_done_notifications_.upper_bound(upperbound_id))
            : make_pair(node_done_notifications_.begin(),
                        node_done_notifications_.end());
    for (auto it = range.first; it != range.second; ++it) {
      it->second->notify_all();
    }
    node_done_notifications_.erase(range.first, range.second);
  }
}

void EagerExecutor::Run() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTcc mht_12(mht_12_v, 525, "", "./tensorflow/core/common_runtime/eager/eager_executor.cc", "EagerExecutor::Run");

  auto thread_exited_notifier =
      gtl::MakeCleanup([this] { thread_exited_notification_.Notify(); });
  while (true) {
    core::RefCountPtr<NodeItem> curr_item;
    {
      tensorflow::mutex_lock l(node_queue_mutex_);
      while (node_queue_.empty() || !status_.ok()) {
        if (state_ == ExecutorState::kShutDown) return;
        nodes_pending_.wait(l);
      }
      // Obtain raw pointer since we don't want to remove from the queue until
      // the node has been run. Otherwise, WaitForAllPendingNodes can return
      // too early.
      // Note, we don't std::move from the here because the front of the queue
      // will then contain a nullptr. This can be a problem in
      // WaitForAllPendingNodes where we get the top EagerNode pointer
      // and register a notification for its completion.
      curr_item.reset(node_queue_.front().get());
      curr_item->Ref();
    }
    Status status = RunItem(std::move(curr_item), /*from_queue=*/true);
    if (!status.ok()) {
      VLOG(1) << "Failed to run item: " << status;
    }
  }
}

Status EagerExecutor::RunItem(core::RefCountPtr<NodeItem> item,
                              bool from_queue) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTcc mht_13(mht_13_v, 557, "", "./tensorflow/core/common_runtime/eager/eager_executor.cc", "EagerExecutor::RunItem");

  DVLOG(3) << "Running Node: [id " << item->id << "] "
           << item->node->DebugString();
  AsyncRemoteExecuteNode* async_remote_node =
      item->node->AsAsyncRemoteExecuteNode();
  if (enable_async_wait_for_remote_function_) {
    if (async_remote_node != nullptr) {
      if (last_eager_client_ != nullptr &&
          async_remote_node->eager_client() != nullptr &&
          last_eager_client_ != async_remote_node->eager_client()) {
        // Running a remote function, need to sync if the function is going to
        // different device than last time we run remote distributed function.
        DVLOG(3) << "Executing Sync Executor for node" << item->id;
        tensorflow::Status status = async_remote_node->SyncExecutors();
        if (!status.ok()) {
          NodeDone(item, status, from_queue);
          return status;
        }
        last_eager_client_ = nullptr;
      }
      if (async_remote_node->eager_client() != nullptr &&
          async_remote_node->needs_remote_inputs() &&
          async_remote_node->allow_multiple_pending_requests()) {
        // We are running remote distributed function, update
        // last_remote_device_name_.
        last_eager_client_ = async_remote_node->eager_client();
      }
    }
  }

  AsyncEagerNode* async_node = item->node->AsAsync();
  if (async_node == nullptr) {
    tensorflow::Status status = item->node->Run();
    NodeDone(item, status, from_queue);
    return status;
  }

  item->state = NodeState::kSCHEDULED;
  auto async_ref = item.get();
  async_ref->Ref();

  TF_RETURN_IF_ERROR(MoveToUnfinished(std::move(item), from_queue));

  async_node->RunAsync([this, async_ref](const Status& status) {
    core::RefCountPtr<NodeItem> async_item(async_ref);
    NodeDone(async_item, status, false);
  });

  // Return the status of the executor in case we are in an error state.
  return status();
}

Status EagerExecutor::MoveToUnfinished(core::RefCountPtr<NodeItem> item,
                                       bool from_queue) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTcc mht_14(mht_14_v, 613, "", "./tensorflow/core/common_runtime/eager/eager_executor.cc", "EagerExecutor::MoveToUnfinished");

  tensorflow::mutex_lock l(node_queue_mutex_);
  if (!status_.ok()) {
    return status_;
  }

  if (from_queue) {
    DCHECK(!node_queue_.empty() && item.get() == node_queue_.front().get());
    node_queue_.pop();
  }

  DVLOG(3) << "Add Node: [id " << item->id << "] to unfinished map.";
  unfinished_nodes_.emplace_hint(unfinished_nodes_.end(), item->id,
                                 std::move(item));

  return Status::OK();
}

void EagerExecutor::AddCleanup(intptr_t key, std::function<void()> callback) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTcc mht_15(mht_15_v, 634, "", "./tensorflow/core/common_runtime/eager/eager_executor.cc", "EagerExecutor::AddCleanup");

  cleanups_[key].push_back(callback);
}

void EagerExecutor::RemoveCleanups(intptr_t key) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSeagerPSeager_executorDTcc mht_16(mht_16_v, 641, "", "./tensorflow/core/common_runtime/eager/eager_executor.cc", "EagerExecutor::RemoveCleanups");
 cleanups_.erase(key); }

}  // namespace tensorflow
