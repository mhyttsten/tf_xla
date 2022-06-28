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
class MHTracer_DTPStensorflowPScorePSframeworkPScancellationDTcc {
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
   MHTracer_DTPStensorflowPScorePSframeworkPScancellationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPScancellationDTcc() {
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

#include "tensorflow/core/framework/cancellation.h"

#include <forward_list>

#include "absl/memory/memory.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

const CancellationToken CancellationManager::kInvalidToken = -1;

CancellationManager::CancellationManager()
    : is_cancelling_(false),
      is_cancelled_(false),
      next_cancellation_token_(0) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScancellationDTcc mht_0(mht_0_v, 201, "", "./tensorflow/core/framework/cancellation.cc", "CancellationManager::CancellationManager");
}

CancellationManager::CancellationManager(CancellationManager* parent)
    : is_cancelling_(false), next_cancellation_token_(0), parent_(parent) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScancellationDTcc mht_1(mht_1_v, 207, "", "./tensorflow/core/framework/cancellation.cc", "CancellationManager::CancellationManager");

  is_cancelled_ = parent->RegisterChild(this);
}

void CancellationManager::StartCancel() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScancellationDTcc mht_2(mht_2_v, 214, "", "./tensorflow/core/framework/cancellation.cc", "CancellationManager::StartCancel");

  // An "OK" status will not be logged by a callback registered by
  // RegisterCallbackWithErrorLogging.
  StartCancelWithStatus(Status::OK());
}

void CancellationManager::StartCancelWithStatus(const Status& status) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScancellationDTcc mht_3(mht_3_v, 223, "", "./tensorflow/core/framework/cancellation.cc", "CancellationManager::StartCancelWithStatus");

  gtl::FlatMap<CancellationToken, CallbackConfiguration> callbacks_to_run;
  std::forward_list<CancellationManager*> children_to_cancel;
  Notification* cancelled_notification = nullptr;
  {
    mutex_lock l(mu_);
    if (is_cancelled_.load(std::memory_order_relaxed) || is_cancelling_) {
      return;
    }
    is_cancelling_ = true;
    if (state_) {
      std::swap(state_->callbacks, callbacks_to_run);

      // Remove all children from the list of children.
      CancellationManager* child = state_->first_child;
      while (child != nullptr) {
        children_to_cancel.push_front(child);
        child->is_removed_from_parent_ = true;
        child = child->next_sibling_;
      }
      state_->first_child = nullptr;

      cancelled_notification = &state_->cancelled_notification;
    }
  }
  // We call these callbacks without holding mu_, so that concurrent
  // calls to DeregisterCallback, which can happen asynchronously, do
  // not block. The callbacks remain valid because any concurrent call
  // to DeregisterCallback will block until the
  // cancelled_notification_ is notified.
  for (auto key_and_value : callbacks_to_run) {
    CallbackConfiguration& config = key_and_value.second;
    if (!status.ok() && config.log_error) {
      LOG(WARNING) << "Cancellation callback \"" << config.name
                   << "\" is triggered due to a "
                   << (StatusGroup::IsDerived(status) ? "derived" : "root")
                   << " error: " << status.ToString();
    }
    config.callback();
  }
  for (CancellationManager* child : children_to_cancel) {
    child->StartCancelWithStatus(status);
  }
  {
    mutex_lock l(mu_);
    is_cancelling_ = false;
    is_cancelled_.store(true, std::memory_order_release);
  }
  if (cancelled_notification) {
    cancelled_notification->Notify();
  }
}

bool CancellationManager::RegisterCallback(CancellationToken token,
                                           CancelCallback callback) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScancellationDTcc mht_4(mht_4_v, 280, "", "./tensorflow/core/framework/cancellation.cc", "CancellationManager::RegisterCallback");

  return RegisterCallbackConfig(
      token, CallbackConfiguration{callback, "", false});
}

bool CancellationManager::RegisterCallbackWithErrorLogging(
    CancellationToken token, CancelCallback callback,
    tensorflow::StringPiece callback_name) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScancellationDTcc mht_5(mht_5_v, 290, "", "./tensorflow/core/framework/cancellation.cc", "CancellationManager::RegisterCallbackWithErrorLogging");

  return RegisterCallbackConfig(
      token, CallbackConfiguration{callback, std::string(callback_name), true});
}

bool CancellationManager::RegisterCallbackConfig(CancellationToken token,
                                                 CallbackConfiguration config) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScancellationDTcc mht_6(mht_6_v, 299, "", "./tensorflow/core/framework/cancellation.cc", "CancellationManager::RegisterCallbackConfig");

  DCHECK_LT(token, next_cancellation_token_) << "Invalid cancellation token";
  mutex_lock l(mu_);
  bool should_register = !is_cancelled_ && !is_cancelling_;
  if (should_register) {
    if (!state_) {
      state_ = absl::make_unique<State>();
    }
    std::swap(state_->callbacks[token], config);
  }
  return should_register;
}

bool CancellationManager::DeregisterCallback(CancellationToken token) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScancellationDTcc mht_7(mht_7_v, 315, "", "./tensorflow/core/framework/cancellation.cc", "CancellationManager::DeregisterCallback");

  mu_.lock();
  if (is_cancelled_) {
    mu_.unlock();
    return false;
  } else if (is_cancelling_) {
    Notification* cancelled_notification =
        state_ ? &state_->cancelled_notification : nullptr;
    mu_.unlock();
    // Wait for all of the cancellation callbacks to be called. This
    // wait ensures that the caller of DeregisterCallback does not
    // return immediately and free objects that may be used in the
    // execution of any currently pending callbacks in StartCancel.
    if (cancelled_notification) {
      cancelled_notification->WaitForNotification();
    }
    return false;
  } else {
    if (state_) {
      state_->callbacks.erase(token);
    }
    mu_.unlock();
    return true;
  }
}

bool CancellationManager::RegisterChild(CancellationManager* child) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScancellationDTcc mht_8(mht_8_v, 344, "", "./tensorflow/core/framework/cancellation.cc", "CancellationManager::RegisterChild");

  mutex_lock l(mu_);
  if (is_cancelled_.load(std::memory_order_relaxed) || is_cancelling_) {
    child->is_removed_from_parent_ = true;
    return true;
  }

  if (!state_) {
    state_ = absl::make_unique<State>();
  }

  // Push `child` onto the front of the list of children.
  CancellationManager* current_head = state_->first_child;
  state_->first_child = child;
  child->prev_sibling_ = nullptr;
  child->next_sibling_ = current_head;
  if (current_head) {
    current_head->prev_sibling_ = child;
  }

  return false;
}

void CancellationManager::DeregisterChild(CancellationManager* child) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScancellationDTcc mht_9(mht_9_v, 370, "", "./tensorflow/core/framework/cancellation.cc", "CancellationManager::DeregisterChild");

  DCHECK_EQ(child->parent_, this);
  Notification* cancelled_notification = nullptr;
  {
    mutex_lock l(mu_);
    if (!child->is_removed_from_parent_) {
      // Remove the child from this manager's list of children.
      DCHECK(state_);

      if (child->prev_sibling_ == nullptr) {
        // The child was at the head of the list.
        DCHECK_EQ(state_->first_child, child);
        state_->first_child = child->next_sibling_;
      } else {
        child->prev_sibling_->next_sibling_ = child->next_sibling_;
      }

      if (child->next_sibling_ != nullptr) {
        child->next_sibling_->prev_sibling_ = child->prev_sibling_;
      }

      child->is_removed_from_parent_ = true;
    }
    if (is_cancelling_) {
      cancelled_notification = &state_->cancelled_notification;
    }
  }

  // Wait for an ongoing call to StartCancel() to finish. This wait ensures that
  // the caller of DeregisterChild does not return immediately and free a child
  // that may currently be being cancelled by StartCancel().
  if (cancelled_notification) {
    cancelled_notification->WaitForNotification();
  }
}

bool CancellationManager::TryDeregisterCallback(CancellationToken token) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScancellationDTcc mht_10(mht_10_v, 409, "", "./tensorflow/core/framework/cancellation.cc", "CancellationManager::TryDeregisterCallback");

  mutex_lock lock(mu_);
  if (is_cancelled_ || is_cancelling_) {
    return false;
  } else {
    if (state_) {
      state_->callbacks.erase(token);
    }
    return true;
  }
}

CancellationManager::~CancellationManager() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScancellationDTcc mht_11(mht_11_v, 424, "", "./tensorflow/core/framework/cancellation.cc", "CancellationManager::~CancellationManager");

  if (parent_) {
    parent_->DeregisterChild(this);
  }
  if (state_) {
    StartCancel();
  }
}

bool CancellationManager::IsCancelling() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScancellationDTcc mht_12(mht_12_v, 436, "", "./tensorflow/core/framework/cancellation.cc", "CancellationManager::IsCancelling");

  mutex_lock lock(mu_);
  return is_cancelling_;
}

Status RegisterCancellationCallback(CancellationManager* cancellation_manager,
                                    CancelCallback callback,
                                    std::function<void()>* deregister_fn) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScancellationDTcc mht_13(mht_13_v, 446, "", "./tensorflow/core/framework/cancellation.cc", "RegisterCancellationCallback");

  if (cancellation_manager) {
    CancellationToken token = cancellation_manager->get_cancellation_token();
    if (!cancellation_manager->RegisterCallback(token, std::move(callback))) {
      return errors::Cancelled("Operation was cancelled");
    }
    *deregister_fn = [cancellation_manager, token]() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScancellationDTcc mht_14(mht_14_v, 455, "", "./tensorflow/core/framework/cancellation.cc", "lambda");

      cancellation_manager->DeregisterCallback(token);
    };
  } else {
    VLOG(1) << "Cancellation manager is not set. Cancellation callback will "
               "not be registered.";
    *deregister_fn = []() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScancellationDTcc mht_15(mht_15_v, 464, "", "./tensorflow/core/framework/cancellation.cc", "lambda");
};
  }
  return Status::OK();
}

}  // end namespace tensorflow
