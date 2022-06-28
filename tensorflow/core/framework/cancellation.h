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

#ifndef TENSORFLOW_CORE_FRAMEWORK_CANCELLATION_H_
#define TENSORFLOW_CORE_FRAMEWORK_CANCELLATION_H_
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
class MHTracer_DTPStensorflowPScorePSframeworkPScancellationDTh {
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
   MHTracer_DTPStensorflowPScorePSframeworkPScancellationDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSframeworkPScancellationDTh() {
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


#include <atomic>
#include <functional>

#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/stringpiece.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

// A token that can be used to register and deregister a
// CancelCallback with a CancellationManager.
//
// CancellationToken values must be created by a call to
// CancellationManager::get_cancellation_token.
typedef int64_t CancellationToken;

// A callback that is invoked when a step is canceled.
//
// NOTE(mrry): See caveats about CancelCallback implementations in the
// comment for CancellationManager::RegisterCallback.
typedef std::function<void()> CancelCallback;

// This class should never simultaneously be used as the cancellation manager
// for two separate sets of executions (i.e two separate steps, or two separate
// function executions).
class CancellationManager {
 public:
  // A value that won't be returned by get_cancellation_token().
  static const CancellationToken kInvalidToken;

  CancellationManager();

  // Constructs a new CancellationManager that is a "child" of `*parent`.
  //
  // If `*parent` is cancelled, `*this` will be cancelled. `*parent` must
  // outlive the created CancellationManager.
  explicit CancellationManager(CancellationManager* parent);

  ~CancellationManager();

  // Run all callbacks associated with this manager.
  void StartCancel();

  // Run all callbacks associated with this manager with a status.
  // Currently the status is for logging purpose only. See also
  // CancellationManager::RegisterCallbackWithErrorLogging.
  void StartCancelWithStatus(const Status& status);

  // Returns true iff StartCancel() has been called.
  bool IsCancelled() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScancellationDTh mht_0(mht_0_v, 243, "", "./tensorflow/core/framework/cancellation.h", "IsCancelled");
 return is_cancelled_.load(std::memory_order_acquire); }

  // Returns a token that must be used in calls to RegisterCallback
  // and DeregisterCallback.
  CancellationToken get_cancellation_token() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSframeworkPScancellationDTh mht_1(mht_1_v, 250, "", "./tensorflow/core/framework/cancellation.h", "get_cancellation_token");

    return next_cancellation_token_.fetch_add(1);
  }

  // Attempts to register the given callback to be invoked when this
  // manager is cancelled. Returns true if the callback was
  // registered; returns false if this manager was already cancelled,
  // and the callback was not registered.
  //
  // If this method returns false, it is the caller's responsibility
  // to perform any cancellation cleanup.
  //
  // This method is tricky to use correctly. The following usage pattern
  // is recommended:
  //
  // class ObjectWithCancellableOperation {
  //   mutex mu_;
  //   void CancellableOperation(CancellationManager* cm,
  //                             std::function<void(Status)> callback) {
  //     bool already_cancelled;
  //     CancellationToken token = cm->get_cancellation_token();
  //     {
  //       mutex_lock(mu_);
  //       already_cancelled = !cm->RegisterCallback(
  //           [this, token]() { Cancel(token); });
  //       if (!already_cancelled) {
  //         // Issue asynchronous operation. Associate the pending operation
  //         // with `token` in some object state, or provide another way for
  //         // the Cancel method to look up the operation for cancellation.
  //         // Ensure that `cm->DeregisterCallback(token)` is called without
  //         // holding `mu_`, before `callback` is invoked.
  //         // ...
  //       }
  //     }
  //     if (already_cancelled) {
  //       callback(errors::Cancelled("Operation was cancelled"));
  //     }
  //   }
  //
  //   void Cancel(CancellationToken token) {
  //     mutex_lock(mu_);
  //     // Take action to cancel the operation with the given cancellation
  //     // token.
  //   }
  //
  // NOTE(mrry): The caller should take care that (i) the calling code
  // is robust to `callback` being invoked asynchronously (e.g. from
  // another thread), (ii) `callback` is deregistered by a call to
  // this->DeregisterCallback(token) when the operation completes
  // successfully, and (iii) `callback` does not invoke any method
  // on this cancellation manager. Furthermore, it is important that
  // the eventual caller of the complementary DeregisterCallback does not
  // hold any mutexes that are required by `callback`.
  bool RegisterCallback(CancellationToken token, CancelCallback callback);

  // Similar to RegisterCallback, but if the cancellation manager starts a
  // cancellation with an error status, it will log the error status before
  // invoking the callback. `callback_name` is a human-readable name of the
  // callback, which will be displayed on the log.
  bool RegisterCallbackWithErrorLogging(CancellationToken token,
                                        CancelCallback callback,
                                        tensorflow::StringPiece callback_name);

  // Deregister the callback that, when registered, was associated
  // with the given cancellation token. Returns true iff the callback
  // was deregistered and will not be invoked; otherwise returns false
  // after the callback has been invoked, blocking if necessary.
  //
  // NOTE(mrry): This method may block if cancellation is in progress.
  // The caller of this method must not hold any mutexes that are required
  // to invoke any cancellation callback that has been registered with this
  // cancellation manager.
  bool DeregisterCallback(CancellationToken token);

  // Deregister the callback that, when registered, was associated
  // with the given cancellation token. Returns true iff the callback
  // was deregistered and will not be invoked; otherwise returns false
  // immediately, with no guarantee that the callback has completed.
  //
  // This method is guaranteed to return true if StartCancel has not been
  // called.
  bool TryDeregisterCallback(CancellationToken token);

  // Returns true iff cancellation is in progress.
  bool IsCancelling();

 private:
  struct CallbackConfiguration {
    CancelCallback callback;
    std::string name;
    bool log_error = false;
  };

  struct State {
    Notification cancelled_notification;
    gtl::FlatMap<CancellationToken, CallbackConfiguration> callbacks;

    // If this CancellationManager has any children, this member points to the
    // head of a doubly-linked list of its children.
    CancellationManager* first_child = nullptr;  // Not owned.
  };

  bool RegisterCallbackConfig(CancellationToken token,
                              CallbackConfiguration config);

  bool RegisterChild(CancellationManager* child);
  void DeregisterChild(CancellationManager* child);

  bool is_cancelling_;
  std::atomic_bool is_cancelled_;
  std::atomic<CancellationToken> next_cancellation_token_;

  CancellationManager* const parent_ = nullptr;  // Not owned.

  // If this CancellationManager is associated with a parent, this member will
  // be set to `true` after this is removed from the parent's list of children.
  bool is_removed_from_parent_ TF_GUARDED_BY(parent_->mu_) = false;

  // If this CancellationManager is associated with a parent, these members form
  // a doubly-linked list of that parent's children.
  //
  // These fields are valid only when `this->is_removed_from_parent_` is false.
  CancellationManager* prev_sibling_ TF_GUARDED_BY(parent_->mu_) =
      nullptr;  // Not owned.
  CancellationManager* next_sibling_ TF_GUARDED_BY(parent_->mu_) =
      nullptr;  // Not owned.

  mutex mu_;
  std::unique_ptr<State> state_ TF_GUARDED_BY(mu_);
};

// Registers the given cancellation callback, returning a function that can be
// used to deregister the callback. If `cancellation_manager` is NULL, no
// registration occurs and `deregister_fn` will be a no-op.
Status RegisterCancellationCallback(CancellationManager* cancellation_manager,
                                    std::function<void()> callback,
                                    std::function<void()>* deregister_fn);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_CANCELLATION_H_
