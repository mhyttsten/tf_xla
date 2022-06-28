/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_PJRT_FUTURE_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_PJRT_FUTURE_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_futureDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_futureDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_futureDTh() {
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


#include <functional>
#include <utility>

#include "absl/types/span.h"
#include "tfrt/host_context/async_dispatch.h"  // from @tf_runtime
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime

namespace xla {

// Helpers for using PjRtFutures.
struct PjRtFutureHelpers {
 public:
  // Keys that are returned by an implementation-specific handler when a client
  // starts to block on a promise.
  //
  // For now, contains a single UID that can be used to identify a TraceMe, but
  // made extensible to allow support for other profilers such as endoscope.
  struct ProfilingKeys {
    uint64_t traceme_context_id = -1;
  };
  // Signature of handler called by the PjRtFuture class before it starts to
  // block a thread.
  using OnBlockStartFn = std::function<ProfilingKeys()>;
  // Signature of handler called by the PjRtFuture class after it finishes
  // blocking a thread.
  using OnBlockEndFn = std::function<void(ProfilingKeys)>;
};

// PjRtFuture<T> is a simple future that is returned by PjRt APIs that
// enqueue asynchronous work, reporting a value of type T (frequently T=Status)
// when the work is complete.
//
// PjRtFuture can be used by the client to wait for work to complete, either via
// a blocking call or a callback.
//
// The implementation wraps a TFRT AsyncValueRef<T>, but we prefer to
// encapsulate the AVR rather than returning it directly for two reasons.
//
// First, we want to retain portability in case a future implementation moves
// away from AsyncValueRef ---- we don't want clients to call arbitrary
// AsyncValueRef APIs.
//
// Second, we want to export different semantics, for
// example we block without the client supplying a HostContext, and support
// integration between blocking and profiling (e.g., TraceMe).
//
// There are two ways to construct a PjRtFuture, one used by clients that
// natively use TFRT, which already have a HostContext and import APIs for
// constructing AsyncValueRefs; and another that avoids exposing TFRT APIs and
// can be used by non-TFRT clients.
template <class T>
class PjRtFuture {
 public:
  // Wrapper for AsyncValueRef<T> that can be used by clients that don't
  // natively use TFRT.
  struct Promise {
   public:
    // Creates an empty promise with !this == true.
    explicit Promise() = default;
    Promise(Promise&& other) = default;
    Promise(const Promise& other) : avr(other.avr.CopyRef()) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_futureDTh mht_0(mht_0_v, 251, "", "./tensorflow/compiler/xla/pjrt/pjrt_future.h", "Promise");
}
    Promise& operator=(const Promise& other) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_futureDTh mht_1(mht_1_v, 255, "", "./tensorflow/compiler/xla/pjrt/pjrt_future.h", "=");

      avr = other.avr.CopyRef();
      return *this;
    }
    bool operator!() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_futureDTh mht_2(mht_2_v, 262, "", "./tensorflow/compiler/xla/pjrt/pjrt_future.h", "!");
 return !avr; }

    // Sets the value of the promise. Must be called at most once.
    //
    // After Set is called, value will be delivered to waiters on the parent
    // PjRtFuture, via blocking or callbacks.
    void Set(T value) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_futureDTh mht_3(mht_3_v, 271, "", "./tensorflow/compiler/xla/pjrt/pjrt_future.h", "Set");
 avr.emplace(std::move(value)); }

   private:
    friend class PjRtFuture<T>;
    explicit Promise(tfrt::AsyncValueRef<T> ref) : avr(std::move(ref)) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_futureDTh mht_4(mht_4_v, 278, "", "./tensorflow/compiler/xla/pjrt/pjrt_future.h", "Promise");
}
    // The underlying TFRT value that can be waited on.
    tfrt::AsyncValueRef<T> avr;
  };

  // Returns a Promise that can be used to construct a PjRtFuture, and then Set
  // later.
  //
  // Used by clients that do not use TFRT natively.
  static Promise CreatePromise() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_futureDTh mht_5(mht_5_v, 290, "", "./tensorflow/compiler/xla/pjrt/pjrt_future.h", "CreatePromise");

    return Promise(tfrt::MakeUnconstructedAsyncValueRef<T>());
  }

  // Constructor for an already-available PjRtFuture.
  //
  // Typically used to eagerly return error values when async work will not
  // be enqueued, e.g., due to invalid arguments.
  explicit PjRtFuture(T t)
      : promise_ref_(tfrt::MakeAvailableAsyncValueRef<T>(t)),
        on_block_start_([]() { return PjRtFutureHelpers::ProfilingKeys(); }),
        on_block_end_([](PjRtFutureHelpers::ProfilingKeys) {}),
        host_ctx_(nullptr) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_futureDTh mht_6(mht_6_v, 305, "", "./tensorflow/compiler/xla/pjrt/pjrt_future.h", "PjRtFuture");
}

  // Constructor used by clients that natively use TFRT and already have a
  // host_ctx that should be used for awaiting promises.
  //
  // on_block_start is called before Await starts to block.
  // on_block_end is called after Await finishes blocking.
  explicit PjRtFuture(
      tfrt::HostContext* host_ctx, tfrt::AsyncValueRef<T> async_value,
      PjRtFutureHelpers::OnBlockStartFn on_block_start =
          []() { return PjRtFutureHelpers::ProfilingKeys(); },
      PjRtFutureHelpers::OnBlockEndFn on_block_end =
          [](PjRtFutureHelpers::ProfilingKeys) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_futureDTh mht_7(mht_7_v, 320, "", "./tensorflow/compiler/xla/pjrt/pjrt_future.h", "lambda");
})
      : promise_ref_(std::move(async_value)),
        on_block_start_(std::move(on_block_start)),
        on_block_end_(std::move(on_block_end)),
        host_ctx_(host_ctx) {}

  // Constructor used by clients that don't natively use TFRT and want to use
  // the wrapped PjRtFuture<T>::Promise class and block without using
  // HostContext.
  //
  // on_block_start is called before Await starts to block.
  // on_block_end is called after Await finishes blocking.
  explicit PjRtFuture(
      Promise promise,
      PjRtFutureHelpers::OnBlockStartFn on_block_start =
          []() { return PjRtFutureHelpers::ProfilingKeys(); },
      PjRtFutureHelpers::OnBlockEndFn on_block_end =
          [](PjRtFutureHelpers::ProfilingKeys) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_futureDTh mht_8(mht_8_v, 340, "", "./tensorflow/compiler/xla/pjrt/pjrt_future.h", "lambda");
})
      : promise_ref_(std::move(promise.avr)),
        on_block_start_(std::move(on_block_start)),
        on_block_end_(std::move(on_block_end)),
        host_ctx_(nullptr) {}

  // Two functions exist to know whether the future is ready, to accomodate
  // the fact some backends (e.g. disributed ones) could take a non-trivial time
  // to check the state of a future.
  //
  // `IsReady()` is guaranteed to return true if the future became ready before
  // `IsReady()` was called. `IsReady()` will return immediately if a call to
  // `Await()` has already returned, or any callback passed to `OnReady` has
  // already been triggered. Otherwise IsReady() may block for the duration of a
  // network message on some backends."
  bool IsReady() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_futureDTh mht_9(mht_9_v, 358, "", "./tensorflow/compiler/xla/pjrt/pjrt_future.h", "IsReady");
 return promise_ref_.IsAvailable(); }
  // `IsKnownReady()` is guaranteed to return immediately. `IsKnownReady()` will
  // always return true if a call to `Await()` has already returned, or any
  // callback passed to `OnReady` has already been triggered. Otherwise,
  // `IsKnownReady()` may return false in some cases in which the future was
  // ready before `IsKnownReady()` was called.
  bool IsKnownReady() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_futureDTh mht_10(mht_10_v, 367, "", "./tensorflow/compiler/xla/pjrt/pjrt_future.h", "IsKnownReady");
 return promise_ref_.IsAvailable(); }

  // Blocks the calling thread until the promise is ready, then returns the
  // final value.
  T Await() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_futureDTh mht_11(mht_11_v, 374, "", "./tensorflow/compiler/xla/pjrt/pjrt_future.h", "Await");

    if (!promise_ref_.IsAvailable()) {
      const auto keys = on_block_start_();
      if (host_ctx_) {
        host_ctx_->Await({promise_ref_.CopyRCRef()});
      } else {
        tfrt::Await({promise_ref_.GetAsyncValue()});
      }
      on_block_end_(keys);
    }
    DCHECK(promise_ref_.IsConcrete());
    return *promise_ref_;
  }

  // Registers callback to be called once the promise is ready, with the final
  // value.
  //
  // callback may be called on an internal system thread or the calling thread.
  // The client should avoid any potentially re-entrant API calls within the
  // callback, for example by using the callback to enqueue work on a
  // client-owned threadpool.
  void OnReady(std::function<void(T)> callback) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_futureDTh mht_12(mht_12_v, 398, "", "./tensorflow/compiler/xla/pjrt/pjrt_future.h", "OnReady");

    promise_ref_.AndThen(
        [promise = promise_ref_.CopyRef(), callback = std::move(callback)]() {
          DCHECK(promise.IsConcrete());
          callback(*promise);
        });
  }

 private:
  // Wrapped object to wait on.
  tfrt::AsyncValueRef<T> promise_ref_;
  // Function that is called before a thread starts blocking on the promise.
  PjRtFutureHelpers::OnBlockStartFn on_block_start_;
  // Function that is called after a thread finishes blocking on the promise.
  PjRtFutureHelpers::OnBlockEndFn on_block_end_;
  // Used only to await promise_ref_.
  tfrt::HostContext* host_ctx_;  // not owned
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_PJRT_FUTURE_H_
