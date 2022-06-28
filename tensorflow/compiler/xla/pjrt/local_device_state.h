/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_LOCAL_DEVICE_STATE_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_LOCAL_DEVICE_STATE_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlocal_device_stateDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlocal_device_stateDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlocal_device_stateDTh() {
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


#include <memory>
#include <random>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/pjrt/event_pool.h"
#include "tensorflow/compiler/xla/pjrt/semaphore.h"
#include "tensorflow/compiler/xla/pjrt/worker_thread.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace xla {

// Class that encapsulates state relating to a device (e.g., a GPU) on which we
// can perform computation and transfers. LocalDeviceState objects only exist
// for devices local to this host.
class LocalDeviceState {
 public:
  // There are three different semantics used by memory allocators on different
  // devices.
  enum AllocationModel {
    // kSynchronous is used by CPU devices.
    //
    // A buffer returned from the allocator can be used immediately.
    //
    // A buffer cannot be freed until after the last stream operation
    // referencing the buffer has completed, so the client is responsible for
    // keeping buffers alive until all device-side activity that consumes those
    // buffers has completed.
    //
    // The client's use of the device allocator corresponds to a view of the
    // tail of the last stream using a buffer.
    kSynchronous,

    // kComputeSynchronous is used by GPU devices.
    //
    // A buffer returned from the allocator at time t can be used after the
    // compute stream has finished executing the last computation enqueued
    // before time t.
    //
    // A buffer b can be freed after:
    //   1) The last use of b on the compute stream has been enqueued, and
    //   2) For any non-compute stream s on which an operation o using b is
    //      enqueued, either:
    //     a) The host has been notified that o has completed, or
    //     b) The next operation to be enqueued on the compute stream is
    //        guaranteed to be started after o has completed.
    //
    // The client's use of the device allocator corresponds to a view of the
    // tail of the compute stream.
    kComputeSynchronized,

    // kAsynchronous is used by TPU devices.
    //
    // A buffer returned from the allocator can be used immediately.
    //
    // A buffer b can be freed as soon as the last stream operation using b has
    // been enqueued.
    //
    // The allocator and lower-level runtime are responsible for keeping buffers
    // alive (if that is needed) from the perspective of the device until any
    // device-side work actually completes.
    //
    // The only exception is when a buffer is transferred between devices since
    // only one of the device executors knows about the transfer, so the buffer
    // must be manually kept alive from the perspective of the other executor.
    kAsynchronous
  };

  // If asynchronous is false, the host will synchronize to the device after
  // each execution or transfer. This is intended for debugging only.
  LocalDeviceState(se::StreamExecutor* executor, LocalClient* client,
                   AllocationModel allocation_model,
                   int max_inflight_computations, bool allow_event_reuse,
                   bool use_callback_stream);
  virtual ~LocalDeviceState();

  se::StreamExecutor* executor() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlocal_device_stateDTh mht_0(mht_0_v, 266, "", "./tensorflow/compiler/xla/pjrt/local_device_state.h", "executor");
 return executor_; }
  // StreamExecutor (local) device ordinal.
  int device_ordinal() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlocal_device_stateDTh mht_1(mht_1_v, 271, "", "./tensorflow/compiler/xla/pjrt/local_device_state.h", "device_ordinal");
 return executor_->device_ordinal(); }

  LocalClient* client() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlocal_device_stateDTh mht_2(mht_2_v, 276, "", "./tensorflow/compiler/xla/pjrt/local_device_state.h", "client");
 return client_; }

  AllocationModel allocation_model() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlocal_device_stateDTh mht_3(mht_3_v, 281, "", "./tensorflow/compiler/xla/pjrt/local_device_state.h", "allocation_model");
 return allocation_model_; }

  EventPool& event_pool() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlocal_device_stateDTh mht_4(mht_4_v, 286, "", "./tensorflow/compiler/xla/pjrt/local_device_state.h", "event_pool");
 return event_pool_; }

  se::Stream* compute_stream() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlocal_device_stateDTh mht_5(mht_5_v, 291, "", "./tensorflow/compiler/xla/pjrt/local_device_state.h", "compute_stream");
 return compute_stream_.get(); }
  se::Stream* host_to_device_stream() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlocal_device_stateDTh mht_6(mht_6_v, 295, "", "./tensorflow/compiler/xla/pjrt/local_device_state.h", "host_to_device_stream");

    return host_to_device_stream_.get();
  }

  // Returns a device to host stream. Allocates streams in a round-robin fashion
  // amongst the available streams.
  se::Stream* GetDeviceToHostStream();

  // Returns a device to device stream. Allocates streams in a round-robin
  // fashion amongst the available streams.
  se::Stream* GetDeviceToDeviceStream();

  // Returns a stream from a pool. The stream is guaranteed not to have any
  // currently outstanding work at its tail.
  std::unique_ptr<se::Stream> BorrowStreamFromPool();
  // Returns a stream to the pool. The caller must ensure the stream does not
  // have any outstanding work at its tail.
  void ReturnStreamToPool(std::unique_ptr<se::Stream> stream);

  // Enqueues a copy of `src_buffer` to `dst_buffer` onto `transfer_stream`.
  virtual Status ThenMemcpyDeviceToDevice(se::Stream* transfer_stream,
                                          se::Stream* dst_stream,
                                          se::DeviceMemoryBase src_buffer,
                                          se::DeviceMemoryBase dst_buffer);

  WorkerThread* execute_thread() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlocal_device_stateDTh mht_7(mht_7_v, 323, "", "./tensorflow/compiler/xla/pjrt/local_device_state.h", "execute_thread");
 return execute_thread_.get(); }

  // Enqueues a host callback on 'stream'. `stream` may, but need not, wait for
  // `callback` to complete. It is safe to call runtime methods from the
  // callback.
  // This API differs from ThenDoHostCallback in two ways:
  // a) ThenDoHostCallback is often constrained in what it can do, in
  //    particular, on GPU the callback runs on a thread belonging to the GPU
  //    runtime and cannot perform GPU operations itself. On GPU, callbacks
  //    execute in a separate thread.
  // b) ThenDoHostCallback waits for the callback to complete.
  void ThenExecuteCallback(se::Stream* stream, std::function<void()> callback);

  // Helpers for releasing values on a worker thread at the tail of a stream on
  // a worker thread. Copies `object`, and destroys the copy when the tail of
  // the stream is reached. The destruction happens either in the caller's
  // thread or on the worker thread (depending on thread schedules), not a
  // device callback, so it is safe if the destructor frees device resource
  // (e.g., GPU objects).
  template <typename T>
  void ThenRelease(se::Stream* stream, T&& object) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlocal_device_stateDTh mht_8(mht_8_v, 346, "", "./tensorflow/compiler/xla/pjrt/local_device_state.h", "ThenRelease");

    ThenExecuteCallback(
        stream, [object = std::forward<T>(object)]() { /* releases object */ });
  }

  Semaphore& compute_semaphore() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSlocal_device_stateDTh mht_9(mht_9_v, 354, "", "./tensorflow/compiler/xla/pjrt/local_device_state.h", "compute_semaphore");
 return compute_semaphore_; }

  // Returns a fresh, PRNG-generated random seed for an XLA computation.
  int GetNewPrngSeed();

 private:
  Status SynchronizeAllActivity();

  AllocationModel allocation_model_;

  EventPool event_pool_;

  // Semaphore used to limit how many programs can be enqueued on the compute
  // stream by the host ahead of the device.
  Semaphore compute_semaphore_;

  se::StreamExecutor* const executor_;
  LocalClient* const client_;
  std::unique_ptr<se::Stream> compute_stream_;
  std::unique_ptr<se::Stream> host_to_device_stream_;
  std::vector<std::unique_ptr<se::Stream>> device_to_host_streams_;
  std::vector<std::unique_ptr<se::Stream>> device_to_device_streams_;

  // Number of device-to-host and device-to-device streams.
  static constexpr int kNumDeviceToHostStreams = 4;
  static constexpr int kNumDeviceToDeviceStreams = 4;

  absl::Mutex mu_;
  int next_device_to_host_stream_ ABSL_GUARDED_BY(mu_) = 0;
  int next_device_to_device_stream_ ABSL_GUARDED_BY(mu_) = 0;
  std::stack<std::unique_ptr<se::Stream>> usage_stream_pool_
      ABSL_GUARDED_BY(mu_);

  std::random_device prng_seed_device_ ABSL_GUARDED_BY(mu_);
  std::mt19937 prng_seed_generator_ ABSL_GUARDED_BY(mu_);
  std::uniform_int_distribution<> prng_seed_distribution_ ABSL_GUARDED_BY(mu_);

  // Callback map pairs callback stream with a device stream and is used for
  // running short host-side callbacks after device side events, without
  // preventing the device-side stream from doing useful work.
  absl::optional<absl::flat_hash_map<se::Stream*, std::unique_ptr<se::Stream>>>
      callback_stream_map_;

  // A worker thread, used for replicated computation launches.
  std::unique_ptr<WorkerThread> execute_thread_;

  // A worker thread, used for callbacks. It is necessary that this be a
  // different thread to the execute thread because we acquire the compute
  // semaphore during calls to Execute but release it from a callback and if
  // they are the same thread we might deadlock.
  std::unique_ptr<WorkerThread> callback_thread_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_LOCAL_DEVICE_STATE_H_
