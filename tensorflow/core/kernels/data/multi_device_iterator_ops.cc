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
class MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc() {
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
#include <atomic>
#include <deque>

#include "tensorflow/core/common_runtime/input_colocation_exemption_registry.h"
#include "tensorflow/core/common_runtime/process_function_library_runtime.h"
#include "tensorflow/core/data/dataset_utils.h"
#include "tensorflow/core/data/root_dataset.h"
#include "tensorflow/core/data/unbounded_thread_pool.h"
#include "tensorflow/core/framework/cancellation.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_handle_cache.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_op_kernel.h"
#include "tensorflow/core/kernels/data/iterator_ops.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/refcount.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace data {
namespace {

const char kAnonymousMultiDeviceIterator[] = "AnonymousMultiDeviceIterator";
const char kAnonymousMultiDeviceIteratorV3[] = "AnonymousMultiDeviceIteratorV3";
const char kDevices[] = "devices";
const char kOutputShapes[] = "output_shapes";
const char kOutputTypes[] = "output_types";

struct HostBufferElement {
  Status status;
  bool end_of_sequence;
  std::vector<Tensor> value;
};

using MultiDeviceIteratorCallback =
    std::function<void(const HostBufferElement&)>;

// MultiDeviceIterator provides the ability for multiple devices to fetch from
// one iterator in a roundrobin sequence, which is deterministic. This means
// that, for exmaple, starting from the beginning GetNextFromShard(0) always
// gets the first element and GetNextFromShard(1) always gets the second
// element, even if GetNextFromShard(1) is called before GetNextFromShard(0).
//
// Note on cancellation:
//   * MultiDeviceIterator can be cancelled as a whole by calling Reset() or
//   cancel MultiDeviceIterator::cancellation_manager().
//   * GetNextFromShard can be cancelled independently. Cancelling
//   GetNextFromShard for one shard doesn't cancel the underlying prefetching,
//   nor does it other calls of GetNextFromShard.
class MultiDeviceIterator : public ResourceBase {
 public:
  MultiDeviceIterator(
      Env* env, const DataTypeVector& output_types,
      const std::vector<PartialTensorShape>& output_shapes,
      const std::vector<string>& devices,
      std::unique_ptr<FunctionLibraryDefinition> flib_def,
      std::unique_ptr<ProcessFunctionLibraryRuntime> pflr,
      FunctionLibraryRuntime* flr,
      std::unique_ptr<FunctionHandleCache> function_handle_cache)
      : unbounded_thread_pool_(env, "tf_data_multi_device_iterator_resource"),
        output_types_(output_types),
        output_shapes_(output_shapes),
        devices_(devices),
        flib_def_(std::move(flib_def)),
        flr_(flr),
        pflr_(std::move(pflr)),
        function_handle_cache_(std::move(function_handle_cache)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc mht_0(mht_0_v, 254, "", "./tensorflow/core/kernels/data/multi_device_iterator_ops.cc", "MultiDeviceIterator");

    DCHECK(flr_ != nullptr);
  }

  string DebugString() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc mht_1(mht_1_v, 261, "", "./tensorflow/core/kernels/data/multi_device_iterator_ops.cc", "DebugString");

    return strings::StrCat("MultiDeviceIterator for ", devices_.size(),
                           " devices");
  }

  Status Init(std::unique_ptr<IteratorBase> iterator, int64_t max_buffer_size,
              int64_t* incarnation_id, DatasetBase* dataset) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc mht_2(mht_2_v, 270, "", "./tensorflow/core/kernels/data/multi_device_iterator_ops.cc", "Init");

    if (iterator) {
      TF_RETURN_IF_ERROR(
          VerifyTypesMatch(output_types_, iterator->output_dtypes()));
      TF_RETURN_IF_ERROR(
          VerifyShapesCompatible(output_shapes_, iterator->output_shapes()));
    }

    mutex_lock l(mu_);
    if (multi_device_buffer_) {
      multi_device_buffer_->Reset();
    }
    dataset->Ref();
    dataset_.reset(dataset);

    ++incarnation_id_;
    *incarnation_id = incarnation_id_;

    multi_device_buffer_ = absl::make_unique<MultiDeviceBuffer>(
        devices_.size(), max_buffer_size, incarnation_id_, std::move(iterator),
        this);
    return Status::OK();
  }

  Status GetNextFromShard(OpKernelContext* ctx, int shard_num,
                          int64_t incarnation_id,
                          MultiDeviceIteratorCallback callback) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc mht_3(mht_3_v, 299, "", "./tensorflow/core/kernels/data/multi_device_iterator_ops.cc", "GetNextFromShard");

    tf_shared_lock l(mu_);
    IteratorContext::Params params(ctx);
    params.flr = flr_;
    params.function_handle_cache = function_handle_cache_.get();
    params.resource_mgr = &resource_mgr_;
    params.thread_factory = unbounded_thread_pool_.get_thread_factory();
    params.thread_pool = &unbounded_thread_pool_;
    params.cancellation_manager = ctx->cancellation_manager();
    IteratorContext iter_ctx(std::move(params));
    multi_device_buffer_->GetNextFromShard(&iter_ctx, shard_num, incarnation_id,
                                           std::move(callback));
    return Status::OK();
  }

  const DataTypeVector& output_types() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc mht_4(mht_4_v, 317, "", "./tensorflow/core/kernels/data/multi_device_iterator_ops.cc", "output_types");
 return output_types_; }

  const std::vector<PartialTensorShape>& output_shapes() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc mht_5(mht_5_v, 322, "", "./tensorflow/core/kernels/data/multi_device_iterator_ops.cc", "output_shapes");

    return output_shapes_;
  }

  FunctionLibraryRuntime* const flr() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc mht_6(mht_6_v, 329, "", "./tensorflow/core/kernels/data/multi_device_iterator_ops.cc", "flr");

    tf_shared_lock l(mu_);
    return flr_;
  }

  FunctionHandleCache* function_handle_cache() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc mht_7(mht_7_v, 337, "", "./tensorflow/core/kernels/data/multi_device_iterator_ops.cc", "function_handle_cache");

    return function_handle_cache_.get();
  }

  ResourceMgr* resource_mgr() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc mht_8(mht_8_v, 344, "", "./tensorflow/core/kernels/data/multi_device_iterator_ops.cc", "resource_mgr");
 return &resource_mgr_; }

  CancellationManager* cancellation_manager() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc mht_9(mht_9_v, 349, "", "./tensorflow/core/kernels/data/multi_device_iterator_ops.cc", "cancellation_manager");
 return &cancellation_manager_; }

 private:
  // A private class that uses a background thread to keep a per device buffer
  // full.
  class MultiDeviceBuffer {
   public:
    MultiDeviceBuffer(size_t size, int64_t max_buffer_size,
                      int64_t incarnation_id,
                      std::unique_ptr<IteratorBase> host_iterator,
                      MultiDeviceIterator* parent)
        : buffer_(size),
          size_(size),
          max_buffer_size_(max_buffer_size),
          incarnation_id_(incarnation_id),
          host_iterator_(std::move(host_iterator)),
          parent_(parent) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc mht_10(mht_10_v, 368, "", "./tensorflow/core/kernels/data/multi_device_iterator_ops.cc", "MultiDeviceBuffer");
}

    ~MultiDeviceBuffer() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc mht_11(mht_11_v, 373, "", "./tensorflow/core/kernels/data/multi_device_iterator_ops.cc", "~MultiDeviceBuffer");

      {
        mutex_lock l(mu_);
        if (!background_thread_started_) return;
      }
      Reset();
    }

    void Reset() TF_LOCKS_EXCLUDED(mu_) {
      {
        mutex_lock l(mu_);
        if (background_thread_ && !background_thread_finished_) {
          cancellation_manager_.StartCancel();
          // Wake up the background thread.
          for (int i = 0; i < size_; ++i) {
            buffer_[i].cond_var.notify_all();
          }

          // Make sure background thread has finished first.
          while (!background_thread_finished_) {
            shutdown_cond_var_.wait(l);
          }
        }
      }
      RunPendingCallbacks();
    }

    void GetNextFromShard(IteratorContext* ctx, int shard_num,
                          int64_t incarnation_id,
                          MultiDeviceIteratorCallback callback) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc mht_12(mht_12_v, 405, "", "./tensorflow/core/kernels/data/multi_device_iterator_ops.cc", "GetNextFromShard");

      HostBufferElement elem;
      if (incarnation_id_ != incarnation_id) {
        elem.status = errors::InvalidArgument(
            "Invalid incarnation id. Provided: ", incarnation_id,
            "; Expected: ", incarnation_id_);
        callback(elem);
        return;
      }

      bool produced_output = false;
      {
        mutex_lock l(mu_);
        if (cancellation_manager_.IsCancelled()) {
          elem.status = errors::Cancelled("Cancelled Multidevice iterator");
          callback(elem);
          return;
        }

        EnsureBackgroundThreadStarted(ctx);

        if (!buffer_[shard_num].data.empty()) {
          produced_output = true;
          std::swap(elem, buffer_[shard_num].data.front());
          buffer_[shard_num].data.pop_front();
          // Wake up background thread if it is blocked on this element.
          if (buffer_[shard_num].data.size() == max_buffer_size_ - 1) {
            buffer_[shard_num].cond_var.notify_all();
          }
        } else {
          if (end_of_iterator_) {
            produced_output = true;
            elem.end_of_sequence = true;
          } else {
            auto callback_container =
                std::make_shared<HostBuffer::CallbackContainer>(
                    std::move(callback));
            elem.status = RegisterCancellationCallback(
                ctx->cancellation_manager(),
                [callback_container]() {
                  if (callback_container->is_called.exchange(true)) {
                    return;
                  }
                  HostBufferElement elem;
                  elem.status =
                      errors::Cancelled("GetNextFromShard was cancelled");
                  callback_container->callback(elem);
                },
                &callback_container->deregister_cancellation);
            if (!elem.status.ok()) {
              callback_container->callback(elem);
              return;
            }
            buffer_[shard_num].callbacks.push_back(
                std::move(callback_container));
            buffer_[shard_num].cond_var.notify_all();
            callback = nullptr;
          }
        }
      }

      if (produced_output) {
        callback(elem);
      }
    }

   private:
    void EnsureBackgroundThreadStarted(IteratorContext* ctx)
        TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc mht_13(mht_13_v, 476, "", "./tensorflow/core/kernels/data/multi_device_iterator_ops.cc", "EnsureBackgroundThreadStarted");

      if (!background_thread_) {
        IteratorContext::Params params(ctx);
        params.cancellation_manager = &cancellation_manager_;
        background_thread_ =
            parent_->unbounded_thread_pool_.get_thread_factory()->StartThread(
                "tf_data_multi_device_iterator",
                std::bind(
                    &MultiDeviceIterator::MultiDeviceBuffer::BackgroundThread,
                    this,
                    std::make_shared<IteratorContext>(std::move(params))));
      }
    }

    void RunPendingCallbacks() TF_LOCKS_EXCLUDED(mu_) {
      // Run all remaining callbacks.

      std::vector<std::shared_ptr<HostBuffer::CallbackContainer>>
          callback_containers;
      std::vector<HostBufferElement> cancellation_elements;
      {
        mutex_lock l(mu_);

        for (int i = 0; i < size_; ++i) {
          while (!buffer_[i].callbacks.empty()) {
            if (buffer_[i].callbacks.front()->is_called.exchange(true)) {
              buffer_[i].callbacks.pop_front();
              continue;
            }
            if (buffer_[i].data.empty()) {
              HostBufferElement elem;
              if (end_of_iterator_) {
                elem.end_of_sequence = true;
              } else {
                elem.status =
                    errors::Cancelled("Cancelled and buffer not filled.");
              }
              cancellation_elements.push_back(std::move(elem));
            } else {
              cancellation_elements.push_back(
                  std::move(buffer_[i].data.front()));
              buffer_[i].data.pop_front();
            }
            callback_containers.push_back(
                std::move(buffer_[i].callbacks.front()));
            buffer_[i].callbacks.pop_front();
          }
        }
      }
      for (int i = 0; i < callback_containers.size(); ++i) {
        if (callback_containers[i]->deregister_cancellation != nullptr) {
          callback_containers[i]->deregister_cancellation();
        }
        // We invoke the callback regardless of whether deregistration succeeds
        // or not, because we have set is_called=true previous which effectively
        // disables the cancellation callback.
        callback_containers[i]->callback(cancellation_elements[i]);
      }
    }

    void BackgroundThread(std::shared_ptr<IteratorContext> ctx) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc mht_14(mht_14_v, 539, "", "./tensorflow/core/kernels/data/multi_device_iterator_ops.cc", "BackgroundThread");

      {
        mutex_lock l(mu_);
        background_thread_started_ = true;
      }
      int shard_to_fetch = 0;
      while (true) {
        HostBufferElement elem;
        bool end_of_iterator = false;

        {
          mutex_lock l(mu_);
          while (!cancellation_manager_.IsCancelled() &&
                 buffer_[shard_to_fetch].data.size() >= max_buffer_size_ &&
                 buffer_[shard_to_fetch].callbacks.empty()) {
            buffer_[shard_to_fetch].cond_var.wait(l);
          }

          if (cancellation_manager_.IsCancelled()) {
            background_thread_finished_ = true;
            shutdown_cond_var_.notify_all();
            return;
          }
        }

        elem.status = host_iterator_->GetNext(ctx.get(), &elem.value,
                                              &elem.end_of_sequence);

        if (elem.status.ok() && elem.end_of_sequence) {
          end_of_iterator = true;
        }

        std::shared_ptr<HostBuffer::CallbackContainer> callback_container;
        {
          mutex_lock l(mu_);
          // Try to find a callback, else just push stuff into buffer.
          if (!buffer_[shard_to_fetch].callbacks.empty()) {
            while (!buffer_[shard_to_fetch].callbacks.empty()) {
              if (buffer_[shard_to_fetch].callbacks.front()->is_called.exchange(
                      true)) {
                // This callback is already cancelled.
                buffer_[shard_to_fetch].callbacks.pop_front();
                continue;
              } else {
                callback_container =
                    std::move(buffer_[shard_to_fetch].callbacks.front());
                buffer_[shard_to_fetch].callbacks.pop_front();
                break;
              }
            }
          } else {
            buffer_[shard_to_fetch].data.push_back(std::move(elem));
            elem = HostBufferElement();
          }
        }

        if (callback_container) {
          if (callback_container->deregister_cancellation != nullptr) {
            callback_container->deregister_cancellation();
          }
          (*ctx->runner())(std::bind(std::move(callback_container->callback),
                                     std::move(elem)));
        }

        // Finish off the thread if we reach the end of the iterator. Runs
        // pending callbacks.
        if (end_of_iterator) {
          {
            mutex_lock l(mu_);
            background_thread_finished_ = true;
            end_of_iterator_ = true;
            shutdown_cond_var_.notify_all();
          }
          RunPendingCallbacks();
          return;
        }
        shard_to_fetch = (shard_to_fetch + 1) % size_;
      }
    }

    struct HostBuffer {
      condition_variable cond_var;
      std::deque<HostBufferElement> data;
      struct CallbackContainer {
        MultiDeviceIteratorCallback callback;
        // Whether callback is already called, either by the background thread
        // of by the cancellation callback.
        std::atomic<bool> is_called;
        std::function<void()> deregister_cancellation;
        explicit CallbackContainer(MultiDeviceIteratorCallback&& callback)
            : callback(std::move(callback)), is_called(false) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc mht_15(mht_15_v, 632, "", "./tensorflow/core/kernels/data/multi_device_iterator_ops.cc", "CallbackContainer");
}
      };
      // The CallbackContainer is shared with the cancellation callback.
      std::deque<std::shared_ptr<CallbackContainer>> callbacks;
    };

    mutex mu_;
    std::unique_ptr<Thread> background_thread_ TF_GUARDED_BY(mu_);
    bool background_thread_finished_ TF_GUARDED_BY(mu_) = false;
    bool background_thread_started_ TF_GUARDED_BY(mu_) = false;
    bool end_of_iterator_ TF_GUARDED_BY(mu_) = false;
    condition_variable shutdown_cond_var_ TF_GUARDED_BY(mu_);

    std::vector<HostBuffer> buffer_;

    const size_t size_;
    const int64_t max_buffer_size_;
    const int64_t incarnation_id_;
    const std::unique_ptr<IteratorBase> host_iterator_;
    CancellationManager cancellation_manager_;
    MultiDeviceIterator* const parent_;  // Not owned.
  };

  UnboundedThreadPool unbounded_thread_pool_;
  mutex mu_;
  const DataTypeVector output_types_;
  const std::vector<PartialTensorShape> output_shapes_;
  const std::vector<string> devices_;
  const std::unique_ptr<FunctionLibraryDefinition> flib_def_;
  FunctionLibraryRuntime* const flr_ = nullptr;  // not owned.
  const std::unique_ptr<ProcessFunctionLibraryRuntime> pflr_;
  const std::unique_ptr<FunctionHandleCache> function_handle_cache_;
  ResourceMgr resource_mgr_;
  CancellationManager cancellation_manager_;

  int64_t incarnation_id_ TF_GUARDED_BY(mu_) = 0;
  std::unique_ptr<MultiDeviceBuffer> multi_device_buffer_ TF_GUARDED_BY(mu_);
  core::RefCountPtr<DatasetBase> dataset_;
};

// Used to generate unique names for anonymous multi device iterators.
static std::atomic<int64_t> current_id_;

// Just creates a MultiDeviceIterator and returns it.
class MultiDeviceIteratorHandleOp : public OpKernel {
 public:
  explicit MultiDeviceIteratorHandleOp(OpKernelConstruction* ctx)
      : OpKernel(ctx), graph_def_version_(ctx->graph_def_version()) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc mht_16(mht_16_v, 682, "", "./tensorflow/core/kernels/data/multi_device_iterator_ops.cc", "MultiDeviceIteratorHandleOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("shared_name", &name_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("container", &container_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kDevices, &devices_));
  }

  // The resource is deleted from the resource manager only when it is private
  // to kernel.
  ~MultiDeviceIteratorHandleOp() override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc mht_17(mht_17_v, 695, "", "./tensorflow/core/kernels/data/multi_device_iterator_ops.cc", "~MultiDeviceIteratorHandleOp");

    if (resource_ != nullptr) {
      resource_->Unref();
      if (cinfo_.resource_is_private_to_kernel()) {
        if (!cinfo_.resource_manager()
                 ->template Delete<MultiDeviceIterator>(cinfo_.container(),
                                                        cinfo_.name())
                 .ok()) {
          // Do nothing; the resource can have been deleted by session resets.
        }
      }
    }
  }

  void Compute(OpKernelContext* context) override TF_LOCKS_EXCLUDED(mu_) {
    string unique_name = cinfo_.name();
    string container_name = cinfo_.container();
    {
      mutex_lock l(mu_);
      if (resource_ == nullptr) {
        FunctionLibraryRuntime* flr;
        std::unique_ptr<FunctionLibraryDefinition> flib_def(nullptr);
        std::unique_ptr<ProcessFunctionLibraryRuntime> pflr(nullptr);
        OP_REQUIRES_OK(context, context->function_library()->Clone(
                                    &flib_def, &pflr, &flr));
        auto function_handle_cache =
            absl::make_unique<FunctionHandleCache>(flr);
        ResourceMgr* mgr = context->resource_manager();
        OP_REQUIRES_OK(context, cinfo_.Init(mgr, def()));

        MultiDeviceIterator* resource;

        if (name_ == ResourceHandle::ANONYMOUS_NAME) {
          unique_name = strings::StrCat("_AnonymousMultiDeviceIterator",
                                        current_id_.fetch_add(1));
          container_name = kAnonymousMultiDeviceIterator;
          resource = new MultiDeviceIterator(
              context->env(), output_types_, output_shapes_, devices_,
              std::move(flib_def), std::move(pflr), flr,
              std::move(function_handle_cache));
          // NOTE: `mgr->Create()` transfers the one reference on `resource` to
          // `mgr`.
          OP_REQUIRES_OK(context, mgr->Create<MultiDeviceIterator>(
                                      container_name, unique_name, resource));
        } else {
          unique_name = cinfo_.name();
          container_name = cinfo_.container();
          OP_REQUIRES_OK(context,
                         mgr->LookupOrCreate<MultiDeviceIterator>(
                             container_name, unique_name, &resource,
                             [this, context, flr, &flib_def, &pflr,
                              &function_handle_cache](MultiDeviceIterator** ret)
                                 TF_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
                                   *ret = new MultiDeviceIterator(
                                       context->env(), output_types_,
                                       output_shapes_, devices_,
                                       std::move(flib_def), std::move(pflr),
                                       flr, std::move(function_handle_cache));
                                   return Status::OK();
                                 }));
          Status s = VerifyResource(resource);
          if (TF_PREDICT_FALSE(!s.ok())) {
            resource->Unref();
            context->SetStatus(s);
            return;
          }
          resource_ = resource;
        }
      }
    }
    OP_REQUIRES_OK(context, MakeResourceHandleToOutput(
                                context, 0, container_name, unique_name,
                                TypeIndex::Make<MultiDeviceIterator>()));
  }

 private:
  // During the first Compute(), resource is either created or looked up using
  // shared_name. In the latter case, the resource found should be verified if
  // it is compatible with this op's configuration. The verification may fail in
  // cases such as two graphs asking queues of the same shared name to have
  // inconsistent capacities.
  Status VerifyResource(MultiDeviceIterator* resource) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc mht_18(mht_18_v, 779, "", "./tensorflow/core/kernels/data/multi_device_iterator_ops.cc", "VerifyResource");

    TF_RETURN_IF_ERROR(
        VerifyTypesMatch(output_types_, resource->output_types()));
    TF_RETURN_IF_ERROR(
        VerifyShapesCompatible(output_shapes_, resource->output_shapes()));
    return Status::OK();
  }

  mutex mu_;
  ContainerInfo cinfo_;  // Written once under mu_ then constant afterwards.
  MultiDeviceIterator* resource_ TF_GUARDED_BY(mu_) = nullptr;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  const int graph_def_version_;
  string name_;
  string container_;
  std::vector<string> devices_;
};

REGISTER_KERNEL_BUILDER(Name("MultiDeviceIterator").Device(DEVICE_CPU),
                        MultiDeviceIteratorHandleOp);

class AnonymousMultiDeviceIteratorOp
    : public AnonymousResourceOp<MultiDeviceIterator> {
 public:
  explicit AnonymousMultiDeviceIteratorOp(OpKernelConstruction* ctx)
      : AnonymousResourceOp<MultiDeviceIterator>(
            ctx,
            /* ref_counting */ true,
            /* Only V1 returns a deleter */
            /* return_deleter */
            ctx->def().op() == kAnonymousMultiDeviceIterator) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc mht_19(mht_19_v, 813, "", "./tensorflow/core/kernels/data/multi_device_iterator_ops.cc", "AnonymousMultiDeviceIteratorOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr(kDevices, &devices_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_dtypes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
  }

 private:
  string name() override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc mht_20(mht_20_v, 823, "", "./tensorflow/core/kernels/data/multi_device_iterator_ops.cc", "name");
 return kAnonymousMultiDeviceIterator; }

  Status CreateResource(OpKernelContext* ctx,
                        std::unique_ptr<FunctionLibraryDefinition> flib_def,
                        std::unique_ptr<ProcessFunctionLibraryRuntime> pflr,
                        FunctionLibraryRuntime* lib,
                        MultiDeviceIterator** resource) override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc mht_21(mht_21_v, 832, "", "./tensorflow/core/kernels/data/multi_device_iterator_ops.cc", "CreateResource");

    auto function_handle_cache = absl::make_unique<FunctionHandleCache>(lib);
    *resource =
        new MultiDeviceIterator(ctx->env(), output_dtypes_, output_shapes_,
                                devices_, std::move(flib_def), std::move(pflr),
                                lib, std::move(function_handle_cache));
    return Status::OK();
  }

  std::vector<string> devices_;
  DataTypeVector output_dtypes_;
  std::vector<PartialTensorShape> output_shapes_;
};

REGISTER_KERNEL_BUILDER(Name(kAnonymousMultiDeviceIterator).Device(DEVICE_CPU),
                        AnonymousMultiDeviceIteratorOp);
REGISTER_KERNEL_BUILDER(
    Name(kAnonymousMultiDeviceIteratorV3).Device(DEVICE_CPU),
    AnonymousMultiDeviceIteratorOp);

// Calls init on the MultiDeviceIterator.
class MultiDeviceIteratorInitOp : public OpKernel {
 public:
  explicit MultiDeviceIteratorInitOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc mht_22(mht_22_v, 859, "", "./tensorflow/core/kernels/data/multi_device_iterator_ops.cc", "MultiDeviceIteratorInitOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc mht_23(mht_23_v, 864, "", "./tensorflow/core/kernels/data/multi_device_iterator_ops.cc", "Compute");

    const Tensor* tensor_max_buffer_size;
    OP_REQUIRES_OK(ctx, ctx->input("max_buffer_size", &tensor_max_buffer_size));
    int64_t max_buffer_size = tensor_max_buffer_size->scalar<int64_t>()();

    DatasetBase* dataset;
    OP_REQUIRES_OK(ctx, GetDatasetFromVariantTensor(ctx->input(0), &dataset));
    core::RefCountPtr<MultiDeviceIterator> resource;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 1), &resource));

    IteratorContext::Params params(ctx);
    params.flr = resource->flr();
    params.function_handle_cache = resource->function_handle_cache();
    params.resource_mgr = resource->resource_mgr();
    params.cancellation_manager = resource->cancellation_manager();
    std::function<void()> deregister_fn;
    OP_REQUIRES_OK(
        ctx, RegisterCancellationCallback(
                 ctx->cancellation_manager(),
                 [cm = params.cancellation_manager]() { cm->StartCancel(); },
                 &deregister_fn));
    auto cleanup = gtl::MakeCleanup(std::move(deregister_fn));
    IteratorContext iter_ctx(std::move(params));

    std::unique_ptr<IteratorBase> iterator;
    DatasetBase* finalized_dataset;
    OP_REQUIRES_OK(ctx, FinalizeDataset(ctx, dataset, &finalized_dataset));
    OP_REQUIRES_OK(ctx, finalized_dataset->MakeIterator(std::move(iter_ctx),
                                                        /*parent=*/nullptr,
                                                        "Iterator", &iterator));
    core::ScopedUnref unref(finalized_dataset);
    int64_t incarnation_id;
    OP_REQUIRES_OK(ctx, resource->Init(std::move(iterator), max_buffer_size,
                                       &incarnation_id, dataset));
    Tensor tensor_incarnation_id(DT_INT64, TensorShape({}));
    tensor_incarnation_id.scalar<int64_t>()() = incarnation_id;
    OP_REQUIRES_OK(ctx,
                   ctx->set_output("incarnation_id", tensor_incarnation_id));
  }
};

REGISTER_KERNEL_BUILDER(Name("MultiDeviceIteratorInit").Device(DEVICE_CPU),
                        MultiDeviceIteratorInitOp);

// Calls GetNextFromShard(shard) and returns a vector of Tensors as output.
class MultiDeviceIteratorGetNextFromShardOp : public AsyncOpKernel {
 public:
  explicit MultiDeviceIteratorGetNextFromShardOp(OpKernelConstruction* ctx)
      : AsyncOpKernel(ctx),
        background_worker_(ctx->env(),
                           "tf_data_multi_device_iterator_get_next") {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc mht_24(mht_24_v, 918, "", "./tensorflow/core/kernels/data/multi_device_iterator_ops.cc", "MultiDeviceIteratorGetNextFromShardOp");
}

  void ComputeAsync(OpKernelContext* ctx, DoneCallback done) override {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc mht_25(mht_25_v, 923, "", "./tensorflow/core/kernels/data/multi_device_iterator_ops.cc", "ComputeAsync");

    const Tensor* tensor_shard_num;
    OP_REQUIRES_OK_ASYNC(ctx, ctx->input("shard_num", &tensor_shard_num), done);
    int32_t shard_num = tensor_shard_num->scalar<int32>()();

    const Tensor* tensor_incarnation_id;
    OP_REQUIRES_OK_ASYNC(
        ctx, ctx->input("incarnation_id", &tensor_incarnation_id), done);
    int64_t incarnation_id = tensor_incarnation_id->scalar<int64_t>()();

    MultiDeviceIterator* iterator;
    OP_REQUIRES_OK_ASYNC(
        ctx, LookupResource(ctx, HandleFromInput(ctx, 0), &iterator), done);

    background_worker_.Schedule(std::bind(
        [ctx, iterator, shard_num, incarnation_id](DoneCallback done) {
          Notification n;
          MultiDeviceIteratorCallback callback = std::bind(
              [ctx, &n](const HostBufferElement& elem) {
                Status s = elem.status;
                if (!s.ok()) {
                  ctx->SetStatus(s);
                } else if (elem.end_of_sequence) {
                  ctx->SetStatus(errors::OutOfRange("End of sequence"));
                } else {
                  for (int i = 0; i < elem.value.size(); ++i) {
                    ctx->set_output(i, elem.value[i]);
                  }
                }
                n.Notify();
              },
              std::placeholders::_1);

          Status s = iterator->GetNextFromShard(ctx, shard_num, incarnation_id,
                                                std::move(callback));
          if (!s.ok()) {
            ctx->SetStatus(s);
            iterator->Unref();
            done();
            return;
          }
          iterator->Unref();
          n.WaitForNotification();
          done();
        },
        std::move(done)));
  }

 private:
  BackgroundWorker background_worker_;
};

REGISTER_KERNEL_BUILDER(
    Name("MultiDeviceIteratorGetNextFromShard").Device(DEVICE_CPU),
    MultiDeviceIteratorGetNextFromShardOp);

class MultiDeviceIteratorToStringHandleOp : public OpKernel {
 public:
  explicit MultiDeviceIteratorToStringHandleOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc mht_26(mht_26_v, 985, "", "./tensorflow/core/kernels/data/multi_device_iterator_ops.cc", "MultiDeviceIteratorToStringHandleOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc mht_27(mht_27_v, 990, "", "./tensorflow/core/kernels/data/multi_device_iterator_ops.cc", "Compute");

    const Tensor& resource_handle_t = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(resource_handle_t.shape()),
                errors::InvalidArgument("resource_handle must be a scalar"));

    // Validate that the handle corresponds to a real resource, and
    // that it is an MultiDeviceIterator.
    core::RefCountPtr<MultiDeviceIterator> resource;
    OP_REQUIRES_OK(ctx,
                   LookupResource(ctx, HandleFromInput(ctx, 0), &resource));

    Tensor* string_handle_t;
    OP_REQUIRES_OK(ctx,
                   ctx->allocate_output(0, TensorShape({}), &string_handle_t));
    string_handle_t->scalar<tstring>()() =
        resource_handle_t.scalar<ResourceHandle>()().SerializeAsString();
  }
};

REGISTER_KERNEL_BUILDER(
    Name("MultiDeviceIteratorToStringHandle").Device(DEVICE_CPU),
    MultiDeviceIteratorToStringHandleOp);

class MultiDeviceIteratorFromStringHandleOp : public OpKernel {
 public:
  explicit MultiDeviceIteratorFromStringHandleOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc mht_28(mht_28_v, 1019, "", "./tensorflow/core/kernels/data/multi_device_iterator_ops.cc", "MultiDeviceIteratorFromStringHandleOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputTypes, &output_types_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr(kOutputShapes, &output_shapes_));
    OP_REQUIRES(
        ctx,
        output_types_.empty() || output_shapes_.empty() ||
            output_types_.size() == output_shapes_.size(),
        errors::InvalidArgument("If both 'output_types' and 'output_shapes' "
                                "are set, they must have the same length."));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc mht_29(mht_29_v, 1033, "", "./tensorflow/core/kernels/data/multi_device_iterator_ops.cc", "Compute");

    const Tensor& string_handle_t = ctx->input(0);
    OP_REQUIRES(ctx, TensorShapeUtils::IsScalar(string_handle_t.shape()),
                errors::InvalidArgument("string_handle must be a scalar"));

    ResourceHandle resource_handle;
    OP_REQUIRES(
        ctx,
        resource_handle.ParseFromString(string_handle_t.scalar<tstring>()()),
        errors::InvalidArgument(
            "Could not parse string_handle as a valid ResourceHandle"));

    OP_REQUIRES(
        ctx, resource_handle.device() == ctx->device()->attributes().name(),
        errors::InvalidArgument("Attempted create an iterator on device \"",
                                ctx->device()->attributes().name(),
                                "\" from handle defined on device \"",
                                resource_handle.device(), "\""));

    // Validate that the handle corresponds to a real resource, and
    // that it is an MultiDeviceIterator.
    core::RefCountPtr<MultiDeviceIterator> resource;
    OP_REQUIRES_OK(ctx, LookupResource(ctx, resource_handle, &resource));
    if (!output_types_.empty()) {
      OP_REQUIRES_OK(ctx,
                     VerifyTypesMatch(output_types_, resource->output_types()));
    }
    if (!output_shapes_.empty()) {
      OP_REQUIRES_OK(ctx, VerifyShapesCompatible(output_shapes_,
                                                 resource->output_shapes()));
    }

    Tensor* resource_handle_t;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_output(0, TensorShape({}), &resource_handle_t));
    resource_handle_t->scalar<ResourceHandle>()() = resource_handle;
  }

 private:
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

REGISTER_KERNEL_BUILDER(
    Name("MultiDeviceIteratorFromStringHandle").Device(DEVICE_CPU),
    MultiDeviceIteratorFromStringHandleOp);

class DeleteMultiDeviceIteratorOp : public OpKernel {
 public:
  explicit DeleteMultiDeviceIteratorOp(OpKernelConstruction* ctx)
      : OpKernel(ctx) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc mht_30(mht_30_v, 1086, "", "./tensorflow/core/kernels/data/multi_device_iterator_ops.cc", "DeleteMultiDeviceIteratorOp");
}

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSdataPSmulti_device_iterator_opsDTcc mht_31(mht_31_v, 1091, "", "./tensorflow/core/kernels/data/multi_device_iterator_ops.cc", "Compute");

    ResourceHandle handle = ctx->input(0).flat<ResourceHandle>()(0);
    // The iterator resource is guaranteed to
    // exist because the variant tensor wrapping the deleter is provided as an
    // unused input to this op, which guarantees that it has not run yet.
    OP_REQUIRES_OK(ctx, DeleteResource(ctx, handle));
  }
};

REGISTER_KERNEL_BUILDER(Name("DeleteMultiDeviceIterator").Device(DEVICE_CPU),
                        DeleteMultiDeviceIteratorOp);
// Since this op takes in Iterator handles as (unused) inputs, we don't want
// to constrain the iterator location to CPU only. Therefore, we exempt the
// colocation restriction for this op allowing the iterators to be placed on
// other devices.
REGISTER_INPUT_COLOCATION_EXEMPTION("DeleteMultiDeviceIterator");

}  // namespace
}  // namespace data
}  // namespace tensorflow
