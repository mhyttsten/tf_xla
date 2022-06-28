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
class MHTracer_DTPStensorflowPScorePSkernelsPSmutex_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSmutex_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmutex_opsDTcc() {
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

#define EIGEN_USE_THREADS

#include <deque>
#include <utility>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shared_ptr_variant.h"
#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {

class Mutex : public ResourceBase {
 public:
  explicit Mutex(OpKernelContext* c, const string& name)
      : locked_(false),
        thread_pool_(new thread::ThreadPool(
            c->env(), ThreadOptions(),
            strings::StrCat("mutex_lock_thread_", SanitizeThreadSuffix(name)),
            1 /* num_threads */, false /* low_latency_hint */)),
        name_(name) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPSmutex_opsDTcc mht_0(mht_0_v, 215, "", "./tensorflow/core/kernels/mutex_ops.cc", "Mutex");

    VLOG(2) << "Creating mutex with name " << name << ": " << this;
  }

  string DebugString() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmutex_opsDTcc mht_1(mht_1_v, 222, "", "./tensorflow/core/kernels/mutex_ops.cc", "DebugString");

    return strings::StrCat("Mutex ", name_);
  }

  class LockReleaser {
   public:
    explicit LockReleaser(Mutex* mutex) : mutex_(mutex) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmutex_opsDTcc mht_2(mht_2_v, 231, "", "./tensorflow/core/kernels/mutex_ops.cc", "LockReleaser");
}

    LockReleaser(const LockReleaser&) = delete;
    LockReleaser& operator=(const LockReleaser&) = delete;

    virtual ~LockReleaser() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmutex_opsDTcc mht_3(mht_3_v, 239, "", "./tensorflow/core/kernels/mutex_ops.cc", "~LockReleaser");

      VLOG(3) << "Destroying LockReleaser " << this << " for mutex: " << mutex_;
      if (mutex_) {
        mutex_lock lock(mutex_->mu_);
        mutex_->locked_ = false;
        mutex_->cv_.notify_all();
        VLOG(3) << "Destroying LockReleaser " << this
                << ": sent notifications.";
      }
    }

   private:
    Mutex* mutex_;
  };

  typedef SharedPtrVariant<LockReleaser> SharedLockReleaser;

  void AcquireAsync(
      OpKernelContext* c,
      std::function<void(const Status& s, SharedLockReleaser lock)> fn) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmutex_opsDTcc mht_4(mht_4_v, 261, "", "./tensorflow/core/kernels/mutex_ops.cc", "AcquireAsync");

    CancellationManager* cm = c->cancellation_manager();
    CancellationToken token{};
    bool* cancelled = nullptr;
    if (cm) {
      cancelled = new bool(false);  // TF_GUARDED_BY(mu_);
      token = cm->get_cancellation_token();
      const bool already_cancelled =
          !cm->RegisterCallback(token, [this, cancelled]() {
            mutex_lock lock(mu_);
            *cancelled = true;
            cv_.notify_all();
          });
      if (already_cancelled) {
        delete cancelled;
        fn(errors::Cancelled("Lock acquisition cancelled."),
           SharedLockReleaser{nullptr});
        return;
      }
    }
    thread_pool_->Schedule(std::bind(
        [this, cm, cancelled,
         token](std::function<void(const Status& s, SharedLockReleaser&& lock)>
                    fn_) {
          bool local_locked;
          {
            mutex_lock lock(mu_);
            while (locked_ && !(cancelled && *cancelled)) {
              cv_.wait(lock);
            }
            local_locked = locked_ = !(cancelled && *cancelled);
          }
          if (cm) {
            cm->DeregisterCallback(token);
            delete cancelled;
          }
          if (local_locked) {  // Not cancelled.
            fn_(Status::OK(),
                SharedLockReleaser{std::make_shared<LockReleaser>(this)});
          } else {
            fn_(errors::Cancelled("Lock acquisition cancelled."),
                SharedLockReleaser{nullptr});
          }
        },
        std::move(fn)));
  }

 private:
  mutex mu_;
  condition_variable cv_ TF_GUARDED_BY(mu_);
  bool locked_ TF_GUARDED_BY(mu_);
  std::unique_ptr<thread::ThreadPool> thread_pool_;
  string name_;
};

}  // namespace

class MutexLockOp : public AsyncOpKernel {
 public:
  explicit MutexLockOp(OpKernelConstruction* c) : AsyncOpKernel(c) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmutex_opsDTcc mht_5(mht_5_v, 323, "", "./tensorflow/core/kernels/mutex_ops.cc", "MutexLockOp");
}

 public:
  void ComputeAsync(OpKernelContext* c, DoneCallback done) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmutex_opsDTcc mht_6(mht_6_v, 329, "", "./tensorflow/core/kernels/mutex_ops.cc", "ComputeAsync");

    Mutex* mutex = nullptr;
    OP_REQUIRES_OK_ASYNC(
        c,
        LookupOrCreateResource<Mutex>(c, HandleFromInput(c, 0), &mutex,
                                      [c](Mutex** ptr) {
                                        *ptr = new Mutex(
                                            c, HandleFromInput(c, 0).name());
                                        return Status::OK();
                                      }),
        done);

    Tensor* variant;
    OP_REQUIRES_OK_ASYNC(c, c->allocate_output(0, TensorShape({}), &variant),
                         done);

    mutex->AcquireAsync(
        c, std::bind(
               [c, variant, mutex](DoneCallback done_,
                                   // End of bound arguments.
                                   const Status& s,
                                   Mutex::SharedLockReleaser&& lock) {
                 VLOG(2) << "Finished locking mutex " << mutex
                         << " with lock: " << lock.shared_ptr.get()
                         << " status: " << s.ToString();
                 if (s.ok()) {
                   variant->scalar<Variant>()() = std::move(lock);
                 } else {
                   c->SetStatus(s);
                 }
                 mutex->Unref();
                 done_();
               },
               std::move(done), std::placeholders::_1, std::placeholders::_2));
  }
};

class ConsumeMutexLockOp : public OpKernel {
 public:
  explicit ConsumeMutexLockOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmutex_opsDTcc mht_7(mht_7_v, 372, "", "./tensorflow/core/kernels/mutex_ops.cc", "ConsumeMutexLockOp");
}

  void Compute(OpKernelContext* c) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmutex_opsDTcc mht_8(mht_8_v, 377, "", "./tensorflow/core/kernels/mutex_ops.cc", "Compute");

    VLOG(2) << "Executing ConsumeMutexLockOp";
    const Tensor& lock_t = c->input(0);
    OP_REQUIRES(
        c, lock_t.dims() == 0,
        errors::InvalidArgument("Expected input to be a scalar, saw shape: ",
                                lock_t.shape().DebugString()));
    OP_REQUIRES(
        c, lock_t.dtype() == DT_VARIANT,
        errors::InvalidArgument("Expected input to be a variant, saw type: ",
                                DataTypeString(lock_t.dtype())));
    const auto* lock =
        lock_t.scalar<Variant>()().get<Mutex::SharedLockReleaser>();
    OP_REQUIRES(c, lock,
                errors::InvalidArgument(
                    "Expected input to contain a SharedLockReleaser "
                    "object, but saw variant: '",
                    lock_t.scalar<Variant>()().DebugString(), "'"));
    const int use_count = lock->shared_ptr.use_count();
    OP_REQUIRES(
        c, use_count == 1,
        errors::InvalidArgument("Expected use count of lock to be 1, but saw: ",
                                use_count));
  }

  bool IsExpensive() override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmutex_opsDTcc mht_9(mht_9_v, 405, "", "./tensorflow/core/kernels/mutex_ops.cc", "IsExpensive");
 return false; }
};

REGISTER_KERNEL_BUILDER(Name("MutexLock").Device(DEVICE_CPU), MutexLockOp);

REGISTER_KERNEL_BUILDER(Name("MutexLock")
                            .Device(DEVICE_DEFAULT)
                            .HostMemory("mutex_lock")
                            .HostMemory("mutex"),
                        MutexLockOp);

REGISTER_KERNEL_BUILDER(
    Name("MutexV2").Device(DEVICE_CPU).HostMemory("resource"),
    ResourceHandleOp<Mutex>);

REGISTER_KERNEL_BUILDER(Name("MutexV2").Device(DEVICE_DEFAULT),
                        ResourceHandleOp<Mutex>);

REGISTER_KERNEL_BUILDER(Name("ConsumeMutexLock").Device(DEVICE_CPU),
                        ConsumeMutexLockOp);

REGISTER_KERNEL_BUILDER(
    Name("ConsumeMutexLock").Device(DEVICE_DEFAULT).HostMemory("mutex_lock"),
    ConsumeMutexLockOp);

}  // namespace tensorflow
