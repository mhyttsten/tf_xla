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
class MHTracer_DTPStensorflowPSlitePSkernelsPSeigen_supportDTcc {
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
   MHTracer_DTPStensorflowPSlitePSkernelsPSeigen_supportDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSlitePSkernelsPSeigen_supportDTcc() {
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
#include "tensorflow/lite/kernels/eigen_support.h"

#include <functional>
#include <memory>
#include <utility>

#include "tensorflow/lite/arena_planner.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/eigen_spatial_convolutions.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace eigen_support {
namespace {

// For legacy reasons, we use 4 threads by default unless the thread count is
// explicitly specified by the context.
const int kDefaultNumThreadpoolThreads = 4;

bool IsValidNumThreads(int num_threads) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSeigen_supportDTcc mht_0(mht_0_v, 203, "", "./tensorflow/lite/kernels/eigen_support.cc", "IsValidNumThreads");
 return num_threads >= -1; }
int GetNumThreads(int num_threads) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSeigen_supportDTcc mht_1(mht_1_v, 207, "", "./tensorflow/lite/kernels/eigen_support.cc", "GetNumThreads");

  return num_threads > -1 ? num_threads : kDefaultNumThreadpoolThreads;
}

#ifndef EIGEN_DONT_ALIGN
// Eigen may require buffers to be aligned to 16, 32 or 64 bytes depending on
// hardware architecture and build configurations.
// If the static assertion fails, try to increase `kDefaultTensorAlignment` to
// in `arena_planner.h` to 32 or 64.
static_assert(
    kDefaultTensorAlignment % EIGEN_MAX_ALIGN_BYTES == 0,
    "kDefaultArenaAlignment doesn't comply with Eigen alignment requirement.");
#endif  // EIGEN_DONT_ALIGN

// Helper routine for updating the global Eigen thread count used for OpenMP.
void SetEigenNbThreads(int threads) {
#if defined(EIGEN_HAS_OPENMP)
  // The global Eigen thread count is only used when OpenMP is enabled. As this
  // call causes problems with tsan, make it only when OpenMP is available.
  Eigen::setNbThreads(threads);
#endif  // defined(EIGEN_HAS_OPENMP)
}

// We have a single global threadpool for all convolution operations. This means
// that inferences started from different threads may block each other, but
// since the underlying resource of CPU cores should be consumed by the
// operations anyway, it shouldn't affect overall performance. Note that we
// also avoid ThreadPool creation if the target thread count is 1, avoiding
// unnecessary overhead, and more closely mimicking Gemmlowp threadpool
// behavior.
class EigenThreadPoolWrapper : public Eigen::ThreadPoolInterface {
 public:
  // Takes ownership of 'pool'
  explicit EigenThreadPoolWrapper(int num_threads) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSeigen_supportDTcc mht_2(mht_2_v, 243, "", "./tensorflow/lite/kernels/eigen_support.cc", "EigenThreadPoolWrapper");

    // Avoid creating any threads for the single-threaded case.
    if (num_threads > 1) {
      pool_.reset(new Eigen::ThreadPool(num_threads));
    }
  }
  ~EigenThreadPoolWrapper() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSeigen_supportDTcc mht_3(mht_3_v, 252, "", "./tensorflow/lite/kernels/eigen_support.cc", "~EigenThreadPoolWrapper");
}

  void Schedule(std::function<void()> fn) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSeigen_supportDTcc mht_4(mht_4_v, 257, "", "./tensorflow/lite/kernels/eigen_support.cc", "Schedule");

    if (pool_) {
      pool_->Schedule(std::move(fn));
    } else {
      fn();
    }
  }
  int NumThreads() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSeigen_supportDTcc mht_5(mht_5_v, 267, "", "./tensorflow/lite/kernels/eigen_support.cc", "NumThreads");
 return pool_ ? pool_->NumThreads() : 1; }
  int CurrentThreadId() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSeigen_supportDTcc mht_6(mht_6_v, 271, "", "./tensorflow/lite/kernels/eigen_support.cc", "CurrentThreadId");

    return pool_ ? pool_->CurrentThreadId() : 0;
  }

 private:
  // May be null if num_threads <= 1.
  std::unique_ptr<Eigen::ThreadPool> pool_;
};

// Utility class for lazily creating an Eigen thread pool/device only when used.
class LazyEigenThreadPoolHolder {
 public:
  explicit LazyEigenThreadPoolHolder(int num_threads) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSeigen_supportDTcc mht_7(mht_7_v, 286, "", "./tensorflow/lite/kernels/eigen_support.cc", "LazyEigenThreadPoolHolder");

    SetNumThreads(num_threads);
  }

  // Gets the ThreadPoolDevice, creating if necessary.
  const Eigen::ThreadPoolDevice* GetThreadPoolDevice() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSeigen_supportDTcc mht_8(mht_8_v, 294, "", "./tensorflow/lite/kernels/eigen_support.cc", "GetThreadPoolDevice");

    if (!device_) {
      thread_pool_wrapper_.reset(
          new EigenThreadPoolWrapper(target_num_threads_));
      device_.reset(new Eigen::ThreadPoolDevice(thread_pool_wrapper_.get(),
                                                target_num_threads_));
    }
    return device_.get();
  }

  // Updates the thread count, invalidating the ThreadPoolDevice if necessary.
  void SetNumThreads(int num_threads) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSeigen_supportDTcc mht_9(mht_9_v, 308, "", "./tensorflow/lite/kernels/eigen_support.cc", "SetNumThreads");

    const int target_num_threads = GetNumThreads(num_threads);
    if (target_num_threads_ != target_num_threads) {
      target_num_threads_ = target_num_threads;
      // As the device references the thread pool wrapper, destroy it first.
      device_.reset();
      thread_pool_wrapper_.reset();
    }
  }

 private:
  int target_num_threads_ = kDefaultNumThreadpoolThreads;
  // Both device_ and thread_pool_wrapper_ are lazily created.
  std::unique_ptr<Eigen::ThreadPoolDevice> device_;
  std::unique_ptr<Eigen::ThreadPoolInterface> thread_pool_wrapper_;
};

struct RefCountedEigenContext : public TfLiteExternalContext {
  std::unique_ptr<LazyEigenThreadPoolHolder> thread_pool_holder;
  int num_references = 0;
};

RefCountedEigenContext* GetEigenContext(TfLiteContext* context) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSeigen_supportDTcc mht_10(mht_10_v, 333, "", "./tensorflow/lite/kernels/eigen_support.cc", "GetEigenContext");

  return reinterpret_cast<RefCountedEigenContext*>(
      context->GetExternalContext(context, kTfLiteEigenContext));
}

TfLiteStatus Refresh(TfLiteContext* context) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSeigen_supportDTcc mht_11(mht_11_v, 341, "", "./tensorflow/lite/kernels/eigen_support.cc", "Refresh");

  if (IsValidNumThreads(context->recommended_num_threads)) {
    SetEigenNbThreads(GetNumThreads(context->recommended_num_threads));
  }

  auto* ptr = GetEigenContext(context);
  if (ptr != nullptr) {
    ptr->thread_pool_holder->SetNumThreads(context->recommended_num_threads);
  }

  return kTfLiteOk;
}

}  // namespace

void IncrementUsageCounter(TfLiteContext* context) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSeigen_supportDTcc mht_12(mht_12_v, 359, "", "./tensorflow/lite/kernels/eigen_support.cc", "IncrementUsageCounter");

  auto* ptr = GetEigenContext(context);
  if (ptr == nullptr) {
    if (IsValidNumThreads(context->recommended_num_threads)) {
      SetEigenNbThreads(context->recommended_num_threads);
    }
    ptr = new RefCountedEigenContext;
    ptr->type = kTfLiteEigenContext;
    ptr->Refresh = Refresh;
    ptr->thread_pool_holder.reset(
        new LazyEigenThreadPoolHolder(context->recommended_num_threads));
    ptr->num_references = 0;
    context->SetExternalContext(context, kTfLiteEigenContext, ptr);
  }
  ptr->num_references++;
}

void DecrementUsageCounter(TfLiteContext* context) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSeigen_supportDTcc mht_13(mht_13_v, 379, "", "./tensorflow/lite/kernels/eigen_support.cc", "DecrementUsageCounter");

  auto* ptr = GetEigenContext(context);
  if (ptr == nullptr) {
    TF_LITE_FATAL(
        "Call to DecrementUsageCounter() not preceded by "
        "IncrementUsageCounter()");
  }
  if (--ptr->num_references == 0) {
    delete ptr;
    context->SetExternalContext(context, kTfLiteEigenContext, nullptr);
  }
}

const Eigen::ThreadPoolDevice* GetThreadPoolDevice(TfLiteContext* context) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSlitePSkernelsPSeigen_supportDTcc mht_14(mht_14_v, 395, "", "./tensorflow/lite/kernels/eigen_support.cc", "GetThreadPoolDevice");

  auto* ptr = GetEigenContext(context);
  if (ptr == nullptr) {
    TF_LITE_FATAL(
        "Call to GetFromContext() not preceded by IncrementUsageCounter()");
  }
  return ptr->thread_pool_holder->GetThreadPoolDevice();
}

}  // namespace eigen_support
}  // namespace tflite
