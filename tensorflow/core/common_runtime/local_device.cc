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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSlocal_deviceDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlocal_deviceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSlocal_deviceDTcc() {
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

#define EIGEN_USE_THREADS

#include "tensorflow/core/common_runtime/local_device.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/common_runtime/process_state.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/cpu_feature_guard.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/numa.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace {

bool OverrideGlobalThreadPoolFromEnvironment() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlocal_deviceDTcc mht_0(mht_0_v, 205, "", "./tensorflow/core/common_runtime/local_device.cc", "OverrideGlobalThreadPoolFromEnvironment");

  static const bool override_global_threadpool = [] {
    bool flag;
    auto status = ReadBoolFromEnvVar("TF_OVERRIDE_GLOBAL_THREADPOOL",
                                     /*default_val=*/false, &flag);
    if (!status.ok()) {
      LOG(ERROR) << "OverrideGlobalThreadPool: " << status.error_message();
      return false;
    }
    return flag;
  }();
  return override_global_threadpool;
}

}  // namespace

/* static */
bool LocalDevice::use_global_threadpool_ = true;
mutex LocalDevice::global_tp_mu_;
gtl::InlinedVector<LocalDevice::EigenThreadPoolInfo*, 4>
    LocalDevice::global_tp_info_;

struct LocalDevice::EigenThreadPoolInfo {
  // Wrapper so we can provide the CPUAllocator to Eigen for use
  // when ops need extra tmp memory.
  class EigenAllocator : public Eigen::Allocator {
   public:
    explicit EigenAllocator(tensorflow::Allocator* a) : allocator_(a) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlocal_deviceDTcc mht_1(mht_1_v, 235, "", "./tensorflow/core/common_runtime/local_device.cc", "EigenAllocator");
}
    void* allocate(size_t num_bytes) const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlocal_deviceDTcc mht_2(mht_2_v, 239, "", "./tensorflow/core/common_runtime/local_device.cc", "allocate");

      return allocator_->AllocateRaw(64, num_bytes);
    }
    void deallocate(void* buffer) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlocal_deviceDTcc mht_3(mht_3_v, 245, "", "./tensorflow/core/common_runtime/local_device.cc", "deallocate");

      allocator_->DeallocateRaw(buffer);
    }
    tensorflow::Allocator* allocator_;
  };

  explicit EigenThreadPoolInfo(const SessionOptions& options, int numa_node,
                               Allocator* allocator) {
    // Use session setting if specified.
    int32_t intra_op_parallelism_threads =
        options.config.intra_op_parallelism_threads();
    // If no session setting, use environment setting.
    if (intra_op_parallelism_threads == 0) {
      static int env_num_threads = NumIntraOpThreadsFromEnvironment();
      intra_op_parallelism_threads = env_num_threads;
      // If no session setting or environment, compute a reasonable default.
      if (intra_op_parallelism_threads == 0) {
        intra_op_parallelism_threads = port::MaxParallelism(numa_node);
      }
    }
    ThreadOptions thread_opts;
    thread_opts.numa_node = numa_node;
    eigen_worker_threads_.num_threads = intra_op_parallelism_threads;
    eigen_worker_threads_.workers = new thread::ThreadPool(
        options.env, thread_opts, strings::StrCat("numa_", numa_node, "_Eigen"),
        intra_op_parallelism_threads,
        !options.config.experimental().disable_thread_spinning(),
        /*allocator=*/nullptr);
    Eigen::ThreadPoolInterface* threadpool =
        eigen_worker_threads_.workers->AsEigenThreadPool();
    if (allocator != nullptr) {
      eigen_allocator_.reset(new EigenAllocator(allocator));
    }
    eigen_device_.reset(new Eigen::ThreadPoolDevice(
        threadpool, eigen_worker_threads_.num_threads, eigen_allocator_.get()));
  }

  ~EigenThreadPoolInfo() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlocal_deviceDTcc mht_4(mht_4_v, 285, "", "./tensorflow/core/common_runtime/local_device.cc", "~EigenThreadPoolInfo");

    eigen_device_.reset();
    delete eigen_worker_threads_.workers;
  }

  DeviceBase::CpuWorkerThreads eigen_worker_threads_;
  std::unique_ptr<Eigen::ThreadPoolDevice> eigen_device_;
  std::unique_ptr<EigenAllocator> eigen_allocator_;
};

LocalDevice::LocalDevice(const SessionOptions& options,
                         const DeviceAttributes& attributes)
    : Device(options.env, attributes), owned_tp_info_(nullptr) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlocal_deviceDTcc mht_5(mht_5_v, 300, "", "./tensorflow/core/common_runtime/local_device.cc", "LocalDevice::LocalDevice");

  // Log info messages if TensorFlow is not compiled with instructions that
  // could speed up performance and are available on the current CPU.
  port::InfoAboutUnusedCPUFeatures();
  LocalDevice::EigenThreadPoolInfo* tp_info;

  if (OverrideGlobalThreadPoolFromEnvironment()) {
    set_use_global_threadpool(false);
  }

  if (use_global_threadpool_) {
    mutex_lock l(global_tp_mu_);
    if (options.config.experimental().use_numa_affinity()) {
      int numa_node = attributes.locality().numa_node();
      int num_numa_nodes = port::NUMANumNodes();
      DCHECK_LT(numa_node, num_numa_nodes);
      Allocator* numa_allocator =
          ProcessState::singleton()->GetCPUAllocator(numa_node);
      while (numa_node >= global_tp_info_.size()) {
        global_tp_info_.push_back(nullptr);
      }
      if (!global_tp_info_[numa_node]) {
        global_tp_info_[numa_node] = new LocalDevice::EigenThreadPoolInfo(
            options, numa_node, numa_allocator);
      }
      tp_info = global_tp_info_[numa_node];
    } else {
      if (global_tp_info_.empty()) {
        global_tp_info_.push_back(new LocalDevice::EigenThreadPoolInfo(
            options, port::kNUMANoAffinity, nullptr));
      }
      tp_info = global_tp_info_[0];
    }
  } else {
    // Each LocalDevice owns a separate ThreadPoolDevice for numerical
    // computations.
    // TODO(tucker): NUMA for these too?
    owned_tp_info_.reset(new LocalDevice::EigenThreadPoolInfo(
        options, port::kNUMANoAffinity, nullptr));
    tp_info = owned_tp_info_.get();
  }
  set_tensorflow_cpu_worker_threads(&tp_info->eigen_worker_threads_);
  set_eigen_cpu_device(tp_info->eigen_device_.get());
}

LocalDevice::~LocalDevice() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSlocal_deviceDTcc mht_6(mht_6_v, 348, "", "./tensorflow/core/common_runtime/local_device.cc", "LocalDevice::~LocalDevice");
}

}  // namespace tensorflow
