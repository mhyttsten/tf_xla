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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTcc() {
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

#define EIGEN_USE_THREADS

#include "tensorflow/compiler/xla/service/backend.h"

#include <algorithm>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/byte_order.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {

BackendOptions& BackendOptions::set_platform(se::Platform* platform) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTcc mht_0(mht_0_v, 211, "", "./tensorflow/compiler/xla/service/backend.cc", "BackendOptions::set_platform");

  platform_ = platform;
  return *this;
}

se::Platform* BackendOptions::platform() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTcc mht_1(mht_1_v, 219, "", "./tensorflow/compiler/xla/service/backend.cc", "BackendOptions::platform");
 return platform_; }

BackendOptions& BackendOptions::set_intra_op_parallelism_threads(
    int num_threads) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTcc mht_2(mht_2_v, 225, "", "./tensorflow/compiler/xla/service/backend.cc", "BackendOptions::set_intra_op_parallelism_threads");

  intra_op_parallelism_threads_ = num_threads;
  return *this;
}

int BackendOptions::intra_op_parallelism_threads() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTcc mht_3(mht_3_v, 233, "", "./tensorflow/compiler/xla/service/backend.cc", "BackendOptions::intra_op_parallelism_threads");

  return intra_op_parallelism_threads_;
}

BackendOptions& BackendOptions::set_allowed_devices(
    const absl::optional<std::set<int>>& allowed_devices) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTcc mht_4(mht_4_v, 241, "", "./tensorflow/compiler/xla/service/backend.cc", "BackendOptions::set_allowed_devices");

  allowed_devices_ = allowed_devices;
  return *this;
}

const absl::optional<std::set<int>>& BackendOptions::allowed_devices() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTcc mht_5(mht_5_v, 249, "", "./tensorflow/compiler/xla/service/backend.cc", "BackendOptions::allowed_devices");

  return allowed_devices_;
}

// Define this in .cc file to avoid having to include eigen or forward declare
// these types in the header.
struct Backend::IntraOpThreadPool {
  explicit IntraOpThreadPool(const int num_threads)
      : pool(new tensorflow::thread::ThreadPool(tensorflow::Env::Default(),
                                                "XLAEigen", num_threads)),
        device(new Eigen::ThreadPoolDevice(pool->AsEigenThreadPool(),
                                           pool->NumThreads())) {}

  std::unique_ptr<tensorflow::thread::ThreadPool> pool;
  std::unique_ptr<Eigen::ThreadPoolDevice> device;
};

/* static */ StatusOr<std::unique_ptr<Backend>> Backend::CreateBackend(
    const BackendOptions& options) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTcc mht_6(mht_6_v, 270, "", "./tensorflow/compiler/xla/service/backend.cc", "Backend::CreateBackend");

  se::Platform* platform = options.platform();
  TF_ASSIGN_OR_RETURN(auto compiler, Compiler::GetForPlatform(platform));
  TF_ASSIGN_OR_RETURN(
      auto stream_executors,
      PlatformUtil::GetStreamExecutors(platform, options.allowed_devices()));
  TF_ASSIGN_OR_RETURN(auto transfer_manager,
                      TransferManager::GetForPlatform(platform));
  TF_ASSIGN_OR_RETURN(auto computation_placer,
                      ComputationPlacer::GetForPlatform(platform));
  std::unique_ptr<Backend> backend(
      new Backend(platform, compiler, stream_executors, transfer_manager,
                  computation_placer, options.intra_op_parallelism_threads()));
  return std::move(backend);
}

/* static */ StatusOr<std::unique_ptr<Backend>>
Backend::CreateDefaultBackend() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTcc mht_7(mht_7_v, 290, "", "./tensorflow/compiler/xla/service/backend.cc", "Backend::CreateDefaultBackend");

  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      PlatformUtil::GetDefaultPlatform());
  BackendOptions backend_options;
  backend_options.set_platform(platform);
  return CreateBackend(backend_options);
}

StatusOr<StreamPool::Ptr> Backend::BorrowStream(int device_ordinal) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTcc mht_8(mht_8_v, 301, "", "./tensorflow/compiler/xla/service/backend.cc", "Backend::BorrowStream");

  TF_ASSIGN_OR_RETURN(auto executor, stream_executor(device_ordinal));
  return BorrowStream(executor);
}

StatusOr<StreamPool::Ptr> Backend::BorrowStream(se::StreamExecutor* executor) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTcc mht_9(mht_9_v, 309, "", "./tensorflow/compiler/xla/service/backend.cc", "Backend::BorrowStream");

  absl::MutexLock l(&mu_);
  if (!stream_pools_.contains(executor)) {
    stream_pools_.emplace(executor, absl::make_unique<StreamPool>());
  }
  return stream_pools_.at(executor)->BorrowStream(executor);
}

Backend::Backend(se::Platform* platform, Compiler* compiler,
                 absl::Span<se::StreamExecutor* const> stream_executors,
                 TransferManager* transfer_manager,
                 ComputationPlacer* computation_placer,
                 int intra_op_parallelism_threads)
    : platform_(platform),
      compiler_(compiler),
      transfer_manager_(transfer_manager),
      computation_placer_(computation_placer),
      stream_executors_(stream_executors.begin(), stream_executors.end()) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTcc mht_10(mht_10_v, 329, "", "./tensorflow/compiler/xla/service/backend.cc", "Backend::Backend");

  // Create a memory allocator for the valid stream executors.
  memory_allocator_ = std::make_shared<se::StreamExecutorMemoryAllocator>(
      platform, stream_executors_);
  CHECK(!stream_executors_.empty())
      << "Service found no devices for backend " << platform_->Name() << '.';

  if (platform->id() == se::host::kHostPlatformId) {
    const int num_threads = intra_op_parallelism_threads > 0
                                ? intra_op_parallelism_threads
                                : tensorflow::port::MaxParallelism();
    intra_op_thread_pool_.reset(new IntraOpThreadPool(num_threads));
  }
}

Backend::~Backend() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTcc mht_11(mht_11_v, 347, "", "./tensorflow/compiler/xla/service/backend.cc", "Backend::~Backend");
}

int Backend::default_device_ordinal() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTcc mht_12(mht_12_v, 352, "", "./tensorflow/compiler/xla/service/backend.cc", "Backend::default_device_ordinal");

  return default_stream_executor()->device_ordinal();
}

const Eigen::ThreadPoolDevice* Backend::eigen_intra_op_thread_pool_device()
    const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTcc mht_13(mht_13_v, 360, "", "./tensorflow/compiler/xla/service/backend.cc", "Backend::eigen_intra_op_thread_pool_device");

  if (intra_op_thread_pool_ == nullptr) {
    return nullptr;
  }
  return intra_op_thread_pool_->device.get();
}

tensorflow::thread::ThreadPool* Backend::eigen_intra_op_thread_pool() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTcc mht_14(mht_14_v, 370, "", "./tensorflow/compiler/xla/service/backend.cc", "Backend::eigen_intra_op_thread_pool");

  if (intra_op_thread_pool_ == nullptr) {
    return nullptr;
  }
  return intra_op_thread_pool_->pool.get();
}

StatusOr<se::StreamExecutor*> Backend::stream_executor(
    int device_ordinal) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTcc mht_15(mht_15_v, 381, "", "./tensorflow/compiler/xla/service/backend.cc", "Backend::stream_executor");

  if (device_ordinal < 0 ||
      device_ordinal > stream_executors_.back()->device_ordinal()) {
    return InvalidArgument(
        "Invalid device ordinal value (%d). Valid range is [0, %d].",
        device_ordinal, stream_executors_.back()->device_ordinal());
  }
  for (auto* executor : stream_executors_) {
    if (executor->device_ordinal() == device_ordinal) {
      return executor;
    }
  }
  return InvalidArgument("device %s not supported by XLA service",
                         device_name(device_ordinal));
}

StatusOr<bool> Backend::devices_equivalent(int device_ordinal_a,
                                           int device_ordinal_b) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTcc mht_16(mht_16_v, 401, "", "./tensorflow/compiler/xla/service/backend.cc", "Backend::devices_equivalent");

  // Use the name from device description to determine equivalence. This is a
  // bit crude but works for GPUs which is the important case where we compile
  // an executable for one GPU and want to know if it will run (well) on
  // another.
  TF_ASSIGN_OR_RETURN(se::StreamExecutor * executor_a,
                      stream_executor(device_ordinal_a));
  TF_ASSIGN_OR_RETURN(se::StreamExecutor * executor_b,
                      stream_executor(device_ordinal_b));
  return (executor_a->GetDeviceDescription().name() ==
          executor_b->GetDeviceDescription().name());
}

Status Backend::ResetDevices() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTcc mht_17(mht_17_v, 417, "", "./tensorflow/compiler/xla/service/backend.cc", "Backend::ResetDevices");

  return transfer_manager_->ResetDevices(stream_executors_);
}

}  // namespace xla
