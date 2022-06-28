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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_BACKEND_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_BACKEND_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTh() {
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


#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/stream_pool.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"

namespace Eigen {
struct ThreadPoolDevice;
}

namespace xla {

// Options to configure the backend when it is created.
class BackendOptions {
 public:
  // Set the platform backing the backend, or nullptr for the default platform.
  BackendOptions& set_platform(se::Platform* platform);
  se::Platform* platform() const;

  // Sets the thread pool size for parallel execution of an individual operator.
  // The default value of -1 will result in initializing the thread pool with
  // the number of threads equal to the number of cores in the system.
  BackendOptions& set_intra_op_parallelism_threads(int num_threads);
  int intra_op_parallelism_threads() const;

  // Sets the allowed_devices for selectively constructing stream executors
  // on the platform.
  BackendOptions& set_allowed_devices(
      const absl::optional<std::set<int>>& allowed_devices);
  const absl::optional<std::set<int>>& allowed_devices() const;

 private:
  se::Platform* platform_ = nullptr;
  int intra_op_parallelism_threads_ = -1;
  absl::optional<std::set<int>> allowed_devices_;
};

// Class which encapsulates an XLA backend. It includes everything necessary
// to compile and execute computations on a particular platform.
//
// It also offers a pooling API for creation/use of initialized streams:
//
//    StreamPool::Ptr stream = backend->BorrowStream().ConsumeValueOrDie();
class Backend {
 public:
  // Creates a new backend.
  static StatusOr<std::unique_ptr<Backend>> CreateBackend(
      const BackendOptions& options);

  // Creates a backend for the default platform. The default platform is defined
  // in PlatformUtil.
  static StatusOr<std::unique_ptr<Backend>> CreateDefaultBackend();

  ~Backend();

  // Accessors for the various objects.
  se::Platform* platform() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTh mht_0(mht_0_v, 256, "", "./tensorflow/compiler/xla/service/backend.h", "platform");
 return platform_; }
  Compiler* compiler() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTh mht_1(mht_1_v, 260, "", "./tensorflow/compiler/xla/service/backend.h", "compiler");
 return compiler_; }
  se::DeviceMemoryAllocator* memory_allocator() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTh mht_2(mht_2_v, 264, "", "./tensorflow/compiler/xla/service/backend.h", "memory_allocator");

    return memory_allocator_.get();
  }
  std::shared_ptr<se::DeviceMemoryAllocator> shared_memory_allocator() const {
    return memory_allocator_;
  }
  TransferManager* transfer_manager() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTh mht_3(mht_3_v, 273, "", "./tensorflow/compiler/xla/service/backend.h", "transfer_manager");
 return transfer_manager_; }
  ComputationPlacer* computation_placer() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTh mht_4(mht_4_v, 277, "", "./tensorflow/compiler/xla/service/backend.h", "computation_placer");
 return computation_placer_; }

  // Returns the number of devices of the platform type which are visible. Not
  // all of these devices may be usable by XLA.
  int device_count() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTh mht_5(mht_5_v, 284, "", "./tensorflow/compiler/xla/service/backend.h", "device_count");
 return stream_executors_.size(); }

  // Returns the device ordinal number of the default device.
  int default_device_ordinal() const;

  // Returns stream executors of all supported devices for this backend. The
  // executors are ordered by the device ordinal.
  const std::vector<se::StreamExecutor*>& stream_executors() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTh mht_6(mht_6_v, 294, "", "./tensorflow/compiler/xla/service/backend.h", "stream_executors");

    return stream_executors_;
  }

  // Returns the stream executor for the given device ordinal.
  StatusOr<se::StreamExecutor*> stream_executor(int device_ordinal) const;

  // Returns the stream executor for the default device ordinal. This stream
  // executor can only be used when the number of computations is 1 (replication
  // can be > 1).
  se::StreamExecutor* default_stream_executor() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTh mht_7(mht_7_v, 307, "", "./tensorflow/compiler/xla/service/backend.h", "default_stream_executor");

    CHECK(!stream_executors_.empty());
    return stream_executors_[0];
  }

  // Borrows a stream for use by the caller, either by grabbing it from an
  // internal pool, or by constructing/initializating it, and returns the result
  // to the caller.
  StatusOr<StreamPool::Ptr> BorrowStream(int device_ordinal);
  StatusOr<StreamPool::Ptr> BorrowStream(se::StreamExecutor* executor);

  // Returns a function to borrow a stream, as `BorrowStream` above does.
  // Purely for convenience, the caller could rather make this anonymous
  // function itself.
  std::function<StatusOr<StreamPool::Ptr>(int)> StreamBorrower() {
    return [this](int device_ordinal) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTh mht_8(mht_8_v, 325, "", "./tensorflow/compiler/xla/service/backend.h", "lambda");
 return BorrowStream(device_ordinal); };
  }

  // Returns whether the given device ordinal of the backend is supported.
  bool device_ordinal_supported(int device_ordinal) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTh mht_9(mht_9_v, 332, "", "./tensorflow/compiler/xla/service/backend.h", "device_ordinal_supported");

    return (device_ordinal >= 0 && device_ordinal < device_count() &&
            stream_executors_[device_ordinal] != nullptr);
  }

  // Return a string identifier for the given device, eg: "GPU:3".
  std::string device_name(int device_ordinal) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSbackendDTh mht_10(mht_10_v, 341, "", "./tensorflow/compiler/xla/service/backend.h", "device_name");

    return absl::StrCat(platform_->Name(), ":", device_ordinal);
  }

  // Returns true if the devices with the given ordinals are equivalent from
  // XLA's perspective. That is, an executable compiled for one device would
  // be equivalent to an executable compiled for the other.
  StatusOr<bool> devices_equivalent(int device_ordinal_a, int device_ordinal_b);

  // For the host platform, returns the configured eigen threadpool device to be
  // used for scheduling work. For other platforms, returns NULL.
  const Eigen::ThreadPoolDevice* eigen_intra_op_thread_pool_device() const;
  tensorflow::thread::ThreadPool* eigen_intra_op_thread_pool() const;

  // Resets the devices associated with this backend.
  Status ResetDevices();

 private:
  Backend(se::Platform* platform, Compiler* compiler,
          absl::Span<se::StreamExecutor* const> stream_executors,
          TransferManager* transfer_manager,
          ComputationPlacer* computation_placer,
          int intra_op_parallelism_threads);
  Backend(const Backend&) = delete;
  Backend& operator=(const Backend&) = delete;

  se::Platform* platform_;
  Compiler* compiler_;
  TransferManager* transfer_manager_;
  ComputationPlacer* computation_placer_;

  // Vector of stream executors. stream_executors_[0] is the default executor.
  std::vector<se::StreamExecutor*> stream_executors_;

  absl::Mutex mu_;

  // Mapping from stream executor to stream pools, used by `BorrowStream` above.
  absl::flat_hash_map<se::StreamExecutor*, std::unique_ptr<StreamPool>>
      stream_pools_ ABSL_GUARDED_BY(mu_);

  // The default memory allocator to use.
  // This must be a shared_ptr, as this is passed all the way down to the
  // cluster compilation. This allows asynchronous compilation to hold a
  // referecence until the compilation is finished.
  std::shared_ptr<se::StreamExecutorMemoryAllocator> memory_allocator_;

  // For the CPU backend, an Eigen threadpool device for use by Eigen code.
  struct IntraOpThreadPool;
  std::unique_ptr<IntraOpThreadPool> intra_op_thread_pool_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_BACKEND_H_
