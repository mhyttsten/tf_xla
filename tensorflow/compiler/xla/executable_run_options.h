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

#ifndef TENSORFLOW_COMPILER_XLA_EXECUTABLE_RUN_OPTIONS_H_
#define TENSORFLOW_COMPILER_XLA_EXECUTABLE_RUN_OPTIONS_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTh() {
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


#include <string>

#include "tensorflow/compiler/xla/types.h"

// These classes are forward declared so that ExecutableRunOptions can be linked
// into an XLA-compiled binary without having to link all of the pointed-to
// objects (e.g., for an ahead-of-time compiled CPU binary, the gpu tools don't
// need to be linked).
namespace stream_executor {
class Stream;
class Platform;
class DeviceMemoryAllocator;
}  // namespace stream_executor

namespace Eigen {
struct ThreadPoolDevice;
}  // namespace Eigen

namespace xla {

class DeviceAssignment;
class ExecutionProfile;
namespace gpu {
class GpuExecutableRunOptions;
}  // namespace gpu

// A unique identifier for a particular "logical execution" of an XLA model.
//
// A logical execution might encompass multiple executions of one or more
// HloModules.  Runs that are part of the same logical execution can
// communicate via collective ops (e.g. kAllToAll), whereas runs that are part
// of different logical executions are isolated.
class RunId {
 public:
  // Creates a new, unique RunId.
  RunId();
  explicit RunId(int64_t value) : data_(value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTh mht_0(mht_0_v, 224, "", "./tensorflow/compiler/xla/executable_run_options.h", "RunId");
}

  RunId(const RunId&) = default;
  RunId& operator=(const RunId&) = default;
  friend bool operator==(const RunId& a, const RunId& b);
  std::string ToString() const;
  int64_t ToInt() const;

  template <typename H>
  friend H AbslHashValue(H h, const RunId& id) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTh mht_1(mht_1_v, 236, "", "./tensorflow/compiler/xla/executable_run_options.h", "AbslHashValue");

    return H::combine(std::move(h), id.data_);
  }

 private:
  int64_t data_;
};

// Callback used by the GPU backend only. This is an "one-sided" version of
// ThenDoHostCallback that enqueues a callback onto a stream. The difference
// with ThenDoHostCallback is that the device does not block waiting for the
// callback to complete; instead the callback is scheduled by the runtime.
// This functionality must be provided by the caller, and hence is provided in
// callback form.
using ThenExecuteFunction =
    std::function<void(stream_executor::Stream*, std::function<void()>)>;

// Class containing options for running a LocalExecutable.
class ExecutableRunOptions {
 public:
  // Specifies the allocator to use during execution.
  ExecutableRunOptions& set_allocator(
      stream_executor::DeviceMemoryAllocator* allocator);
  stream_executor::DeviceMemoryAllocator* allocator() const;

  // If set, this is the device to run the computation on. Valid device_ordinal
  // values are: 0 to # of devices - 1. These values are identical to the device
  // ordinal values used by StreamExecutor. The device must be of the same type
  // as the executable was compiled for. A value of -1 indicates this option has
  // not been set.
  ExecutableRunOptions& set_device_ordinal(int device_ordinal);
  int device_ordinal() const;

  // If set, this is the stream to run the computation on. The platform of the
  // stream must match the platform the executable was built for.  A value of
  // nullptr indicates the option has not been set.
  ExecutableRunOptions& set_stream(stream_executor::Stream* stream);
  stream_executor::Stream* stream() const;

  // If set, this is the stream to perform any pre-computation transfers on.
  // The platform of the stream must match the platform the executable was
  // built for.  A value of nullptr indicates the option has not been set.
  ExecutableRunOptions& set_host_to_device_stream(
      stream_executor::Stream* stream);
  stream_executor::Stream* host_to_device_stream() const;

  // Sets the thread pool device on which to run Eigen subcomputations.
  //
  // This field must be set for XLA:CPU models that call Eigen routines, but may
  // be null otherwise.  Routines that use this field should always CHECK (or
  // TF_RET_CHECK) that it's not null before dereferencing it, so that users get
  // a clean crash rather than a segfault.
  //
  // Does not take ownership.
  ExecutableRunOptions& set_intra_op_thread_pool(
      const Eigen::ThreadPoolDevice* intra_op_thread_pool);
  const Eigen::ThreadPoolDevice* intra_op_thread_pool() const;

  // If set, profiling information is written to 'profile'.
  ExecutionProfile* execution_profile() const;
  ExecutableRunOptions& set_execution_profile(ExecutionProfile* profile);

  ExecutableRunOptions& set_device_assignment(
      const DeviceAssignment* device_assignment);
  const DeviceAssignment* device_assignment() const;

  ExecutableRunOptions& set_rng_seed(int rng_seed);
  int rng_seed() const;

  ExecutableRunOptions& set_launch_id(int32_t launch_id) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTh mht_2(mht_2_v, 308, "", "./tensorflow/compiler/xla/executable_run_options.h", "set_launch_id");

    launch_id_ = launch_id;
    return *this;
  }

  int32_t launch_id() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTh mht_3(mht_3_v, 316, "", "./tensorflow/compiler/xla/executable_run_options.h", "launch_id");
 return launch_id_; }

  ExecutableRunOptions& set_run_id(RunId id);
  RunId run_id() const;

  // See documentation on ThenExecuteFunction.
  ExecutableRunOptions& set_then_execute_function(ThenExecuteFunction* f) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTh mht_4(mht_4_v, 325, "", "./tensorflow/compiler/xla/executable_run_options.h", "set_then_execute_function");

    then_execute_function_ = f;
    return *this;
  }
  ThenExecuteFunction* then_execute_function() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSexecutable_run_optionsDTh mht_5(mht_5_v, 332, "", "./tensorflow/compiler/xla/executable_run_options.h", "then_execute_function");

    return then_execute_function_;
  }

  // GPU-backend specific options. These are kept out-of-line to avoid bloating
  // the size of this dependency for CPU-only AOT builds.
  ExecutableRunOptions& set_gpu_executable_run_options(
      const gpu::GpuExecutableRunOptions* gpu_executable_run_options);
  const gpu::GpuExecutableRunOptions* gpu_executable_run_options() const;

 private:
  stream_executor::DeviceMemoryAllocator* allocator_ = nullptr;
  int device_ordinal_ = -1;
  const DeviceAssignment* device_assignment_ = nullptr;
  stream_executor::Stream* stream_ = nullptr;
  const Eigen::ThreadPoolDevice* intra_op_thread_pool_ = nullptr;
  ExecutionProfile* execution_profile_ = nullptr;
  int rng_seed_ = 0;
  int32_t launch_id_ = 0;
  stream_executor::Stream* host_to_device_stream_ = nullptr;
  ThenExecuteFunction* then_execute_function_ = nullptr;
  RunId run_id_;
  const gpu::GpuExecutableRunOptions* gpu_executable_run_options_ = nullptr;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_EXECUTABLE_RUN_OPTIONS_H_
