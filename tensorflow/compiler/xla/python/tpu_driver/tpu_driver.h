// Copyright 2019 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_TPU_DRIVER_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_TPU_DRIVER_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPStpu_driverDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPStpu_driverDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPStpu_driverDTh() {
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


#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/python/tpu_driver/platform/external/compat.h"
#include "tensorflow/compiler/xla/python/tpu_driver/tpu_driver.pb.h"
#include "tensorflow/compiler/xla/service/hlo.pb.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/logging.h"

// This API is EXPERIMENTAL and under active development. It is subject to
// change without notice.

namespace tpu_driver {

int64_t ComputeBytesFromShape(const xla::ShapeProto& shape);

// Represents the deferred completion of a scheduled operation.
//
// Events may be blocked on, or used as `wait_for` arguments to enforce
// inter-operation dependencies.
class Event {
 public:
  virtual ~Event() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPStpu_driverDTh mht_0(mht_0_v, 221, "", "./tensorflow/compiler/xla/python/tpu_driver/tpu_driver.h", "~Event");
}

  // Blocks until the event completes and returns the result status.
  virtual xla::Status Await() = 0;
  // Returns an empty result if the wait times out.
  virtual absl::optional<xla::Status> AwaitWithTimeout(
      absl::Duration duration) = 0;

  // If the event is already done, the callback is called immediately.
  virtual void AddCallback(std::function<void(xla::Status)> callback) = 0;
};

// Represents a device memory allocation.
class BufferHandle {
 public:
  virtual ~BufferHandle() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPStpu_driverDTh mht_1(mht_1_v, 239, "", "./tensorflow/compiler/xla/python/tpu_driver/tpu_driver.h", "~BufferHandle");
}

  // This event completes after the device memory is actually allocated.
  //
  // Methods that take a buffer handle, such as ExecuteProgram and Transfer*,
  // automatically add this event as a dependency.
  virtual std::shared_ptr<Event> OnReady() = 0;

  virtual int64_t size_in_bytes() = 0;
  virtual absl::optional<xla::ShapeProto> shape() = 0;
};

// Represents a compiled program on the host.
class CompiledProgramHandle {
 public:
  virtual ~CompiledProgramHandle() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPStpu_driverDTh mht_2(mht_2_v, 257, "", "./tensorflow/compiler/xla/python/tpu_driver/tpu_driver.h", "~CompiledProgramHandle");
}

  // This Event completes after the program is actually compiled on the host.
  //
  // Methods that take a compiled program handle, including LoadProgram,
  // automatically add this event as a dependency.
  virtual std::shared_ptr<Event> OnReady() = 0;

  virtual int64_t size_in_bytes() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPStpu_driverDTh mht_3(mht_3_v, 268, "", "./tensorflow/compiler/xla/python/tpu_driver/tpu_driver.h", "size_in_bytes");

    LOG(FATAL) << "Unimplemented.";
    return 0;
  }

  // Returns the shape of the compiled program. Blocks until compile completes.
  virtual xla::Status program_shape(xla::ProgramShapeProto* program_shape) = 0;
};

// Represents a program loaded on the device.
class LoadedProgramHandle {
 public:
  virtual ~LoadedProgramHandle() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPStpu_driverDTh mht_4(mht_4_v, 283, "", "./tensorflow/compiler/xla/python/tpu_driver/tpu_driver.h", "~LoadedProgramHandle");
}

  // This Event completes after the program is actually loaded on the device.
  //
  // Methods that take a loaded program handle, including ExecuteProgram and
  // UnloadProgram, automatically add this event as a dependency.
  virtual std::shared_ptr<Event> OnReady() = 0;

  virtual int64_t size_in_bytes() {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPStpu_driverDTh mht_5(mht_5_v, 294, "", "./tensorflow/compiler/xla/python/tpu_driver/tpu_driver.h", "size_in_bytes");

    LOG(FATAL) << "Unimplemented.";
    return 0;
  }
};

// A TpuLinearizer manages the linearization and delinearization of user buffers
// in the TPU driver. This interface is not yet implemented.
class TpuLinearizer {
 public:
  virtual ~TpuLinearizer() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPStpu_driverDTh mht_6(mht_6_v, 307, "", "./tensorflow/compiler/xla/python/tpu_driver/tpu_driver.h", "~TpuLinearizer");
}

  int64_t ComputeBytesFromShape(const xla::ShapeProto& shape) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPStpu_driverDTh mht_7(mht_7_v, 312, "", "./tensorflow/compiler/xla/python/tpu_driver/tpu_driver.h", "ComputeBytesFromShape");

    return ::tpu_driver::ComputeBytesFromShape(shape);
  }
  virtual int64_t ComputeLinearizedBytesFromShape(
      const xla::ShapeProto& shape) = 0;

  virtual xla::Status LinearizeShape(void* dst, const void* src,
                                     const xla::ShapeProto& shape) = 0;
  virtual xla::Status DelinearizeShape(void* dst, const void* src,
                                       const xla::ShapeProto& shape) = 0;
};

// A TpuDriver manages a set of operations scheduled to run on a TPU system.
//
// By default, two independently scheduled operations may execute in any order.
// Ordering can be imposed in one of two ways:
//
// 1. Users can specify event dependencies via the `wait_for` argument.
// 2. Operations using buffer or program handles implicitly wait for the handles
//    to become ready before executing.
//
// For returned handle objects, the user is responsible for calling the release
// methods (Deallocate, UnloadProgram, etc.) that consume the given unique_ptr
// arguments and free up device resources. For returned event objects, there is
// no release method; the user can let them go out of scope naturally. As soon
// as those methods accepting plain-pointer arguments return, the user can let
// the corresponding smart-pointer objects be released or go out of scope,
// regardless of whether the scheduled device operations have started execution.
class TpuDriver {
 public:
  virtual ~TpuDriver() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPStpu_driverDTh mht_8(mht_8_v, 345, "", "./tensorflow/compiler/xla/python/tpu_driver/tpu_driver.h", "~TpuDriver");
}

  virtual void QuerySystemInfo(SystemInfo* system_info) = 0;
  // Synchronous. Reset the state of the TPU driver. After Reset(), this TPU
  // driver object is no longer usable. Users must destroy this object and
  // create a new one.
  //
  // All running programs will be terminated and all allocations reset. All
  // events and buffer handles created prior to Reset() will be invalid, and any
  // use will result in undefined behavior.
  virtual xla::Status Reset() = 0;

  virtual std::unique_ptr<BufferHandle> Allocate(
      int32_t core_id, MemoryRegion region, int64_t num_bytes,
      absl::Span<Event* const> wait_for) = 0;
  virtual std::unique_ptr<BufferHandle> Allocate(
      int32_t core_id, MemoryRegion region, const xla::ShapeProto& shape,
      absl::Span<Event* const> wait_for) = 0;

  // Allocate a buffer representing a tuple of `children` buffers.
  //
  // The returned tuple buffer handle does not manage the memory of `children`:
  // all `children` buffer handles must outlive the last usage of this tuple
  // buffer handle. One way to guarantee that is to deallocate the tuple buffer
  // handle before deallocating any buffer handle in `children`.
  //
  // All `children` buffers must exist in the same `core_id` and `region`.
  // If `children` is empty, a zero-sized tuple will be allocated in `region`.
  virtual std::unique_ptr<BufferHandle> AllocateTuple(
      int32_t core_id, MemoryRegion region,
      absl::Span<BufferHandle* const> children,
      absl::Span<Event* const> wait_for) = 0;
  virtual std::shared_ptr<Event> Deallocate(
      std::unique_ptr<BufferHandle> handle,
      absl::Span<Event* const> wait_for) = 0;

  /* For buffers declared with an xla::ShapeProto rather than a raw size,
   * `src` must be laid out in consecutive row-major format for ingestion, and
   * each element must take up the number of bytes specified by the type.
   *
   * For example, for a [3,3,3] tensor with a Float32 type, the memory layout
   * would be as follows:
   *
   * [0,0,0], [0,0,1], [0,0,2], [0,1,0], [0,1,1], ..., [0,2,2], [1,0,0], ...
   * [1,2,2], [2,0,0], ..., [2,2,2],
   *
   * and the entire buffer will be 108 bytes (27 elements x 4 bytes).
   *
   * See
   * https://eli.thegreenplace.net/2015/memory-layout-of-multi-dimensional-arrays
   * for a more detailed description.
   *
   * `TransferFromDevice` will write out the shape back in this order as well.
   */
  virtual std::shared_ptr<Event> TransferToDevice(
      const void* src, BufferHandle* dst,
      absl::Span<Event* const> wait_for) = 0;
  virtual std::shared_ptr<Event> TransferFromDevice(
      const BufferHandle* src, void* dst,
      absl::Span<Event* const> wait_for) = 0;

  virtual std::shared_ptr<Event> TransferFromDeviceToDevice(
      const BufferHandle* src, BufferHandle* dst,
      absl::Span<Event* const> wait_for) = 0;

  virtual std::unique_ptr<CompiledProgramHandle> CompileProgram(
      const xla::HloProto& source, int32_t num_replicas,
      absl::Span<Event* const> wait_for) = 0;
  virtual std::unique_ptr<LoadedProgramHandle> LoadProgram(
      int32_t core_id, const CompiledProgramHandle* handle,
      absl::Span<Event* const> wait_for) = 0;
  virtual std::shared_ptr<Event> UnloadProgram(
      std::unique_ptr<LoadedProgramHandle> handle,
      absl::Span<Event* const> wait_for) = 0;
  virtual std::shared_ptr<Event> ExecuteProgram(
      LoadedProgramHandle* program, absl::Span<BufferHandle* const> inputs,
      absl::Span<BufferHandle* const> outputs,
      const xla::DeviceAssignmentProto& device_assignment,
      absl::Span<Event* const> wait_for) = 0;

  virtual std::unique_ptr<TpuLinearizer> GetLinearizer() { return nullptr; }
};

class TpuDriverRegistry {
 public:
  static xla::StatusOr<std::unique_ptr<TpuDriver>> Open(
      const TpuDriverConfig& config);
  static int RegisterDriver(
      const std::string& prefix,
      const std::function<xla::StatusOr<std::unique_ptr<TpuDriver>>(
          const TpuDriverConfig&)>& creator);
};

#define REGISTER_TPU_DRIVER(prefix, fn) \
  REGISTER_TPU_DRIVER_HELPER(__COUNTER__, prefix, fn)
#define REGISTER_TPU_DRIVER_HELPER(ctr, prefix, fn)   \
  static int register_tpu_driver_count_unused_##ctr = \
      ::tpu_driver::TpuDriverRegistry::RegisterDriver(prefix, fn);

}  // namespace tpu_driver

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_TPU_DRIVER_TPU_DRIVER_H_
