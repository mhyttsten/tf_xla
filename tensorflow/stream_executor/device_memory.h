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

// Suite of types that represent device memory allocations. These are
// allocated by the StreamExecutor interface, which produces values appropriate
// for the underlying platform (whether it be CUDA or OpenCL).
//
// The untyped base class (like a device void*) is DeviceMemoryBase, which can
// be specialized for a given allocation type (like a device T*) using
// DeviceMemory<T>.

#ifndef TENSORFLOW_STREAM_EXECUTOR_DEVICE_MEMORY_H_
#define TENSORFLOW_STREAM_EXECUTOR_DEVICE_MEMORY_H_
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
class MHTracer_DTPStensorflowPSstream_executorPSdevice_memoryDTh {
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
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memoryDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPSdevice_memoryDTh() {
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


#include <stddef.h>

#include "tensorflow/stream_executor/platform/port.h"

namespace perftools {
namespace gputools {

// Temporarily pull stream_executor into perftools::gputools while we migrate
// code to the new namespace.  TODO(b/77980417): Remove this once we've
// completed the migration.
using namespace stream_executor;  // NOLINT[build/namespaces]

}  // namespace gputools
}  // namespace perftools

namespace stream_executor {

class DeviceMemoryAllocator;
class StreamExecutor;

// void*-analogous device memory allocation. For the typed variation, see
// DeviceMemory<T>.
//
// This is effectively a two-tuple of a pointer and size; however, note that the
// pointer may not be to the virtual address itself -- in OpenCL the pointer is
// to a cl_mem handle that describes the device allocation. Therefore,
// DeviceMemoryBase::opaque does not necessarily produce a pointer that can be
// referenced directly, so use it with caution.
//
// Thread-compatible.
class DeviceMemoryBase {
 public:
  // Default constructor instantiates a null-pointed, zero-sized device memory
  // region. An opaque pointer may be provided -- see header for details on the
  // opacity of that pointer.
  explicit DeviceMemoryBase(void *opaque = nullptr, uint64_t size = 0)
      : opaque_(opaque), size_(size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memoryDTh mht_0(mht_0_v, 232, "", "./tensorflow/stream_executor/device_memory.h", "DeviceMemoryBase");
}

  // Returns whether the backing memory is the null pointer.
  // A `== nullptr` convenience method is also provided.
  bool is_null() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memoryDTh mht_1(mht_1_v, 239, "", "./tensorflow/stream_executor/device_memory.h", "is_null");
 return opaque_ == nullptr; }
  bool operator==(std::nullptr_t other) const { return is_null(); }
  bool operator!=(std::nullptr_t other) const { return !is_null(); }

  // Provides a partial order between device memory values.
  //
  // This operator is provided so that this object can be used as a key in an
  // ordered map.
  bool operator<(const DeviceMemoryBase &other) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memoryDTh mht_2(mht_2_v, 250, "", "./tensorflow/stream_executor/device_memory.h", "operator<");

    return opaque() < other.opaque();
  }

  // Returns the size, in bytes, for the backing memory.
  uint64_t size() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memoryDTh mht_3(mht_3_v, 258, "", "./tensorflow/stream_executor/device_memory.h", "size");
 return size_; }

  // Warning: note that the pointer returned is not necessarily directly to
  // device virtual address space, but is platform-dependent.
  void *opaque() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memoryDTh mht_4(mht_4_v, 265, "", "./tensorflow/stream_executor/device_memory.h", "opaque");
 return opaque_; }
  const void *opaque() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memoryDTh mht_5(mht_5_v, 269, "", "./tensorflow/stream_executor/device_memory.h", "opaque");
 return opaque_; }

  // Returns the payload of this memory region.
  uint64_t payload() const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memoryDTh mht_6(mht_6_v, 275, "", "./tensorflow/stream_executor/device_memory.h", "payload");
 return payload_; }

  // Sets payload to given value.
  void SetPayload(uint64_t payload) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memoryDTh mht_7(mht_7_v, 281, "", "./tensorflow/stream_executor/device_memory.h", "SetPayload");
 payload_ = payload; }

  // Returns whether the two DeviceMemoryBase segments are identical (both in
  // their opaque pointer and size).
  bool IsSameAs(const DeviceMemoryBase &other) const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memoryDTh mht_8(mht_8_v, 288, "", "./tensorflow/stream_executor/device_memory.h", "IsSameAs");

    return opaque() == other.opaque() && size() == other.size();
  }

 protected:
  friend class StreamExecutor;

  // Resets the internal values of the opaque pointer and number of bytes in the
  // memory region, just as in the constructor.
  void Reset(void *opaque, uint64_t bytes) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memoryDTh mht_9(mht_9_v, 300, "", "./tensorflow/stream_executor/device_memory.h", "Reset");

    opaque_ = opaque;
    size_ = bytes;
  }

 private:
  void *opaque_;  // Platform-dependent value representing allocated memory.
  uint64_t size_;         // Size in bytes of this allocation.
  uint64_t payload_ = 0;  // Payload data associated with this allocation.
};

// Typed wrapper around "void *"-like DeviceMemoryBase.
//
// For example, DeviceMemory<int> is a simple wrapper around DeviceMemoryBase
// that represents one or more integers in Device memory.
//
// Thread-compatible.
template <typename ElemT>
class DeviceMemory final : public DeviceMemoryBase {
 public:
  // Default constructor instantiates a null-pointed, zero-sized memory region.
  DeviceMemory() : DeviceMemoryBase(nullptr, 0) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memoryDTh mht_10(mht_10_v, 324, "", "./tensorflow/stream_executor/device_memory.h", "DeviceMemory");
}
  explicit DeviceMemory(std::nullptr_t) : DeviceMemory() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memoryDTh mht_11(mht_11_v, 328, "", "./tensorflow/stream_executor/device_memory.h", "DeviceMemory");
}

  // Typed device memory regions may be constructed from untyped device memory
  // regions, this effectively amounts to a cast from a void*.
  explicit DeviceMemory(const DeviceMemoryBase &other)
      : DeviceMemoryBase(const_cast<DeviceMemoryBase &>(other).opaque(),
                         other.size()) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memoryDTh mht_12(mht_12_v, 337, "", "./tensorflow/stream_executor/device_memory.h", "DeviceMemory");

    SetPayload(other.payload());
  }

  // Returns the number of elements of type ElemT that constitute this
  // allocation.
  uint64_t ElementCount() const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memoryDTh mht_13(mht_13_v, 346, "", "./tensorflow/stream_executor/device_memory.h", "ElementCount");
 return size() / sizeof(ElemT); }

  // Returns whether this is a single-element allocation.
  bool IsScalar() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memoryDTh mht_14(mht_14_v, 352, "", "./tensorflow/stream_executor/device_memory.h", "IsScalar");
 return ElementCount() == 1; }

  // Create a typed area of DeviceMemory with a given opaque pointer and the
  // quantity of bytes in the allocation. This function is broken out to
  // distinguish bytes from an element count.
  static DeviceMemory<ElemT> MakeFromByteSize(void *opaque, uint64_t bytes) {
    return DeviceMemory<ElemT>(opaque, bytes);
  }

  // Resets the DeviceMemory data, in MakeFromByteSize fashion.
  // This simply clobbers the prior values.
  void ResetFromByteSize(void *opaque, uint64_t bytes) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memoryDTh mht_15(mht_15_v, 366, "", "./tensorflow/stream_executor/device_memory.h", "ResetFromByteSize");

    // TODO(leary) when NVCC is eliminated we can add this check (and the
    // logging include it requires).
    // CHECK_EQ(0, bytes % sizeof(ElemT));
    DeviceMemoryBase::Reset(opaque, bytes);
  }

  // ------------------------------------------------------------

 protected:
  // This constructor is solely used from derived classes; it is made protected
  // because it accepts a byte-size instead of an element count, which could
  // potentially be misused given the ElementCount() nature of this interface.
  //
  // In order to specify the desire to use byte size instead of element count
  // explicitly, use MakeFromByteSize.
  DeviceMemory(void *opaque, uint64_t size) : DeviceMemoryBase(opaque, size) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memoryDTh mht_16(mht_16_v, 385, "", "./tensorflow/stream_executor/device_memory.h", "DeviceMemory");
}
};

// A class to encapsulate the type and size of a dynamic shared memory
// buffer. Because the buffer exists solely on the device and is not copyable
// to the host, memory objects of this type do not maintain buffer pointers
// on the host.
template <typename ElemT>
class SharedDeviceMemory final : public DeviceMemoryBase {
 public:
  explicit SharedDeviceMemory(uint64_t elem_count)
      : DeviceMemoryBase(nullptr, elem_count * kElemSize) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memoryDTh mht_17(mht_17_v, 399, "", "./tensorflow/stream_executor/device_memory.h", "SharedDeviceMemory");
}

  static constexpr size_t kElemSize = sizeof(ElemT);

  // Returns the number of elements of type ElemT that constitute this
  // allocation.
  uint64_t ElementCount() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memoryDTh mht_18(mht_18_v, 408, "", "./tensorflow/stream_executor/device_memory.h", "ElementCount");
 return size() / kElemSize; }

  // Returns whether this is a single-element allocation.
  bool IsScalar() const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSstream_executorPSdevice_memoryDTh mht_19(mht_19_v, 414, "", "./tensorflow/stream_executor/device_memory.h", "IsScalar");
 return ElementCount() == 1; }
};

// Host-side representation of packed-and-aligned vector datatypes on the device
// side. Since these can appear in device kernel signatures, we support
// launching them with these datatypes in launch signatures.

struct Float2 {
  float x, y;
};

struct Float4 {
  Float2 xz, yw;
};

struct Double2 {
  double x, y;
};

static_assert(sizeof(Float2) == 2 * sizeof(float), "Float2 must be packed");
static_assert(sizeof(Float4) == 4 * sizeof(float), "Float4 must be packed");
static_assert(sizeof(Double2) == 2 * sizeof(double), "Double2 must be packed");

}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_DEVICE_MEMORY_H_
