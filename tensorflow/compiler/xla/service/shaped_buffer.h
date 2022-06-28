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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SHAPED_BUFFER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SHAPED_BUFFER_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTh() {
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
#include <ostream>
#include <string>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"

namespace xla {

class ScopedShapedBuffer;

// Class which encapsulates a buffer or set of buffers containing data of a
// particular XLA shape.
class ShapedBuffer {
 public:
  // Construct a ShapedBuffer with null DeviceMemoryBases at each index. The
  // shape of the data on the host and the device may differ because the device
  // may have a different representation for different data types. Therefore,
  // both the on-host and on-device shape are required. The on-device shape
  // determines the number of device allocations (DeviceMemoryBase) held by the
  // ShapedBuffer.
  ShapedBuffer(Shape on_device_shape, int device_ordinal);

  // TODO(b/170310047): remove this overload.
  ShapedBuffer(Shape on_host_shape, Shape on_device_shape, int device_ordinal);

  // Movable, but not copyable.
  ShapedBuffer(ShapedBuffer&& s);
  ShapedBuffer& operator=(ShapedBuffer&&);
  ShapedBuffer(const ShapedBuffer&) = delete;
  ShapedBuffer& operator=(const ShapedBuffer&) = delete;

  // Prevent (some forms of) accidental object slicing.
  ShapedBuffer(const ScopedShapedBuffer&) = delete;
  ShapedBuffer& operator=(const ScopedShapedBuffer&) = delete;

  virtual ~ShapedBuffer();

  // Returns the shape of the on-host representation of the data held by this
  // ShapedBuffer.
  const Shape& on_host_shape() const {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTh mht_0(mht_0_v, 233, "", "./tensorflow/compiler/xla/service/shaped_buffer.h", "on_host_shape");
 return on_host_shape_; }

  // Returns the shape of the on-device representation of the data held by this
  // ShapedBuffer.
  const Shape& on_device_shape() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTh mht_1(mht_1_v, 240, "", "./tensorflow/compiler/xla/service/shaped_buffer.h", "on_device_shape");
 return on_device_shape_; }

  int device_ordinal() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTh mht_2(mht_2_v, 245, "", "./tensorflow/compiler/xla/service/shaped_buffer.h", "device_ordinal");
 return device_ordinal_; }

  // Return the root buffer of the shape (shape index {}).
  const se::DeviceMemoryBase& root_buffer() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTh mht_3(mht_3_v, 251, "", "./tensorflow/compiler/xla/service/shaped_buffer.h", "root_buffer");

    return buffer(/*index=*/{});
  }

  // Returns the buffer at the given shape index where index is defined as in
  // ShapeUtil::GetSubshape.
  const se::DeviceMemoryBase& buffer(const ShapeIndex& index) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTh mht_4(mht_4_v, 260, "", "./tensorflow/compiler/xla/service/shaped_buffer.h", "buffer");

    return buffers_.element(index);
  }

  // Sets the device memory buffer at the given index.
  void set_buffer(const se::DeviceMemoryBase& buffer, const ShapeIndex& index) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTh mht_5(mht_5_v, 268, "", "./tensorflow/compiler/xla/service/shaped_buffer.h", "set_buffer");

    *buffers_.mutable_element(index) = buffer;
  }

  // Sets all buffers.
  //
  // Precondition: buffers.shape == on_device_shape_
  void set_buffers(ShapeTree<se::DeviceMemoryBase> buffers) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTh mht_6(mht_6_v, 278, "", "./tensorflow/compiler/xla/service/shaped_buffer.h", "set_buffers");

    CHECK(ShapeUtil::Equal(buffers.shape(), on_device_shape_));
    buffers_ = std::move(buffers);
    buffers_.replace_shape_ptr(on_device_shape_);
  }

  // Reset the shape of this shaped buffer and underlying buffer structure.
  //
  // Precondition: EqualStructure(this->on_device_shape_, on_device_shape).
  void set_shapes(const Shape& on_device_shape) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTh mht_7(mht_7_v, 290, "", "./tensorflow/compiler/xla/service/shaped_buffer.h", "set_shapes");

    CHECK(ShapeUtil::EqualStructure(on_device_shape, on_device_shape_))
        << "Structures are not the same. new: " << on_device_shape
        << ", old: " << on_device_shape_;
    on_host_shape_ = ShapeUtil::DeviceShapeToHostShape(on_device_shape);
    on_device_shape_ = on_device_shape;
    buffers_.replace_shape_ptr(on_device_shape_);
  }
  // TODO(b/170310047): remove this overload.
  void set_shapes(const Shape& on_host_shape, const Shape& on_device_shape) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTh mht_8(mht_8_v, 302, "", "./tensorflow/compiler/xla/service/shaped_buffer.h", "set_shapes");

    set_shapes(on_device_shape);
  }

  // Returns the underlying ShapeTree containing all the device addresses in the
  // ShapedBuffer.
  const ShapeTree<se::DeviceMemoryBase>& buffers() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTh mht_9(mht_9_v, 311, "", "./tensorflow/compiler/xla/service/shaped_buffer.h", "buffers");
 return buffers_; }
  ShapeTree<se::DeviceMemoryBase>& buffers() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTh mht_10(mht_10_v, 315, "", "./tensorflow/compiler/xla/service/shaped_buffer.h", "buffers");
 return buffers_; }

  StatusOr<ShapedBuffer> SubShapedBuffer(const ShapeIndex& index) const;

  // Set all device memory pointers in the object to null.
  void clear();

  std::string ToString() const;

 protected:
  Shape on_host_shape_;

  // The shape of the data on the device.
  Shape on_device_shape_;

  // The device the memory is allocated on.
  int device_ordinal_;

  // The tree of device buffers. Its shape is on_device_shape().
  ShapeTree<se::DeviceMemoryBase> buffers_;
};

std::ostream& operator<<(std::ostream& out, const ShapedBuffer& buffer);

// ScopedShapedBuffer takes allocated buffers as inputs, and deallocates on
// destruction. This class represents an owning wrapper around `ShapedBuffer`.
//
// TODO(timshen): Remove inheritance between ScopedShapedBuffer and
// ShapedBuffer.  There should never be a need to consider a ScopedShapedBuffer
// as a ShapedBuffer, because in that case we should just be able to pass around
// our ShapeTree<DeviceMemoryBase>.  Inheritance only adds complexity.  See
// discussion in cl/192849370.
class ScopedShapedBuffer : public ShapedBuffer {
 public:
  // Creates a ScopedShapedBuffer with null DeviceMemoryBases at each index.
  explicit ScopedShapedBuffer(Shape on_device_shape,
                              se::DeviceMemoryAllocator* allocator,
                              int device_ordinal);
  // TODO(b/170310047): remove this overload.
  explicit ScopedShapedBuffer(Shape on_host_shape, Shape on_device_shape,
                              se::DeviceMemoryAllocator* allocator,
                              int device_ordinal);

  // Create a ScopedShapedBuffer by taking over the memory from the incoming
  // ShapedBuffer.
  explicit ScopedShapedBuffer(ShapedBuffer shaped_buffer,
                              se::DeviceMemoryAllocator* allocator);

  // Movable, but not copyable.
  ScopedShapedBuffer(ScopedShapedBuffer&& s);
  ScopedShapedBuffer& operator=(ScopedShapedBuffer&&);
  ScopedShapedBuffer(const ScopedShapedBuffer&) = delete;
  ScopedShapedBuffer& operator=(const ScopedShapedBuffer&) = delete;

  // All buffers in the shape are deallocated on destruction.
  ~ScopedShapedBuffer() override;

  // Return the allocator used to allocate the device memory held in this
  // ScopedShapedBuffer.
  se::DeviceMemoryAllocator* memory_allocator() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTh mht_11(mht_11_v, 377, "", "./tensorflow/compiler/xla/service/shaped_buffer.h", "memory_allocator");
 return allocator_; }

  // Sets the device memory buffer at the given index.
  //
  // If the given buffer's device memory is non-null, its device_ordinal and
  // allocator must match those in `this`.
  void set_buffer(se::OwningDeviceMemory buffer, const ShapeIndex& index) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTh mht_12(mht_12_v, 386, "", "./tensorflow/compiler/xla/service/shaped_buffer.h", "set_buffer");

    if (!buffer.is_null()) {
      CHECK_EQ(buffer.device_ordinal(), device_ordinal());
      CHECK_EQ(buffer.allocator(), allocator_);
      *buffers_.mutable_element(index) = buffer.Release();
    } else {
      *buffers_.mutable_element(index) = se::DeviceMemoryBase();
    }
  }

  // Like unique_ptr::release(), creates and returns a regular ShapedBuffer from
  // this ScopedShapedBuffer, without freeing any of the associated memory.
  //
  // It's the caller's job to ensure that the memory contained therein is freed.
  ABSL_MUST_USE_RESULT ShapedBuffer release();

  // Extracts the sub-tree rooted at 'index' and returns a ScopedShapedBuffer
  // that holds ownership of the subtree. Sets the buffers corresponding to the
  // subtree to null in 'this'.
  ScopedShapedBuffer TakeSubTree(ShapeIndexView index);

 protected:
  void Deallocate();

  se::DeviceMemoryAllocator* allocator_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SHAPED_BUFFER_H_
