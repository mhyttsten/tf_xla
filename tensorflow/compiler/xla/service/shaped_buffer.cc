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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTcc() {
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

#include "tensorflow/compiler/xla/service/shaped_buffer.h"

#include <string>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

ShapedBuffer::ShapedBuffer(Shape on_device_shape, int device_ordinal)
    : on_device_shape_(std::move(on_device_shape)),
      device_ordinal_(device_ordinal),
      buffers_(&on_device_shape_) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTcc mht_0(mht_0_v, 206, "", "./tensorflow/compiler/xla/service/shaped_buffer.cc", "ShapedBuffer::ShapedBuffer");

  on_host_shape_ = ShapeUtil::DeviceShapeToHostShape(on_device_shape_);
}

ShapedBuffer::ShapedBuffer(Shape on_host_shape, Shape on_device_shape,
                           int device_ordinal)
    : ShapedBuffer(on_device_shape, device_ordinal) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTcc mht_1(mht_1_v, 215, "", "./tensorflow/compiler/xla/service/shaped_buffer.cc", "ShapedBuffer::ShapedBuffer");
}

ShapedBuffer::ShapedBuffer(ShapedBuffer&& s)
    : on_host_shape_(std::move(s.on_host_shape_)),
      on_device_shape_(std::move(s.on_device_shape_)),
      device_ordinal_(s.device_ordinal_),
      buffers_(std::move(s.buffers_)) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTcc mht_2(mht_2_v, 224, "", "./tensorflow/compiler/xla/service/shaped_buffer.cc", "ShapedBuffer::ShapedBuffer");

  // s.buffers_ has a pointer to s.on_device_shape_. When we move s.buffers_
  // into buffers_, we also need to update this pointer so that buffers_ doesn't
  // point into s.
  buffers_.replace_shape_ptr(on_device_shape_);
}

ShapedBuffer& ShapedBuffer::operator=(ShapedBuffer&& s) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTcc mht_3(mht_3_v, 234, "", "./tensorflow/compiler/xla/service/shaped_buffer.cc", "=");

  on_device_shape_ = std::move(s.on_device_shape_);
  on_host_shape_ = std::move(s.on_host_shape_);
  device_ordinal_ = s.device_ordinal_;
  buffers_ = std::move(s.buffers_);
  // buffers_ has a pointer to its on_device_shape_. When we move s.buffers_
  // into buffers_, we also need to update this pointer so that buffers_ doesn't
  // point into s.
  buffers_.replace_shape_ptr(on_device_shape_);
  return *this;
}

ShapedBuffer::~ShapedBuffer() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTcc mht_4(mht_4_v, 249, "", "./tensorflow/compiler/xla/service/shaped_buffer.cc", "ShapedBuffer::~ShapedBuffer");
}

StatusOr<ShapedBuffer> ShapedBuffer::SubShapedBuffer(
    const ShapeIndex& index) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTcc mht_5(mht_5_v, 255, "", "./tensorflow/compiler/xla/service/shaped_buffer.cc", "ShapedBuffer::SubShapedBuffer");

  TF_ASSIGN_OR_RETURN(const Shape* device_sub_shape,
                      ShapeUtil::TryGetSubshape(on_device_shape(), index));
  ShapedBuffer sub_shaped_buffer(*device_sub_shape, device_ordinal_);
  TF_ASSIGN_OR_RETURN(ShapeTree<se::DeviceMemoryBase> sub_buffers,
                      buffers_.SubShapeTree(index));
  sub_shaped_buffer.set_buffers(std::move(sub_buffers));
  return std::move(sub_shaped_buffer);
}

void ShapedBuffer::clear() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTcc mht_6(mht_6_v, 268, "", "./tensorflow/compiler/xla/service/shaped_buffer.cc", "ShapedBuffer::clear");

  for (auto& pair : buffers_) {
    // A default constructed DeviceMemoryBase is a null pointer.
    pair.second = se::DeviceMemoryBase();
  }
}

std::string ShapedBuffer::ToString() const {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTcc mht_7(mht_7_v, 278, "", "./tensorflow/compiler/xla/service/shaped_buffer.cc", "ShapedBuffer::ToString");

  std::string s =
      absl::StrCat("ShapedBuffer(", device_ordinal(),
                   "), on-device shape=" +
                       ShapeUtil::HumanStringWithLayout(on_device_shape()),
                   ":\n");
  ShapeUtil::ForEachSubshape(
      on_device_shape(),
      [this, &s](const Shape& subshape, const ShapeIndex& index) {
        std::string shape_str;
        if (subshape.IsTuple()) {
          shape_str = "tuple";
        } else {
          shape_str = ShapeUtil::HumanStringWithLayout(subshape);
        }
        const se::DeviceMemoryBase& memory = buffer(index);
        absl::StrAppendFormat(&s, "  %s%p (%d bytes) : %s\n",
                              std::string(index.size() * 2, ' '),
                              memory.opaque(), memory.size(), shape_str);
      });
  return s;
}

std::ostream& operator<<(std::ostream& out, const ShapedBuffer& buffer) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTcc mht_8(mht_8_v, 304, "", "./tensorflow/compiler/xla/service/shaped_buffer.cc", "operator<<");

  out << buffer.ToString();
  return out;
}

ScopedShapedBuffer::ScopedShapedBuffer(Shape on_device_shape,
                                       se::DeviceMemoryAllocator* allocator,
                                       int device_ordinal)
    : ShapedBuffer(std::move(on_device_shape), device_ordinal),
      allocator_(allocator) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTcc mht_9(mht_9_v, 316, "", "./tensorflow/compiler/xla/service/shaped_buffer.cc", "ScopedShapedBuffer::ScopedShapedBuffer");
}

ScopedShapedBuffer::ScopedShapedBuffer(Shape on_host_shape,
                                       Shape on_device_shape,
                                       se::DeviceMemoryAllocator* allocator,
                                       int device_ordinal)
    : ScopedShapedBuffer(std::move(on_device_shape), allocator,
                         device_ordinal) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTcc mht_10(mht_10_v, 326, "", "./tensorflow/compiler/xla/service/shaped_buffer.cc", "ScopedShapedBuffer::ScopedShapedBuffer");
}

ScopedShapedBuffer::ScopedShapedBuffer(ShapedBuffer shaped_buffer,
                                       se::DeviceMemoryAllocator* allocator)
    : ShapedBuffer(std::move(shaped_buffer)), allocator_(allocator) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTcc mht_11(mht_11_v, 333, "", "./tensorflow/compiler/xla/service/shaped_buffer.cc", "ScopedShapedBuffer::ScopedShapedBuffer");
}

ScopedShapedBuffer::ScopedShapedBuffer(ScopedShapedBuffer&& s)
    : ShapedBuffer(static_cast<ShapedBuffer&&>(s)), allocator_(s.allocator_) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTcc mht_12(mht_12_v, 339, "", "./tensorflow/compiler/xla/service/shaped_buffer.cc", "ScopedShapedBuffer::ScopedShapedBuffer");

  // Null out s.allocator_ so it doesn't try to free anything in its destructor.
  s.allocator_ = nullptr;
}

ScopedShapedBuffer& ScopedShapedBuffer::operator=(ScopedShapedBuffer&& s) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTcc mht_13(mht_13_v, 347, "", "./tensorflow/compiler/xla/service/shaped_buffer.cc", "=");

  Deallocate();

  *static_cast<ShapedBuffer*>(this) = std::move(static_cast<ShapedBuffer&>(s));
  allocator_ = s.allocator_;
  // Null out s.allocator_ so it doesn't try to free anything in its destructor.
  s.allocator_ = nullptr;
  return *this;
}

ScopedShapedBuffer::~ScopedShapedBuffer() {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTcc mht_14(mht_14_v, 360, "", "./tensorflow/compiler/xla/service/shaped_buffer.cc", "ScopedShapedBuffer::~ScopedShapedBuffer");
 Deallocate(); }

ShapedBuffer ScopedShapedBuffer::release() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTcc mht_15(mht_15_v, 365, "", "./tensorflow/compiler/xla/service/shaped_buffer.cc", "ScopedShapedBuffer::release");

  ShapedBuffer shaped_buffer(static_cast<ShapedBuffer&&>(*this));
  buffers_ = ShapeTree<se::DeviceMemoryBase>();
  return shaped_buffer;
}

void ScopedShapedBuffer::Deallocate() {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTcc mht_16(mht_16_v, 374, "", "./tensorflow/compiler/xla/service/shaped_buffer.cc", "ScopedShapedBuffer::Deallocate");

  // allocator_ will be null if we were moved-from.
  if (allocator_ == nullptr) {
    return;
  }
  // Deallocate all non-null buffers. A buffer may appear in more than one spot
  // in the shape (eg, a tuple with a repeated element) so keep track of what
  // has been deallocated.
  absl::flat_hash_set<void*> deallocated_ptrs;
  for (auto& pair : buffers_) {
    se::DeviceMemoryBase& memory_base = pair.second;
    if (!memory_base.is_null() &&
        deallocated_ptrs.insert(memory_base.opaque()).second) {
      TF_CHECK_OK(allocator_->Deallocate(device_ordinal(), memory_base));
    }
  }
}

ScopedShapedBuffer ScopedShapedBuffer::TakeSubTree(ShapeIndexView index) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSshaped_bufferDTcc mht_17(mht_17_v, 395, "", "./tensorflow/compiler/xla/service/shaped_buffer.cc", "ScopedShapedBuffer::TakeSubTree");

  const xla::Shape& sub_on_device_shape =
      xla::ShapeUtil::GetSubshape(on_device_shape(), {index});

  ScopedShapedBuffer output(sub_on_device_shape, memory_allocator(),
                            device_ordinal());
  auto src_it = buffers().find(index);
  auto dst_it = output.buffers().begin();
  while (dst_it != output.buffers().end()) {
    dst_it->second = src_it->second;
    src_it->second = tensorflow::se::DeviceMemoryBase(nullptr, 0);
    ++src_it;
    ++dst_it;
  }
  return output;
}

}  // namespace xla
