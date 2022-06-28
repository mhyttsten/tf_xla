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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_xfeedDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_xfeedDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_xfeedDTcc() {
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

#include "tensorflow/compiler/xla/service/cpu/cpu_xfeed.h"

#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/compiler/xla/service/cpu/xfeed_manager.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/notification.h"

namespace xla {
namespace {

class CpuInfeedBuffer : public cpu::runtime::XfeedBuffer {
 public:
  explicit CpuInfeedBuffer(int32_t length)
      : length_(length), buffer_(new char[length]) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_xfeedDTcc mht_0(mht_0_v, 215, "", "./tensorflow/compiler/xla/service/cpu/cpu_xfeed.cc", "CpuInfeedBuffer");
}
  ~CpuInfeedBuffer() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_xfeedDTcc mht_1(mht_1_v, 219, "", "./tensorflow/compiler/xla/service/cpu/cpu_xfeed.cc", "~CpuInfeedBuffer");
 delete[] buffer_; }

  int32_t length() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_xfeedDTcc mht_2(mht_2_v, 224, "", "./tensorflow/compiler/xla/service/cpu/cpu_xfeed.cc", "length");
 return length_; }
  void* data() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_xfeedDTcc mht_3(mht_3_v, 228, "", "./tensorflow/compiler/xla/service/cpu/cpu_xfeed.cc", "data");
 return buffer_; }
  void Done(StatusOr<Shape> /*shape*/) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_xfeedDTcc mht_4(mht_4_v, 232, "", "./tensorflow/compiler/xla/service/cpu/cpu_xfeed.cc", "Done");
 delete this; }

 private:
  int32_t length_;
  char* buffer_;
};

class CpuOutfeedBuffer : public cpu::runtime::XfeedBuffer {
 public:
  CpuOutfeedBuffer(void* destination, int32_t length)
      : destination_(destination), length_(length) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_xfeedDTcc mht_5(mht_5_v, 245, "", "./tensorflow/compiler/xla/service/cpu/cpu_xfeed.cc", "CpuOutfeedBuffer");
}

  StatusOr<Shape> WaitForNotification() {
    done_.WaitForNotification();
    return status_;
  }

  int32_t length() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_xfeedDTcc mht_6(mht_6_v, 255, "", "./tensorflow/compiler/xla/service/cpu/cpu_xfeed.cc", "length");
 return length_; }
  void* data() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_xfeedDTcc mht_7(mht_7_v, 259, "", "./tensorflow/compiler/xla/service/cpu/cpu_xfeed.cc", "data");
 return destination_; }
  void Done(StatusOr<Shape> shape) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_xfeedDTcc mht_8(mht_8_v, 263, "", "./tensorflow/compiler/xla/service/cpu/cpu_xfeed.cc", "Done");

    status_ = std::move(shape);
    done_.Notify();
  }

 private:
  void* destination_;
  int32_t length_;
  StatusOr<Shape> status_;
  tensorflow::Notification done_;
};

// Transfers infeed data to device. InfeedBuffer->Done() must be called to
// clean up the memory allocated for InfeedBuffer.
StatusOr<cpu::runtime::XfeedBuffer*> TransferBufferToInfeedInternal(
    int64_t size, const void* source) {
  if (size > std::numeric_limits<int32_t>::max()) {
    return InvalidArgument("CPU infeed of %d bytes exceeds maximum of %d bytes",
                           size, std::numeric_limits<int32_t>::max());
  }

  if (size <= 0) {
    return InvalidArgument("Infeed shape must have positive size; got %d",
                           size);
  }

  auto size_32 = static_cast<int32_t>(size);
  auto queued_buffer = new CpuInfeedBuffer(size_32);
  std::memcpy(queued_buffer->data(), source, size);

  return queued_buffer;
}

Status TransferBufferToInfeed(int device_ordinal, int64_t size,
                              const void* source) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_xfeedDTcc mht_9(mht_9_v, 300, "", "./tensorflow/compiler/xla/service/cpu/cpu_xfeed.cc", "TransferBufferToInfeed");

  TF_ASSIGN_OR_RETURN(cpu::runtime::XfeedBuffer * buffer,
                      TransferBufferToInfeedInternal(size, source));

  cpu::runtime::XfeedManager* xfeed_manager =
      cpu::runtime::GetXfeedManager(device_ordinal);
  xfeed_manager->infeed()->EnqueueBuffersAtomically({buffer});

  return Status::OK();
}

StatusOr<Shape> TransferBuffersFromOutfeedInternal(
    int device_ordinal, absl::Span<const std::pair<void*, int64_t>> buffer_data,
    bool is_tuple) {
  std::vector<std::unique_ptr<CpuOutfeedBuffer>> buffers;
  for (auto b : buffer_data) {
    int64_t size = b.second;
    if (size > std::numeric_limits<int32_t>::max()) {
      return InvalidArgument("Outfeed shape is too large: needs %d bytes",
                             size);
    }

    if (size < 0) {
      return InvalidArgument(
          "Outfeed shape must have non-negative size; got %d", size);
    }

    auto size_32 = static_cast<int32_t>(size);
    VLOG(2)
        << "Enqueueing outfeed buffer (for the device to populate) of length "
        << size_32 << "B";
    buffers.push_back(absl::make_unique<CpuOutfeedBuffer>(b.first, size_32));
  }

  std::vector<cpu::runtime::XfeedBuffer*> buffer_pointers;
  buffer_pointers.reserve(buffers.size());
  for (auto& b : buffers) {
    buffer_pointers.push_back(b.get());
  }

  cpu::runtime::XfeedManager* xfeed_manager =
      cpu::runtime::GetXfeedManager(device_ordinal);
  xfeed_manager->outfeed()->EnqueueBuffersAtomically(buffer_pointers);
  VLOG(2) << "Waiting for buffer to be notified as populated.";
  std::vector<Shape> outfed_shapes;
  outfed_shapes.reserve(buffers.size());
  for (auto& buffer : buffers) {
    TF_ASSIGN_OR_RETURN(Shape outfed_shape, buffer->WaitForNotification());
    outfed_shapes.push_back(std::move(outfed_shape));
  }
  if (is_tuple) {
    return ShapeUtil::MakeTupleShape(outfed_shapes);
  }
  TF_RET_CHECK(outfed_shapes.size() == 1);
  return std::move(outfed_shapes[0]);
}

StatusOr<Shape> TransferArrayBufferFromOutfeed(int device_ordinal,
                                               void* destination,
                                               int64_t size_bytes) {
  return TransferBuffersFromOutfeedInternal(
      device_ordinal, {{destination, size_bytes}}, /*is_tuple=*/false);
}

StatusOr<Shape> TransferTupleBuffersFromOutfeed(
    int device_ordinal,
    absl::Span<const std::pair<void*, int64_t>> buffer_data) {
  return TransferBuffersFromOutfeedInternal(device_ordinal, buffer_data,
                                            /*is_tuple=*/true);
}
}  // namespace

Status TransferLiteralToInfeedOnCpu(int device_ordinal,
                                    const LiteralSlice& literal) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_xfeedDTcc mht_10(mht_10_v, 376, "", "./tensorflow/compiler/xla/service/cpu/cpu_xfeed.cc", "TransferLiteralToInfeedOnCpu");

  const Shape& shape = literal.shape();
  VLOG(2) << "Transferring literal to infeed with shape: "
          << ShapeUtil::HumanString(shape);

  if (!shape.IsTuple()) {
    int64_t size = cpu::runtime::GetByteSizeRequirement(shape, sizeof(void*));
    return TransferBufferToInfeed(device_ordinal, size, literal.untyped_data());
  }

  if (ShapeUtil::IsNestedTuple(shape)) {
    return Unimplemented(
        "Infeed with a nested tuple shape is not supported: %s",
        ShapeUtil::HumanString(literal.shape()));
  }

  // For a tuple, we transfer each of its elements to the device and
  // enqueue the resulting destination device addresses with the
  // infeed manager.
  std::vector<cpu::runtime::XfeedBuffer*> buffers;
  buffers.reserve(ShapeUtil::TupleElementCount(shape));
  auto cleanup = tensorflow::gtl::MakeCleanup([&buffers]() {
    for (cpu::runtime::XfeedBuffer* b : buffers) {
      b->Done(Cancelled("Failed to infeed buffer to device."));
    }
  });

  for (int64_t i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
    const Shape& tuple_element_shape = ShapeUtil::GetSubshape(shape, {i});
    int64_t tuple_element_size = cpu::runtime::GetByteSizeRequirement(
        tuple_element_shape, sizeof(void*));
    TF_ASSIGN_OR_RETURN(cpu::runtime::XfeedBuffer * buffer,
                        TransferBufferToInfeedInternal(
                            tuple_element_size, literal.untyped_data({i})));
    buffers.push_back(buffer);
  }

  cpu::runtime::XfeedManager* xfeed_manager =
      cpu::runtime::GetXfeedManager(device_ordinal);
  xfeed_manager->infeed()->EnqueueBuffersAtomically(buffers);

  cleanup.release();
  return Status::OK();
}

Status TransferLiteralFromOutfeedOnCpu(int device_ordinal,
                                       MutableBorrowingLiteral literal) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_xfeedDTcc mht_11(mht_11_v, 425, "", "./tensorflow/compiler/xla/service/cpu/cpu_xfeed.cc", "TransferLiteralFromOutfeedOnCpu");

  if (!literal.shape().IsTuple()) {
    int64_t size =
        cpu::runtime::GetByteSizeRequirement(literal.shape(), sizeof(void*));
    // Note: OSS build didn't like implicit conversion from
    // literal.shape().dimensions() to the array slice on 2017-07-10.
    absl::Span<const int64_t> dimensions(
        absl::bit_cast<const int64_t*>(literal.shape().dimensions().data()),
        literal.shape().dimensions().size());
    TF_ASSIGN_OR_RETURN(Shape received_shape,
                        TransferArrayBufferFromOutfeed(
                            device_ordinal, literal.untyped_data(), size));
    TF_RET_CHECK(ShapeUtil::Compatible(received_shape, literal.shape()))
        << "Shape received from outfeed "
        << ShapeUtil::HumanString(received_shape)
        << " did not match the shape that was requested for outfeed: "
        << ShapeUtil::HumanString(literal.shape());
    TF_RET_CHECK(size == cpu::runtime::GetByteSizeRequirement(received_shape,
                                                              sizeof(void*)));
    *literal.mutable_shape_do_not_use() = received_shape;
    return Status::OK();
  }

  if (ShapeUtil::IsNestedTuple(literal.shape())) {
    return Unimplemented(
        "Nested tuple outfeeds are not yet implemented on CPU.");
  }

  std::vector<std::pair<void*, int64_t>> buffer_data;
  for (int i = 0; i < literal.shape().tuple_shapes_size(); ++i) {
    const Shape& tuple_element_shape =
        ShapeUtil::GetTupleElementShape(literal.shape(), i);
    int64_t size = cpu::runtime::GetByteSizeRequirement(tuple_element_shape,
                                                        sizeof(void*));
    buffer_data.push_back({literal.untyped_data({i}), size});
  }

  TF_ASSIGN_OR_RETURN(Shape received_shape, TransferTupleBuffersFromOutfeed(
                                                device_ordinal, buffer_data));

  TF_RET_CHECK(ShapeUtil::Compatible(received_shape, literal.shape()))
      << "Shape received from outfeed "
      << ShapeUtil::HumanString(received_shape)
      << " did not match the shape that was requested for outfeed: "
      << ShapeUtil::HumanString(literal.shape());
  TF_RET_CHECK(
      cpu::runtime::GetByteSizeRequirement(literal.shape(), sizeof(void*)) ==
      cpu::runtime::GetByteSizeRequirement(received_shape, sizeof(void*)));

  TF_RET_CHECK(ShapeUtil::Equal(literal.shape(), literal.shape()));
  return Status::OK();
}

Status ReadDynamicShapesOnCpu(
    ShapedBuffer* device_buffer, Shape* device_shape,
    HloCostAnalysis::ShapeSizeFunction shape_size_fn) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePScpuPScpu_xfeedDTcc mht_12(mht_12_v, 483, "", "./tensorflow/compiler/xla/service/cpu/cpu_xfeed.cc", "ReadDynamicShapesOnCpu");

  TF_RET_CHECK(device_shape->is_dynamic());
  Shape original_device_shape = *device_shape;
  TF_RETURN_IF_ERROR(device_buffer->buffers().ForEachMutableElementWithStatus(
      [&](const ShapeIndex& index, se::DeviceMemoryBase* buffer) {
        const Shape& buffer_shape =
            ShapeUtil::GetSubshape(*device_shape, index);
        if (buffer_shape.IsTuple()) {
          return Status::OK();
        }
        Shape& device_sub_shape =
            *ShapeUtil::GetMutableSubshape(device_shape, index);
        if (device_sub_shape.is_static()) {
          return Status::OK();
        }
        void* memory = buffer->opaque();

        // Read the dynamic shape metadata from the device stream.
        Shape buffer_shape_static = ShapeUtil::MakeStaticShape(buffer_shape);
        const int64_t offset = shape_size_fn(buffer_shape_static);
        int64_t metadata_size = shape_size_fn(buffer_shape) - offset;
        if (metadata_size == 0) {
          return InvalidArgument("Dynamic shape metadata size should not be 0");
        }
        auto buffer_8 = static_cast<int8_t*>(memory);
        auto metadata_buffer = reinterpret_cast<int32_t*>(buffer_8 + offset);

        // Update shape size from metadata.
        for (int64_t i = 0; i < device_sub_shape.rank(); ++i) {
          device_sub_shape.mutable_dimensions()[i] = metadata_buffer[i];
        }
        return Status::OK();
      }));
  device_shape->clear_dynamic_dimensions();

  TF_RET_CHECK(ShapeUtil::DynamicShapeIsCompatible(*device_shape,
                                                   original_device_shape));
  return Status::OK();
}
}  // namespace xla
