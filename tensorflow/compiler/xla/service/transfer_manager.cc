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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStransfer_managerDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStransfer_managerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStransfer_managerDTcc() {
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

#include "tensorflow/compiler/xla/service/transfer_manager.h"

#include <functional>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/maybe_owning_device_memory.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/notification.h"

using absl::StrCat;

namespace xla {

/* static */ absl::Mutex TransferManager::platform_transfer_manager_mutex_(
    absl::kConstInit);

/* static */ absl::flat_hash_map<se::Platform::Id, TransferManager::State>*
TransferManager::GetPlatformTransferManagers() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStransfer_managerDTcc mht_0(mht_0_v, 211, "", "./tensorflow/compiler/xla/service/transfer_manager.cc", "TransferManager::GetPlatformTransferManagers");

  static auto* r =
      new absl::flat_hash_map<se::Platform::Id, TransferManager::State>;
  return r;
}

TransferManager::TransferMetadata::~TransferMetadata() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStransfer_managerDTcc mht_1(mht_1_v, 220, "", "./tensorflow/compiler/xla/service/transfer_manager.cc", "TransferManager::TransferMetadata::~TransferMetadata");
}

StatusOr<Literal> TransferManager::TransferLiteralFromDevice(
    se::Stream* stream, const ShapedBuffer& device_buffer,
    const TransferMetadata* transfer_metadata) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStransfer_managerDTcc mht_2(mht_2_v, 227, "", "./tensorflow/compiler/xla/service/transfer_manager.cc", "TransferManager::TransferLiteralFromDevice");

  StatusOr<Literal> ret;

  se::Stream* substream = stream->GetOrCreateSubStream();
  substream->ThenWaitFor(stream);
  auto cleanup = tensorflow::gtl::MakeCleanup(
      [&]() { stream->ReturnSubStream(substream); });

  tensorflow::Notification n;
  Status s;
  Literal literal(device_buffer.on_host_shape());
  TransferLiteralFromDevice(
      substream, device_buffer, &literal,
      [&](Status status) {
        s = status;
        n.Notify();
      },
      transfer_metadata);
  n.WaitForNotification();
  if (!s.ok()) {
    return s;
  }
  return std::move(literal);
}

Status TransferManager::TransferLiteralFromDevice(
    se::Stream* stream, const ShapedBuffer& device_buffer,
    const MutableBorrowingLiteral& literal,
    const TransferMetadata* transfer_metadata) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStransfer_managerDTcc mht_3(mht_3_v, 258, "", "./tensorflow/compiler/xla/service/transfer_manager.cc", "TransferManager::TransferLiteralFromDevice");

  se::Stream* substream = stream->GetOrCreateSubStream();
  auto cleanup = tensorflow::gtl::MakeCleanup(
      [&]() { stream->ReturnSubStream(substream); });

  Status ret;
  tensorflow::Notification n;
  TransferLiteralFromDevice(
      substream, device_buffer, literal,
      [&](Status status) {
        ret = status;
        n.Notify();
      },
      transfer_metadata);
  n.WaitForNotification();
  return ret;
}

Status TransferManager::TransferLiteralToDevice(
    se::Stream* stream, const LiteralSlice& literal,
    const ShapedBuffer& device_buffer,
    const TransferMetadata* transfer_metadata) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStransfer_managerDTcc mht_4(mht_4_v, 282, "", "./tensorflow/compiler/xla/service/transfer_manager.cc", "TransferManager::TransferLiteralToDevice");

  // Implement the synchronous version by waiting on the asynchronous version.
  // Use a substream so that if we are called from a HostCallback we don't
  // deadlock.
  se::Stream* substream = stream->GetOrCreateSubStream();
  substream->ThenWaitFor(stream);
  auto cleanup = tensorflow::gtl::MakeCleanup(
      [&]() { stream->ReturnSubStream(substream); });
  TF_RETURN_IF_ERROR(TransferLiteralToDeviceAsync(
      substream, literal, device_buffer, transfer_metadata));
  return substream->BlockHostUntilDone();
}

StatusOr<Literal> TransferManager::TransferArrayFromDevice(
    se::Stream* stream, const Shape& shape, const se::DeviceMemoryBase& source,
    const TransferMetadata* transfer_metadata) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStransfer_managerDTcc mht_5(mht_5_v, 300, "", "./tensorflow/compiler/xla/service/transfer_manager.cc", "TransferManager::TransferArrayFromDevice");

  StatusOr<Literal> ret;
  // Implement the synchronous version by waiting on the asynchronous version.
  // Use a substream so that if we are called from a HostCallback we don't
  // deadlock.
  se::Stream* substream = stream->GetOrCreateSubStream();
  auto cleanup = tensorflow::gtl::MakeCleanup(
      [&]() { stream->ReturnSubStream(substream); });

  tensorflow::Notification n;
  Literal literal(shape);
  Status s;
  TransferArrayFromDevice(
      substream, shape, source, &literal,
      [&](Status status) {
        s = status;
        n.Notify();
      },
      transfer_metadata);
  n.WaitForNotification();
  if (!s.ok()) {
    return s;
  }
  return std::move(literal);
}

Status TransferManager::TransferArrayToDevice(
    se::Stream* stream, const LiteralSlice& literal,
    const se::DeviceMemoryBase& dest,
    const TransferMetadata* transfer_metadata) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStransfer_managerDTcc mht_6(mht_6_v, 332, "", "./tensorflow/compiler/xla/service/transfer_manager.cc", "TransferManager::TransferArrayToDevice");

  // Implement the synchronous version by waiting on the asynchronous version.
  // Use a substream so that if we are called from a HostCallback we don't
  // deadlock.
  se::Stream* substream = stream->GetOrCreateSubStream();
  auto cleanup = tensorflow::gtl::MakeCleanup(
      [&]() { stream->ReturnSubStream(substream); });
  TF_RETURN_IF_ERROR(
      TransferArrayToDeviceAsync(substream, literal, dest, transfer_metadata));
  return substream->BlockHostUntilDone();
}

Status TransferManager::TransferArrayToDeviceAsync(
    se::Stream* stream, const LiteralSlice& literal,
    const se::DeviceMemoryBase& dest,
    const TransferMetadata* transfer_metadata) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStransfer_managerDTcc mht_7(mht_7_v, 350, "", "./tensorflow/compiler/xla/service/transfer_manager.cc", "TransferManager::TransferArrayToDeviceAsync");

  const Shape on_device_shape = HostShapeToDeviceShape(literal.shape());
  TF_RET_CHECK(on_device_shape.IsArray())
      << "On-device representation of "
      << ShapeUtil::HumanString(literal.shape())
      << " is not an array: " << ShapeUtil::HumanString(on_device_shape);
  if (dest.size() < GetByteSizeRequirement(on_device_shape)) {
    return FailedPrecondition(
        "Allocation on device not large enough for array: "
        "%d < %d",
        dest.size(), GetByteSizeRequirement(on_device_shape));
  }
  ShapedBuffer shaped_buffer(on_device_shape,
                             stream->parent()->device_ordinal());
  shaped_buffer.set_buffer(dest, /*index=*/{});
  return TransferLiteralToDevice(stream, literal, shaped_buffer,
                                 transfer_metadata);
}

void TransferManager::TransferArrayFromDevice(
    se::Stream* stream, const Shape& shape, const se::DeviceMemoryBase& source,
    const MutableBorrowingLiteral& literal, std::function<void(Status)> done,
    const TransferMetadata* transfer_metadata) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStransfer_managerDTcc mht_8(mht_8_v, 375, "", "./tensorflow/compiler/xla/service/transfer_manager.cc", "TransferManager::TransferArrayFromDevice");

  if (!Shape::Equal().MinorToMajorOnlyInLayout()(HostShapeToDeviceShape(shape),
                                                 shape)) {
    auto error = StrCat("Shape ", ShapeUtil::HumanString(shape),
                        " has a differently shaped representation on-device: ",
                        ShapeUtil::HumanString(HostShapeToDeviceShape(shape)));
    return done(FailedPrecondition("%s", error));
  }
  if (source.size() < GetByteSizeRequirement(shape)) {
    return done(
        FailedPrecondition("Allocation on device not large enough for array: "
                           "%d < %d",
                           source.size(), GetByteSizeRequirement(shape)));
  }
  ShapedBuffer shaped_buffer(shape, stream->parent()->device_ordinal());
  shaped_buffer.set_buffer(source, /*index=*/{});
  return TransferLiteralFromDevice(stream, shaped_buffer, literal,
                                   std::move(done), transfer_metadata);
}

Status TransferManager::ReadDynamicShapes(se::Stream* stream,
                                          ShapedBuffer* device_buffer,
                                          Shape* device_shape) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStransfer_managerDTcc mht_9(mht_9_v, 400, "", "./tensorflow/compiler/xla/service/transfer_manager.cc", "TransferManager::ReadDynamicShapes");

  DCHECK(device_shape->is_dynamic());
  Shape original_device_shape = *device_shape;
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());

  TF_ASSIGN_OR_RETURN(auto compiler,
                      Compiler::GetForPlatform(stream->parent()->platform()));
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

        // Read the dynamic shape metadata from the device stream.  The dynamic
        // shape itself is stored at the end of the buffer.
        auto shape_size_fn = compiler->ShapeSizeBytesFunction();
        Shape buffer_shape_static = ShapeUtil::MakeStaticShape(buffer_shape);
        const int64_t offset = shape_size_fn(buffer_shape_static);
        int64_t metadata_size = shape_size_fn(buffer_shape) - offset;
        if (metadata_size == 0) {
          return InvalidArgument("Dynamic shape metadata size should not be 0");
        }
        auto buffer_8 = se::DeviceMemory<uint8_t>(*buffer);
        auto metadata_buffer =
            stream->parent()->GetSubBuffer(&buffer_8, offset, metadata_size);
        TF_ASSIGN_OR_RETURN(
            auto metadata,
            TransferArrayFromDevice(
                stream,
                ShapeUtil::MakeShape(S32, {buffer_shape.dimensions_size()}),
                metadata_buffer));

        // Update shape size from metadata.
        for (int64_t i = 0; i < metadata.element_count(); ++i) {
          device_sub_shape.mutable_dimensions()[i] = metadata.Get<int32_t>({i});
        }
        return Status::OK();
      }));
  device_shape->clear_dynamic_dimensions();

  TF_RET_CHECK(ShapeUtil::DynamicShapeIsCompatible(*device_shape,
                                                   original_device_shape));
  return Status::OK();
}

/* static */ void TransferManager::RegisterTransferManager(
    se::Platform::Id platform_id,
    TransferManagerCreationFunction creation_function) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStransfer_managerDTcc mht_10(mht_10_v, 457, "", "./tensorflow/compiler/xla/service/transfer_manager.cc", "TransferManager::RegisterTransferManager");

  absl::MutexLock lock(&TransferManager::platform_transfer_manager_mutex_);
  auto* managers = GetPlatformTransferManagers();
  CHECK(managers->find(platform_id) == managers->end());
  (*managers)[platform_id].creation_function = creation_function;
}

/* static */ StatusOr<TransferManager*> TransferManager::GetForPlatform(
    const se::Platform* platform) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStransfer_managerDTcc mht_11(mht_11_v, 468, "", "./tensorflow/compiler/xla/service/transfer_manager.cc", "TransferManager::GetForPlatform");

  absl::MutexLock lock(&TransferManager::platform_transfer_manager_mutex_);
  auto* managers = GetPlatformTransferManagers();

  auto it = managers->find(platform->id());
  if (it == managers->end()) {
    return NotFound(
        "could not find registered transfer manager for platform %s -- check "
        "target linkage",
        platform->Name());
  }

  if (it->second.manager == nullptr) {
    // Lazily create the transfer manager the first time it is needed
    it->second.manager = (*it->second.creation_function)();
  }

  return it->second.manager.get();
}

Status TransferManager::WriteTupleIndexTables(
    se::Stream* stream, const ShapedBuffer& device_buffer) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStransfer_managerDTcc mht_12(mht_12_v, 492, "", "./tensorflow/compiler/xla/service/transfer_manager.cc", "TransferManager::WriteTupleIndexTables");

  TF_RETURN_IF_ERROR(WriteTupleIndexTablesAsync(stream, device_buffer));
  return stream->BlockHostUntilDone();
}

Status TransferManager::WriteTupleIndexTablesAsync(
    se::Stream* stream, const ShapedBuffer& device_buffer) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStransfer_managerDTcc mht_13(mht_13_v, 501, "", "./tensorflow/compiler/xla/service/transfer_manager.cc", "TransferManager::WriteTupleIndexTablesAsync");

  VLOG(2) << "Writing tuple index tables for " << device_buffer;

  return ShapeUtil::ForEachSubshapeWithStatus(
      device_buffer.on_device_shape(),
      [&](const Shape& device_subshape, const ShapeIndex& index) -> Status {
        if (device_subshape.IsTuple() &&
            ShapeUtil::TupleElementCount(device_subshape) > 0) {
          se::DeviceMemoryBase device_memory = device_buffer.buffer(index);
          TF_RET_CHECK(GetByteSizeRequirement(device_subshape) ==
                       device_memory.size());

          std::vector<se::DeviceMemoryBase> elements;
          ShapeIndex element_index = index;
          for (int64_t i = 0; i < ShapeUtil::TupleElementCount(device_subshape);
               ++i) {
            element_index.push_back(i);
            elements.push_back(device_buffer.buffer(element_index));
            element_index.pop_back();
          }
          return WriteSingleTupleIndexTable(stream, elements, device_subshape,
                                            &device_memory);
        }

        return Status::OK();
      });
}

Status TransferManager::WriteRootTupleIndexTable(
    se::Stream* stream, const ShapedBuffer& device_buffer) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStransfer_managerDTcc mht_14(mht_14_v, 533, "", "./tensorflow/compiler/xla/service/transfer_manager.cc", "TransferManager::WriteRootTupleIndexTable");

  TF_RET_CHECK(device_buffer.on_device_shape().IsTuple());
  if (ShapeUtil::TupleElementCount(device_buffer.on_device_shape()) == 0) {
    return Status::OK();
  }
  se::DeviceMemoryBase device_memory = device_buffer.buffer({});
  TF_RET_CHECK(GetByteSizeRequirement(device_buffer.on_device_shape()) ==
               device_memory.size());

  std::vector<se::DeviceMemoryBase> elements;
  for (int64_t i = 0;
       i < ShapeUtil::TupleElementCount(device_buffer.on_device_shape()); ++i) {
    elements.push_back(device_buffer.buffer({i}));
  }
  return WriteSingleTupleIndexTable(
      stream, elements, device_buffer.on_device_shape(), &device_memory);
}

Status TransferManager::WriteRootTupleIndexTable(
    se::Stream* stream, const ShapeTree<MaybeOwningDeviceMemory>& buffer_tree) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStransfer_managerDTcc mht_15(mht_15_v, 555, "", "./tensorflow/compiler/xla/service/transfer_manager.cc", "TransferManager::WriteRootTupleIndexTable");

  TF_RET_CHECK(buffer_tree.shape().IsTuple());
  if (ShapeUtil::TupleElementCount(buffer_tree.shape()) == 0) {
    return Status::OK();
  }
  se::DeviceMemoryBase device_memory =
      buffer_tree.element({}).AsDeviceMemoryBase();
  TF_RET_CHECK(GetByteSizeRequirement(buffer_tree.shape()) ==
               device_memory.size());

  std::vector<se::DeviceMemoryBase> elements;
  for (int64_t i = 0; i < ShapeUtil::TupleElementCount(buffer_tree.shape());
       ++i) {
    elements.push_back(buffer_tree.element({i}).AsDeviceMemoryBase());
  }
  return WriteSingleTupleIndexTable(stream, elements, buffer_tree.shape(),
                                    &device_memory);
}

Status TransferManager::TransferBufferFromDevice(
    se::Stream* stream, const se::DeviceMemoryBase& source, int64_t size,
    void* destination) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStransfer_managerDTcc mht_16(mht_16_v, 579, "", "./tensorflow/compiler/xla/service/transfer_manager.cc", "TransferManager::TransferBufferFromDevice");

  if (source.size() < size) {
    return FailedPrecondition(
        "Source allocation on device not large enough for data transfer: "
        "%d < %d",
        source.size(), size);
  }
  stream->ThenMemcpy(destination, source, size);
  return Status::OK();
}

Status TransferManager::TransferBufferToDevice(
    se::Stream* stream, int64_t size, const void* source,
    se::DeviceMemoryBase* destination) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStransfer_managerDTcc mht_17(mht_17_v, 595, "", "./tensorflow/compiler/xla/service/transfer_manager.cc", "TransferManager::TransferBufferToDevice");

  if (destination->size() < size) {
    return FailedPrecondition(
        "Destination allocation on device not large enough for data transfer: "
        "%d < %d",
        destination->size(), size);
  }
  stream->ThenMemcpy(destination, source, size);
  return Status::OK();
}

StatusOr<ScopedShapedBuffer> TransferManager::AllocateScopedShapedBuffer(
    const Shape& on_host_shape, se::DeviceMemoryAllocator* allocator,
    int device_ordinal, DeviceShapeRepresentationFn shape_representation_fn) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStransfer_managerDTcc mht_18(mht_18_v, 611, "", "./tensorflow/compiler/xla/service/transfer_manager.cc", "TransferManager::AllocateScopedShapedBuffer");

  if (!LayoutUtil::HasLayout(on_host_shape)) {
    return InvalidArgument("Shape must have a layout: %s",
                           ShapeUtil::HumanStringWithLayout(on_host_shape));
  }
  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShape(on_host_shape));
  Shape on_device_shape = (shape_representation_fn == nullptr)
                              ? HostShapeToDeviceShape(on_host_shape)
                              : shape_representation_fn(on_host_shape);
  TF_RET_CHECK(LayoutUtil::HasLayout(on_device_shape));

  ScopedShapedBuffer shaped_buffer(std::move(on_device_shape), allocator,
                                   device_ordinal);

  // Allocate an appropriate sized buffer for each element in the shape
  // including the tuple pointer arrays.
  for (auto& pair : shaped_buffer.buffers()) {
    const ShapeIndex& index = pair.first;
    se::DeviceMemoryBase& memory_base = pair.second;
    const Shape& subshape =
        ShapeUtil::GetSubshape(shaped_buffer.on_device_shape(), index);
    TF_ASSIGN_OR_RETURN(auto memory,
                        allocator->Allocate(shaped_buffer.device_ordinal(),
                                            GetByteSizeRequirement(subshape),
                                            /*retry_on_failure=*/true,
                                            subshape.layout().memory_space()));
    // Move the allocated buffer into the ScopedShapedBuffer, which owns it.
    memory_base = memory.Release();
  }

  return std::move(shaped_buffer);
}

StatusOr<Shape> TransferManager::ChooseCompactLayoutForShape(
    const Shape& host_shape) const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStransfer_managerDTcc mht_19(mht_19_v, 648, "", "./tensorflow/compiler/xla/service/transfer_manager.cc", "TransferManager::ChooseCompactLayoutForShape");

  return LayoutUtil::GetWithDefaultLayout(host_shape);
}

xla::Shape TransferManager::ChooseGoodInfeedLayout(const Shape& shape) const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePStransfer_managerDTcc mht_20(mht_20_v, 655, "", "./tensorflow/compiler/xla/service/transfer_manager.cc", "TransferManager::ChooseGoodInfeedLayout");

  return LayoutUtil::GetWithDefaultLayout(shape);
}

}  // namespace xla
