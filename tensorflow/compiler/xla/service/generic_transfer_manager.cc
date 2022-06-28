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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgeneric_transfer_managerDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgeneric_transfer_managerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgeneric_transfer_managerDTcc() {
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

#include "tensorflow/compiler/xla/service/generic_transfer_manager.h"

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {

GenericTransferManager::GenericTransferManager(se::Platform::Id platform_id,
                                               size_t pointer_size)
    : platform_id_(platform_id), pointer_size_(pointer_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgeneric_transfer_managerDTcc mht_0(mht_0_v, 205, "", "./tensorflow/compiler/xla/service/generic_transfer_manager.cc", "GenericTransferManager::GenericTransferManager");
}

se::Platform::Id GenericTransferManager::PlatformId() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgeneric_transfer_managerDTcc mht_1(mht_1_v, 210, "", "./tensorflow/compiler/xla/service/generic_transfer_manager.cc", "GenericTransferManager::PlatformId");

  return platform_id_;
}

Status GenericTransferManager::WriteSingleTupleIndexTable(
    se::Stream* stream, absl::Span<const se::DeviceMemoryBase> elements,
    const Shape& shape, se::DeviceMemoryBase* region) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgeneric_transfer_managerDTcc mht_2(mht_2_v, 219, "", "./tensorflow/compiler/xla/service/generic_transfer_manager.cc", "GenericTransferManager::WriteSingleTupleIndexTable");

  TF_RET_CHECK(elements.size() == ShapeUtil::TupleElementCount(shape));

  auto element_pointers = std::make_shared<std::vector<const void*>>();
  element_pointers->reserve(elements.size());
  for (const se::DeviceMemoryBase& element : elements) {
    element_pointers->push_back(element.opaque());
  }
  TF_RETURN_IF_ERROR(TransferBufferToDevice(
      stream, GetByteSizeRequirement(shape), element_pointers->data(), region));
  // Ensure the buffer is transferred before we destroy element_pointers.
  stream->ThenDoHostCallback([element_pointers{std::move(element_pointers)}]() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgeneric_transfer_managerDTcc mht_3(mht_3_v, 233, "", "./tensorflow/compiler/xla/service/generic_transfer_manager.cc", "lambda");

    /* holds reference to element_pointers in closure */
  });
  return Status::OK();
}

void GenericTransferManager::TransferLiteralFromDevice(
    se::Stream* stream, const ShapedBuffer& device_buffer,
    MutableBorrowingLiteral literal, std::function<void(Status)> done,
    const TransferMetadata* /*transfer_metadata*/) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgeneric_transfer_managerDTcc mht_4(mht_4_v, 245, "", "./tensorflow/compiler/xla/service/generic_transfer_manager.cc", "GenericTransferManager::TransferLiteralFromDevice");

  VLOG(2) << "transferring literal from device ordinal "
          << stream->parent()->device_ordinal()
          << "; device buffer: " << device_buffer;
  Status status = [&]() -> Status {
    TF_RET_CHECK(stream->parent()->device_ordinal() ==
                 device_buffer.device_ordinal());

    TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
        device_buffer.on_device_shape(),
        [&](const Shape& subshape, const ShapeIndex& index) -> Status {
          if (subshape.IsArray()) {
            stream->ThenMemcpy(
                /*host_dst=*/literal.untyped_data(index),
                /*gpu_src=*/device_buffer.buffer(index),
                // With bounded dynamic shapes, the shape of the device buffer
                // (bounded allocation) can be bigger than the literal.
                /*size=*/
                GetByteSizeRequirement(
                    ShapeUtil::GetSubshape(literal.shape(), index)));
          }
          return Status::OK();
        }));
    return Status::OK();
  }();
  if (!status.ok()) {
    done(status);
    return;
  }
  done(stream->BlockHostUntilDone());
}

Status GenericTransferManager::TransferLiteralToDeviceAsync(
    se::Stream* stream, const LiteralSlice& literal,
    const ShapedBuffer& device_buffer,
    const TransferMetadata* /*transfer_metadata*/) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgeneric_transfer_managerDTcc mht_5(mht_5_v, 283, "", "./tensorflow/compiler/xla/service/generic_transfer_manager.cc", "GenericTransferManager::TransferLiteralToDeviceAsync");

  const Shape& shape = literal.shape();
  VLOG(2) << "transferring literal shape to device: "
          << ShapeUtil::HumanString(shape)
          << "; device buffer: " << device_buffer;

  TF_RET_CHECK(
      ShapeUtil::Compatible(literal.shape(), device_buffer.on_device_shape()));
  TF_RET_CHECK(stream->parent()->device_ordinal() ==
               device_buffer.device_ordinal());

  TF_RETURN_IF_ERROR(WriteTupleIndexTablesAsync(stream, device_buffer));

  return ShapeUtil::ForEachSubshapeWithStatus(
      device_buffer.on_device_shape(),
      [&](const Shape& device_subshape, const ShapeIndex& index) -> Status {
        se::DeviceMemoryBase device_memory = device_buffer.buffer(index);
        if (device_subshape.IsArray()) {
          TF_RET_CHECK(GetByteSizeRequirement(device_subshape) ==
                       device_memory.size());
          // Element is array-shaped: transfer array data to device buffer.
          const auto subliteral = LiteralSlice(literal, index);
          Literal relayed_out_literal;
          const void* source;
          if (LayoutUtil::Equal(device_subshape.layout(),
                                subliteral.shape().layout())) {
            source = subliteral.untyped_data();
            return TransferBufferToDevice(
                stream,
                /*size=*/GetByteSizeRequirement(device_subshape), source,
                &device_memory);
          } else {
            // Relayout data before transferring.
            relayed_out_literal = subliteral.Relayout(device_subshape.layout(),
                                                      /*shape_index=*/{});
            source = relayed_out_literal.untyped_data();
            TF_RETURN_IF_ERROR(TransferBufferToDevice(
                stream,
                /*size=*/GetByteSizeRequirement(device_subshape), source,
                &device_memory));
            return stream->BlockHostUntilDone();
          }
        }
        return Status::OK();
      });
}

Status GenericTransferManager::TransferLiteralToInfeed(
    se::StreamExecutor* executor, const LiteralSlice& literal) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgeneric_transfer_managerDTcc mht_6(mht_6_v, 334, "", "./tensorflow/compiler/xla/service/generic_transfer_manager.cc", "GenericTransferManager::TransferLiteralToInfeed");

  return Unimplemented("Generic transfer to Infeed");
}

Status GenericTransferManager::TransferLiteralFromOutfeed(
    se::StreamExecutor* executor, MutableBorrowingLiteral literal) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgeneric_transfer_managerDTcc mht_7(mht_7_v, 342, "", "./tensorflow/compiler/xla/service/generic_transfer_manager.cc", "GenericTransferManager::TransferLiteralFromOutfeed");

  return Unimplemented("Generic transfer from Outfeed");
}

Status GenericTransferManager::ResetDevices(
    absl::Span<se::StreamExecutor* const>
    /*executors*/) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgeneric_transfer_managerDTcc mht_8(mht_8_v, 351, "", "./tensorflow/compiler/xla/service/generic_transfer_manager.cc", "GenericTransferManager::ResetDevices");

  return Unimplemented(
      "Device reset is not yet supported on this platform (b/30481585)");
}

int64_t GenericTransferManager::GetByteSizeRequirement(
    const Shape& shape) const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgeneric_transfer_managerDTcc mht_9(mht_9_v, 360, "", "./tensorflow/compiler/xla/service/generic_transfer_manager.cc", "GenericTransferManager::GetByteSizeRequirement");

  if (shape.is_static() || shape.IsTuple()) {
    return ShapeUtil::ByteSizeOf(shape, pointer_size_);
  }
  int64_t metadata_size = sizeof(int32_t) * shape.dimensions_size();
  return ShapeUtil::ByteSizeOf(shape, pointer_size_) + metadata_size;
}

}  // namespace xla
