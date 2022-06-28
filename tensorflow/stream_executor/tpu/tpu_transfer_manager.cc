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
class MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_transfer_managerDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_transfer_managerDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_transfer_managerDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/stream_executor/tpu/tpu_transfer_manager.h"

#include <utility>

#include "absl/cleanup/cleanup.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/tpu/c_api_conversions.h"
#include "tensorflow/stream_executor/tpu/noncopyable_buffer.h"
#include "tensorflow/stream_executor/tpu/proto_helper.h"
#include "tensorflow/stream_executor/tpu/status_helper.h"
#include "tensorflow/stream_executor/tpu/tpu_executor.h"
#include "tensorflow/stream_executor/tpu/tpu_executor_c_api.h"
#include "tensorflow/stream_executor/tpu/tpu_platform.h"
#include "tensorflow/stream_executor/tpu/tpu_platform_id.h"

namespace tensorflow {
namespace tpu {

using Status = stream_executor::port::Status;
template <typename T>
using StatusOr = stream_executor::port::StatusOr<T>;

TpuTransferManager::TpuTransferManager() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_transfer_managerDTcc mht_0(mht_0_v, 211, "", "./tensorflow/stream_executor/tpu/tpu_transfer_manager.cc", "TpuTransferManager::TpuTransferManager");

  manager_ = tpu::ExecutorApiFn()->TpuTransferManager_NewFn();
}

TpuTransferManager::~TpuTransferManager() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_transfer_managerDTcc mht_1(mht_1_v, 218, "", "./tensorflow/stream_executor/tpu/tpu_transfer_manager.cc", "TpuTransferManager::~TpuTransferManager");

  tpu::ExecutorApiFn()->TpuTransferManager_FreeFn(manager_);
}

stream_executor::Platform::Id TpuTransferManager::PlatformId() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_transfer_managerDTcc mht_2(mht_2_v, 225, "", "./tensorflow/stream_executor/tpu/tpu_transfer_manager.cc", "TpuTransferManager::PlatformId");

  return GetTpuPlatformId();
}

xla::Shape TpuTransferManager::HostShapeToDeviceShape(
    const xla::Shape& host_shape) const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_transfer_managerDTcc mht_3(mht_3_v, 233, "", "./tensorflow/stream_executor/tpu/tpu_transfer_manager.cc", "TpuTransferManager::HostShapeToDeviceShape");

  XLA_Shape c_host_shape;
  XLA_Shape c_device_shape;

  ApiConverter::ToC(host_shape, &c_host_shape);

  tpu::ExecutorApiFn()->TpuTransferManager_HostShapeToDeviceShapeFn(
      manager_, &c_host_shape, &c_device_shape);
  xla::Shape device_shape = ApiConverter::FromC(&c_device_shape);
  ApiConverter::Destroy(&c_host_shape);
  ApiConverter::Destroy(&c_device_shape);
  return device_shape;
}

Status TpuTransferManager::TransferLiteralToDeviceAsync(
    stream_executor::Stream* stream, const xla::LiteralSlice& literal,
    const xla::ShapedBuffer& device_buffer,
    const TransferMetadata* transfer_metadata) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_transfer_managerDTcc mht_4(mht_4_v, 253, "", "./tensorflow/stream_executor/tpu/tpu_transfer_manager.cc", "TpuTransferManager::TransferLiteralToDeviceAsync");

  StatusHelper status;

  XLA_Literal c_literal;
  ApiConverter::ToC(literal, &c_literal);

  XLA_ShapedBuffer c_device_buffer;
  ApiConverter::ToC(device_buffer, &c_device_buffer);

  tpu::ExecutorApiFn()->TpuTransferManager_TransferLiteralToDeviceAsyncFn(
      manager_,
      TpuPlatform::GetRegisteredPlatform()->LookupStream(
          stream->implementation()),
      &c_literal, &c_device_buffer, status.c_status);
  ApiConverter::Destroy(&c_device_buffer);
  ApiConverter::Destroy(&c_literal);
  return status.status();
}

Status TpuTransferManager::TransferLiteralToInfeed(
    stream_executor::StreamExecutor* executor,
    const xla::LiteralSlice& literal) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_transfer_managerDTcc mht_5(mht_5_v, 277, "", "./tensorflow/stream_executor/tpu/tpu_transfer_manager.cc", "TpuTransferManager::TransferLiteralToInfeed");

  StatusHelper status;
  XLA_Literal c_literal;
  ApiConverter::ToC(literal, &c_literal);
  auto* tpu_executor = static_cast<TpuExecutor*>(executor->implementation());

  tpu::ExecutorApiFn()->TpuTransferManager_TransferLiteralToInfeedFn(
      manager_, tpu_executor->se_executor(), &c_literal, status.c_status);

  ApiConverter::Destroy(&c_literal);

  return status.status();
}

Status TpuTransferManager::TransferBuffersToInfeed(
    se::StreamExecutor* executor,
    const std::deque<tensorflow::tpu::NoncopyableBuffer>& buffers) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_transfer_managerDTcc mht_6(mht_6_v, 296, "", "./tensorflow/stream_executor/tpu/tpu_transfer_manager.cc", "TpuTransferManager::TransferBuffersToInfeed");

  StatusHelper status;
  auto* tpu_executor = static_cast<TpuExecutor*>(executor->implementation());

  std::vector<int64_t> buffers_size;
  std::vector<uint32_t*> buffers_array;

  buffers_size.reserve(buffers.size());
  buffers_array.reserve(buffers.size());

  for (int64_t i = 0; i < buffers.size(); ++i) {
    absl::Span<const uint32_t> span = buffers[i].const_data<uint32_t>();
    buffers_array.push_back(const_cast<uint32_t*>(span.data()));
    buffers_size.push_back(span.size());
  }

  tpu::ExecutorApiFn()->TpuTransferManager_TransferBuffersToInfeedFn(
      manager_, tpu_executor->se_executor(), buffers_array.data(),
      buffers_size.data(), buffers_size.size(), status.c_status);
  return status.status();
}

Status TpuTransferManager::TransferLiteralFromOutfeed(
    stream_executor::StreamExecutor* executor,
    xla::MutableBorrowingLiteral literal) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_transfer_managerDTcc mht_7(mht_7_v, 323, "", "./tensorflow/stream_executor/tpu/tpu_transfer_manager.cc", "TpuTransferManager::TransferLiteralFromOutfeed");

  StatusHelper status;
  XLA_Shape c_shape;
  XLA_Literal c_literal;
  auto* tpu_executor = static_cast<TpuExecutor*>(executor->implementation());

  ApiConverter::ToC(literal.shape(), &c_shape);
  ApiConverter::ToC(literal, &c_literal);

  tpu::ExecutorApiFn()->TpuTransferManager_TransferLiteralFromOutfeedFn(
      manager_, tpu_executor->se_executor(), &c_shape, &c_literal,
      status.c_status);

  ApiConverter::Destroy(&c_shape);
  ApiConverter::Destroy(&c_literal);

  return status.status();
}

Status TpuTransferManager::ResetDevices(
    absl::Span<stream_executor::StreamExecutor* const> executor) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_transfer_managerDTcc mht_8(mht_8_v, 346, "", "./tensorflow/stream_executor/tpu/tpu_transfer_manager.cc", "TpuTransferManager::ResetDevices");

  StatusHelper status;
  std::vector<SE_StreamExecutor*> se;
  se.reserve(executor.size());
  for (int64_t i = 0; i < executor.size(); ++i) {
    se.push_back(static_cast<TpuExecutor*>(executor[i]->implementation())
                     ->se_executor());
  }

  tpu::ExecutorApiFn()->TpuTransferManager_ResetDevicesFn(
      manager_, se.data(), executor.size(), status.c_status);
  return status.status();
}

struct TransferFromDeviceState {
  std::atomic<int64_t> remaining_transfers;
  TF_Status* overall_status =
      tpu::ExecutorApiFn()->TpuStatus_NewFn();  // OK or the first error
  std::function<void(Status)> done;

  void TransferFinished(TF_Status* status) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_transfer_managerDTcc mht_9(mht_9_v, 369, "", "./tensorflow/stream_executor/tpu/tpu_transfer_manager.cc", "TransferFinished");

    if (!tpu::ExecutorApiFn()->TpuStatus_OkFn(status) &&
        tpu::ExecutorApiFn()->TpuStatus_OkFn(overall_status)) {
      std::swap(overall_status, status);
    }
    tpu::ExecutorApiFn()->TpuStatus_FreeFn(status);

    if (--remaining_transfers == 0) {
      done(StatusHelper::FromC(overall_status));
      tpu::ExecutorApiFn()->TpuStatus_FreeFn(overall_status);
      delete this;
    }
  }
};

void TransferLiteralFromDeviceTrampoline(void* ctx, TF_Status* status) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_transfer_managerDTcc mht_10(mht_10_v, 387, "", "./tensorflow/stream_executor/tpu/tpu_transfer_manager.cc", "TransferLiteralFromDeviceTrampoline");

  reinterpret_cast<TransferFromDeviceState*>(ctx)->TransferFinished(status);
}

void TpuTransferManager::TransferLiteralFromDevice(
    stream_executor::Stream* stream, const xla::ShapedBuffer& device_buffer,
    xla::MutableBorrowingLiteral literal, std::function<void(Status)> done,
    const TransferMetadata* transfer_metadata) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_transfer_managerDTcc mht_11(mht_11_v, 397, "", "./tensorflow/stream_executor/tpu/tpu_transfer_manager.cc", "TpuTransferManager::TransferLiteralFromDevice");

  TransferFromDeviceState* state = new TransferFromDeviceState;
  state->remaining_transfers = 1;
  state->done = done;
  XLA_ShapedBuffer c_device_buffer;
  ApiConverter::ToC(device_buffer, &c_device_buffer);
  XLA_Literal c_literal;
  ApiConverter::ToC(literal, &c_literal);

  tpu::ExecutorApiFn()->TpuTransferManager_TransferLiteralFromDeviceFn(
      manager_,
      TpuPlatform::GetRegisteredPlatform()->LookupStream(
          stream->implementation()),
      &c_device_buffer, &c_literal, TransferLiteralFromDeviceTrampoline, state);
  ApiConverter::Destroy(&c_device_buffer);
  ApiConverter::Destroy(&c_literal);
}

int64_t TpuTransferManager::GetByteSizeRequirement(
    const xla::Shape& shape) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_transfer_managerDTcc mht_12(mht_12_v, 419, "", "./tensorflow/stream_executor/tpu/tpu_transfer_manager.cc", "TpuTransferManager::GetByteSizeRequirement");

  XLA_Shape c_shape;
  ApiConverter::ToC(shape, &c_shape);

  int64_t size_in_bytes =
      tpu::ExecutorApiFn()->TpuTransferManager_GetByteSizeRequirementFn(
          manager_, &c_shape);

  ApiConverter::Destroy(&c_shape);
  return size_in_bytes;
}

StatusOr<xla::Shape> TpuTransferManager::ChooseCompactLayoutForShape(
    const xla::Shape& host_shape) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_transfer_managerDTcc mht_13(mht_13_v, 435, "", "./tensorflow/stream_executor/tpu/tpu_transfer_manager.cc", "TpuTransferManager::ChooseCompactLayoutForShape");

  XLA_Shape c_host_shape;
  ApiConverter::ToC(host_shape, &c_host_shape);
  XLA_Shape c_output;
  StatusHelper status;
  tpu::ExecutorApiFn()->TpuTransferManager_ChooseCompactLayoutForShapeFn(
      manager_, &c_host_shape, &c_output, status.c_status);
  // TODO(skyewm): use a scoped version of XLA_Shape
  ApiConverter::Destroy(&c_host_shape);
  if (!status.status().ok()) {
    ApiConverter::Destroy(&c_output);
    return status.status();
  }
  xla::Shape output = ApiConverter::FromC(&c_output);
  ApiConverter::Destroy(&c_output);
  return output;
}

bool TpuTransferManager::CanShapedBufferBeAccessedNow(
    stream_executor::StreamExecutor* executor,
    const xla::ShapedBuffer& device_buffer) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_transfer_managerDTcc mht_14(mht_14_v, 458, "", "./tensorflow/stream_executor/tpu/tpu_transfer_manager.cc", "TpuTransferManager::CanShapedBufferBeAccessedNow");

  auto* tpu_executor = down_cast<TpuExecutor*>(executor->implementation());
  XLA_ShapedBuffer c_device_buffer;
  ApiConverter::ToC(device_buffer, &c_device_buffer);
  auto cleanup = absl::MakeCleanup(
      [&c_device_buffer]() { ApiConverter::Destroy(&c_device_buffer); });
  return tpu::ExecutorApiFn()
      ->TpuTransferManager_CanShapedBufferBeAccessedNowFn(
          manager_, tpu_executor->se_executor(), &c_device_buffer);
}

bool TpuTransferManager::CanBufferBeAccessedNow(
    se::StreamExecutor* executor,
    const se::DeviceMemoryBase& device_buffer) const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_transfer_managerDTcc mht_15(mht_15_v, 474, "", "./tensorflow/stream_executor/tpu/tpu_transfer_manager.cc", "TpuTransferManager::CanBufferBeAccessedNow");

  auto* tpu_executor = down_cast<TpuExecutor*>(executor->implementation());
  SE_DeviceMemoryBase c_device_buffer{const_cast<void*>(device_buffer.opaque()),
                                      device_buffer.size(),
                                      device_buffer.payload()};
  return tpu::ExecutorApiFn()->TpuTransferManager_CanBufferBeAccessedNowFn(
      manager_, tpu_executor->se_executor(), &c_device_buffer);
}

Status TpuTransferManager::WriteSingleTupleIndexTable(
    stream_executor::Stream* stream,
    absl::Span<const stream_executor::DeviceMemoryBase> elements,
    const xla::Shape& shape, stream_executor::DeviceMemoryBase* region) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_transfer_managerDTcc mht_16(mht_16_v, 489, "", "./tensorflow/stream_executor/tpu/tpu_transfer_manager.cc", "TpuTransferManager::WriteSingleTupleIndexTable");

  CHECK_GT(elements.size(), 0);
  SE_DeviceMemoryBase* elements_bases =
      new SE_DeviceMemoryBase[elements.size()];
  for (int i = 0; i < elements.size(); i++) {
    elements_bases[i] =
        SE_DeviceMemoryBase{const_cast<void*>(elements[i].opaque()),
                            elements[i].size(), elements[i].payload()};
  }
  XLA_Shape c_shape;
  ApiConverter::ToC(shape, &c_shape);
  SE_DeviceMemoryBase region_base{region->opaque(), region->size(),
                                  region->payload()};
  StatusHelper status;

  tpu::ExecutorApiFn()->TpuTransferManager_WriteSingleTupleIndexTableFn(
      manager_,
      TpuPlatform::GetRegisteredPlatform()->LookupStream(
          stream->implementation()),
      elements_bases, elements.size(), &c_shape, &region_base, status.c_status);

  delete[] elements_bases;
  ApiConverter::Destroy(&c_shape);
  return status.status();
}

Status TpuTransferManager::LinearizeToBuffers(
    const xla::LiteralSlice& literal,
    std::deque<tensorflow::tpu::NoncopyableBuffer>* buffers) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_transfer_managerDTcc mht_17(mht_17_v, 520, "", "./tensorflow/stream_executor/tpu/tpu_transfer_manager.cc", "TpuTransferManager::LinearizeToBuffers");

  XLA_Literal c_literal;
  ApiConverter::ToC(literal, &c_literal);

  char** buffers_array;
  int64_t* buffers_size;
  int64_t buffers_array_size;
  StatusHelper status;

  tpu::ExecutorApiFn()->TpuTransferManager_LinearizeToBuffersFn(
      manager_, &c_literal, &buffers_array, &buffers_size, &buffers_array_size,
      status.c_status);

  for (int64_t i = 0; i < buffers_array_size; ++i) {
    tpu::NoncopyableBuffer buf(buffers_size[i]);
    memcpy(buf.mutable_data<uint8_t>().data(), buffers_array[i],
           buffers_size[i]);
    buffers->push_back(std::move(buf));
  }

  tpu::ExecutorApiFn()->TpuTransferManager_FreeBuffersFn(
      buffers_array, buffers_size, buffers_array_size);

  ApiConverter::Destroy(&c_literal);
  return status.status();
}

Status TpuTransferManager::ReadDynamicShapes(se::Stream* stream,
                                             xla::ShapedBuffer* device_buffer,
                                             xla::Shape* device_shape) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_transfer_managerDTcc mht_18(mht_18_v, 552, "", "./tensorflow/stream_executor/tpu/tpu_transfer_manager.cc", "TpuTransferManager::ReadDynamicShapes");

  XLA_ShapedBuffer c_device_buffer;
  XLA_Shape c_device_shape;
  ApiConverter::ToC(*device_buffer, &c_device_buffer);
  ApiConverter::ToC(*device_shape, &c_device_shape);
  XLA_Shape c_updated_shape;
  StatusHelper status;
  ExecutorApiFn()->TpuTransferManager_ReadDynamicShapesFn(
      TpuPlatform::GetRegisteredPlatform()->LookupStream(
          stream->implementation()),
      &c_device_buffer, c_device_shape, &c_updated_shape, status.c_status);
  ApiConverter::Destroy(&c_device_buffer);
  ApiConverter::Destroy(&c_device_shape);
  if (!status.ok()) {
    return status.status();
  }
  *device_shape = ApiConverter::FromC(&c_updated_shape);
  ApiConverter::Destroy(&c_updated_shape);
  return Status::OK();
}

}  // namespace tpu
}  // namespace tensorflow
