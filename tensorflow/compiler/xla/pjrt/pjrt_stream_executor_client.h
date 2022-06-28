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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_PJRT_STREAM_EXECUTOR_CLIENT_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_PJRT_STREAM_EXECUTOR_CLIENT_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh() {
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


#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/synchronization/notification.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/layout.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/pjrt/local_device_state.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_future.h"
#include "tensorflow/compiler/xla/pjrt/tracked_device_buffer.h"
#include "tensorflow/compiler/xla/pjrt/transpose.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_executable_run_options.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/stream_executor/stream.h"

namespace xla {

class PjRtStreamExecutorDevice : public PjRtDevice {
 public:
  explicit PjRtStreamExecutorDevice(
      int id, std::unique_ptr<LocalDeviceState> local_device_state,
      std::string device_kind, int process_index = 0)
      : id_(id),
        device_ordinal_(
            local_device_state ? local_device_state->device_ordinal() : -1),
        local_device_state_(std::move(local_device_state)),
        process_index_(process_index),
        device_kind_(std::move(device_kind)) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("device_kind: \"" + device_kind + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_0(mht_0_v, 241, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "PjRtStreamExecutorDevice");
}
  ~PjRtStreamExecutorDevice() override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_1(mht_1_v, 245, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "~PjRtStreamExecutorDevice");
}

  // Must set client exactly once.
  void SetClient(PjRtClient* client) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_2(mht_2_v, 251, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "SetClient");

    CHECK(client_ == nullptr);
    client_ = client;
  }

  int process_index() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_3(mht_3_v, 259, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "process_index");
 return process_index_; }

  // Return `platform_id` from client.
  PjRtPlatformId platform_id() const;

  // Return `platform_name` from client.
  absl::string_view platform_name() const;

  PjRtClient* client() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_4(mht_4_v, 270, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "client");
 return client_; }

  int id() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_5(mht_5_v, 275, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "id");
 return id_; }

  bool IsAddressable() const override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_6(mht_6_v, 280, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "IsAddressable");
 return device_ordinal_ != -1; }

  int local_hardware_id() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_7(mht_7_v, 285, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "local_hardware_id");
 return device_ordinal_; }

  // If this is a device local to this host, returns a LocalDeviceState object
  // that can be used to manipulate the device. Returns nullptr if the device is
  // not local to this host.
  LocalDeviceState* local_device_state() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_8(mht_8_v, 293, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "local_device_state");

    return local_device_state_.get();
  }

  // If this is a device local to this host, returns a LocalDeviceState object
  // that can be used to manipulate the device. Returns an error if the device
  // is not local to this host.
  StatusOr<LocalDeviceState*> GetLocalDeviceState() const;

  absl::string_view device_kind() const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_9(mht_9_v, 305, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "device_kind");
 return device_kind_; }

  std::string DebugString() const override;

  Status TransferToInfeed(const LiteralSlice& literal) override;

  Status TransferFromOutfeed(MutableBorrowingLiteral literal) override;

 private:
  const int id_;
  const int device_ordinal_;  // -1 means not local.
  const std::unique_ptr<LocalDeviceState> local_device_state_;
  const int process_index_;
  const std::string device_kind_;
  PjRtClient* client_ = nullptr;
};

class PjRtStreamExecutorClient : public PjRtClient {
 public:
  // `allocator` may null, in which case the platform default allocator is used.
  explicit PjRtStreamExecutorClient(
      std::string platform_name, LocalClient* client,
      std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices,
      int process_index, std::unique_ptr<se::DeviceMemoryAllocator> allocator,
      std::unique_ptr<tensorflow::Allocator> host_memory_allocator,
      bool should_stage_host_to_device_transfers,
      std::unique_ptr<gpu::GpuExecutableRunOptions> gpu_run_options);
  ~PjRtStreamExecutorClient() override = default;

  int process_index() const override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_10(mht_10_v, 337, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "process_index");
 return process_index_; }

  int device_count() const override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_11(mht_11_v, 342, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "device_count");
 return devices_.size(); }
  int addressable_device_count() const override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_12(mht_12_v, 346, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "addressable_device_count");

    return addressable_devices_.size();
  }
  absl::Span<PjRtDevice* const> devices() const override { return devices_; }
  absl::Span<PjRtDevice* const> addressable_devices() const override {
    return addressable_devices_;
  }

  StatusOr<PjRtDevice*> LookupDevice(int device_id) const override {
    auto it = id_to_device_.find(device_id);
    if (it != id_to_device_.end()) {
      return it->second;
    }
    return InvalidArgument("No matching device found for device_id %d",
                           device_id);
  }

  StatusOr<PjRtDevice*> LookupAddressableDevice(
      int local_hardware_id) const override;

  PjRtPlatformId platform_id() const override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_13(mht_13_v, 369, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "platform_id");
 return platform_id_; }
  absl::string_view platform_name() const override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_14(mht_14_v, 373, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "platform_name");
 return platform_name_; }
  absl::string_view platform_version() const override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_15(mht_15_v, 377, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "platform_version");
 return "<unknown>"; }
  PjRtRuntimeType runtime_type() const override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_16(mht_16_v, 381, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "runtime_type");
 return kStreamExecutor; }

  // Most platforms expect device-to-device transfers to be enqueued on the
  // source d2d stream, but some platforms use the destination d2d stream. This
  // function specifies which one the platform expects.
  virtual bool EnqueueD2DTransfersOnSrcStream() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_17(mht_17_v, 389, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "EnqueueD2DTransfersOnSrcStream");
 return true; }

  StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;

  StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      const XlaComputation& computation, CompileOptions options) override;
  StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      mlir::ModuleOp mlir_module, CompileOptions options) override;

  StatusOr<absl::optional<std::string>> ExecutableFingerprint(
      const PjRtExecutable& executable) const override {
    return absl::optional<std::string>();
  }

  StatusOr<std::string> SerializeExecutable(
      const PjRtExecutable& executable) const override {
    return Unimplemented("SerializeExecutable not implemented on %s",
                         platform_name());
  }

  StatusOr<std::unique_ptr<PjRtExecutable>> DeserializeExecutable(
      absl::string_view serialized, CompileOptions options) override {
    return Unimplemented("DeserializeExecutable not implemented on %s",
                         platform_name());
  }

  StatusOr<std::unique_ptr<HloCostAnalysis>> GetHloCostAnalysis() override;

  // Creates a buffer on the device without initializing or copying any data.
  // An optional `definition_event` may be speficied that can be used to
  // ensure the buffer isn't referenced until some external mechanism has
  // initialized the data.
  StatusOr<std::unique_ptr<PjRtBuffer>> CreateUninitializedBuffer(
      const Shape& shape, PjRtDevice* device) override;
  StatusOr<std::unique_ptr<PjRtBuffer>> CreateUninitializedBuffer(
      const Shape& shape, PjRtDevice* device,
      std::shared_ptr<BufferSequencingEvent> definition_event);

  StatusOr<std::unique_ptr<PjRtClient::AsyncBufferTransferManager>>
  CreateBuffersForAsyncTransfer(absl::Span<const Shape> shapes,
                                PjRtDevice* device) override {
    return Unimplemented("Async transfer to buffers not implemented");
  };

  StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostBuffer(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      absl::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      std::function<void()> on_done_with_host_buffer,
      PjRtDevice* device) override;

  StatusOr<std::unique_ptr<PjRtBuffer>> BufferFromHostLiteral(
      const LiteralSlice& literal, PjRtDevice* device) override;

  void MakeCrossHostReceiveBuffers(
      absl::Span<const Shape> shapes, PjRtDevice* device,
      PjRtCrossHostRecvNotifier&& notifier) override;

  void MakeCrossHostReceiveBuffersForGather(
      absl::Span<const Shape> shapes, std::vector<GatherDetails> gather_details,
      PjRtDevice* device, PjRtCrossHostRecvNotifier&& notifier) override;

  StatusOr<std::unique_ptr<PjRtBuffer>> CreateViewOfDeviceBuffer(
      void* device_ptr, const Shape& shape, PjRtDevice* device,
      std::function<void()> on_delete_callback) override;

  StatusOr<ChannelHandle> CreateChannelHandle() override {
    return client()->CreateChannelHandle();
  }
  StatusOr<ChannelHandle> CreateDeviceToHostChannelHandle() override {
    return client()->CreateDeviceToHostChannelHandle();
  }
  StatusOr<ChannelHandle> CreateHostToDeviceChannelHandle() override {
    return client()->CreateHostToDeviceChannelHandle();
  }

  // TODO(zhangqiaorjc): Experimental. Will be removed.
  Status Defragment() override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_18(mht_18_v, 470, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "Defragment");

    return Unimplemented("Defragment not implemented");
  }

  LocalDeviceState& device_state(int device_ordinal) const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_19(mht_19_v, 477, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "device_state");

    return *tensorflow::down_cast<PjRtStreamExecutorDevice*>(
                addressable_devices_.at(device_ordinal))
                ->local_device_state();
  }
  LocalClient* client() const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_20(mht_20_v, 485, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "client");
 return client_; }
  se::DeviceMemoryAllocator* allocator() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_21(mht_21_v, 489, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "allocator");
 return allocator_; }
  tensorflow::Allocator* host_memory_allocator() const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_22(mht_22_v, 493, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "host_memory_allocator");

    return host_memory_allocator_.get();
  }
  bool should_stage_host_to_device_transfers() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_23(mht_23_v, 499, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "should_stage_host_to_device_transfers");

    return should_stage_host_to_device_transfers_;
  }

  gpu::GpuExecutableRunOptions* gpu_run_options() const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_24(mht_24_v, 506, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "gpu_run_options");

    return gpu_run_options_.get();
  }

  tensorflow::thread::ThreadPool* thread_pool() {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_25(mht_25_v, 513, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "thread_pool");
 return &thread_pool_; }

 protected:
  friend class PjRtStreamExecutorBuffer;

  virtual void EnqueueCrossHostReceive(
      std::vector<std::unique_ptr<PjRtBuffer>>&& buffers,
      std::shared_ptr<BufferSequencingEvent> definition_event,
      PjRtCrossHostRecvNotifier&& notifier,
      absl::optional<std::vector<GatherDetails>> gather_details) const {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_26(mht_26_v, 525, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "EnqueueCrossHostReceive");

    notifier(Unimplemented("Cross host receives not implemented."));
  }

  virtual void CopyToRemoteDevice(
      PjRtBuffer* buffer, absl::string_view serialized_descriptor,
      PjRtBuffer::RemoteSendCallback on_done) const {
   std::vector<std::string> mht_27_v;
   mht_27_v.push_back("serialized_descriptor: \"" + std::string(serialized_descriptor.data(), serialized_descriptor.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_27(mht_27_v, 535, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "CopyToRemoteDevice");

    on_done(Unimplemented("Cross host sends not implemented."),
            /*sends_were_enqueued=*/false);
  }

  virtual void CopyToRemoteDeviceScattered(
      PjRtBuffer* buffer,
      absl::Span<const std::pair<std::string, PjRtBuffer::RemoteSendCallback>>
          serialized_descriptors_and_callbacks,
      const PjRtBuffer::ScatterDetails& scatter_details) const {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_28(mht_28_v, 547, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "CopyToRemoteDeviceScattered");

    for (const auto& d_and_cb : serialized_descriptors_and_callbacks) {
      d_and_cb.second(
          Unimplemented("Scattered cross host sends not implemented."),
          /*sends_were_enqueued=*/false);
    }
  }

  virtual Status CopyRawSubBufferToHost(PjRtBuffer* buffer, void* dst,
                                        int64_t offset, int64_t transfer_size,
                                        std::function<void(Status)> on_ready) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_29(mht_29_v, 560, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "CopyRawSubBufferToHost");

    return Unimplemented("Raw copies to host not implemented.");
  }

  // Helper function for creating PjRtStreamExecutorExecutables. Modifies
  // `options` in-place.
  struct ExecutableExtras {
    std::shared_ptr<DeviceAssignment> device_assignment;
    std::vector<PjRtExecutable::LogicalDeviceIds>
        addressable_device_logical_ids;
    std::vector<PjRtDevice*> addressable_devices;
  };
  StatusOr<ExecutableExtras> GetExecutableExtras(CompileOptions* options);

  const PjRtPlatformId platform_id_;
  const std::string platform_name_;
  LocalClient* client_;

  // Allocator to be used for staging memory transfers to devices.
  std::unique_ptr<tensorflow::Allocator> host_memory_allocator_;

  // Device memory allocator. If owned, the allocator must outlive the devices,
  // because it is the device destructor that waits for any outstanding work to
  // complete.
  se::DeviceMemoryAllocator* allocator_;
  std::unique_ptr<se::DeviceMemoryAllocator> owned_allocator_;

  // Includes all devices, including non-local devices on multi-host platforms.
  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> owned_devices_;
  // Pointers to `owned_devices_`.
  std::vector<PjRtDevice*> devices_;
  // Maps Device::id() to the corresponding Device. Includes all devices.
  std::map<int, PjRtDevice*> id_to_device_;
  // Local devices indexed by local device ordinal.
  std::vector<PjRtDevice*> addressable_devices_;
  int process_index_;

  // Should we always prefer to stage host-to-device transfers via memory
  // allocated on host_memory_allocator_? True only on GPU, where we prefer to
  // transfer via pinned memory.
  bool should_stage_host_to_device_transfers_;

  std::unique_ptr<gpu::GpuExecutableRunOptions> gpu_run_options_;

  tensorflow::thread::ThreadPool thread_pool_;

  absl::Mutex transpose_mu_;
  TransposePlanCache transpose_cache_ ABSL_GUARDED_BY(transpose_mu_);
};

// Converts a 2D set of Device objects indexed by [replica][partition] into an
// xla::DeviceAssignment.
StatusOr<DeviceAssignment> DevicesToDeviceAssignment(
    absl::Span<const std::vector<PjRtDevice*>> devices);

class PjRtStreamExecutorBuffer : public PjRtBuffer {
 public:
  // Helper class to retain a "hold" on a PjRtStreamExecutorBuffer. A ScopedHold
  // may not outlive its parent PjRtStreamExecutorBuffer.
  //
  // There are three types of hold, as follows:
  //
  // 1) Usage hold: a transient hold while an operation using the buffer is
  //    being enqueued onto a stream.
  // A client acquires a usage hold by calling
  // PjRtStreamExecutorBuffer::GetBufferWithHold(kUsage) or the convenience
  // wrapper GetBufferWithUsageHold(). If the enqueue completes successfully the
  // hold should be released using a call to ConvertUsageHold. If the ScopedHold
  // is deleted without ConvertUsageHold being called, e.g., on error, the hold
  // is dropped. It is legal to drop a usage hold instead of calling
  // ConvertUsageHold, even if the buffer was successfully enqueued, as long as
  // the client ensures that all necessary synchronization has been done.
  //
  // 2) External hold: a potentially long-lived hold while the buffer is being
  //    shared by an external framework, e.g., NumPy.
  // A client acquires an external hold by calling
  // PjRtStreamExecutorBuffer::GetBufferWithHold(kExternal) or the convenience
  // wrapper GetBufferWithExternalReference and releases it by deleting the
  // ScopedHold. The external framework should not modify the underlying buffer
  // unless it is confident via its own synchronization that modifications do
  // not race with reads from the PjRtStreamExecutorBuffer.
  //
  // 3) Donation hold: a transient hold while an execution that donates the
  //    buffer is being enqueued onto the compute stream.
  // A client acquires a donation hold by calling
  // PjRtStreamExecutorBuffer::GetBufferWithHold(kDonation). If the enqueue
  // completes successfully the hold should be released using a call to
  // ConfirmDonation after which the buffer is invalid. If the ScopedHold is
  // deleted without ConfirmDonation being called, e.g., on error, the hold is
  // dropped and the buffer remains valid. If the buffer is successfully
  // enqueued the client *must* call ConfirmDonation.
  //
  // Donation holds behave like exclusive write locks: when a donation hold
  // has been acquired, any attempt to acquire another hold of any type will
  // block until the donation hold is dropped or confirmed. Acquiring a donation
  // hold will fail with an error if there is any outstanding external hold, and
  // will block if there are any outstanding usage holds until those holds are
  // dropped or converted.
  //
  // Calls to PjRtStreamExecutorBuffer::Release (and transitively to
  // PjRtStreamExecutorBuffer::Delete() and ~PjRtStreamExecutorBuffer()) will
  // block until all usage and donation holds are either deleted or
  // converted/confirmed.
  class ScopedHold {
   public:
    enum Type { kUsage = 0, kExternalReference, kDonation, kMaxValue };
    // Use a State enum instead of encoding the state in an error Status to
    // avoid creating Status values in non-error cases. Creating a Status
    // entails several allocations and can add O(us) to every use of a hold.
    enum State {
      kUninitialized = 0,
      kValid,
      kMoved,
      kConverted,
      kReleased,
      kDonated,
      kError
    };

    ~ScopedHold();
    ScopedHold(ScopedHold&& other);
    ScopedHold(const ScopedHold&) = delete;
    ScopedHold& operator=(const ScopedHold&) = delete;

    Type type() const {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_30(mht_30_v, 687, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "type");
 return type_; }

    Status status() const {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_31(mht_31_v, 692, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "status");

      // Lazily create Status values only when they are requested.
      switch (state_) {
        case kUninitialized:
          return InvalidArgument("Buffer has not been initialized");
        case kValid:
          return Status::OK();
        case kMoved:
          return InvalidArgument("Buffer has been moved.");
        case kConverted:
          return InvalidArgument("Buffer has been converted");
        case kReleased:
          return InvalidArgument("Buffer has been released");
        case kDonated:
          return InvalidArgument("Buffer has been donated");
        case kError:
          return status_;
        default:
          CHECK(false) << "Unexpected state value " << state_;
      }
    }
    bool ok() const {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_32(mht_32_v, 716, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "ok");
 return state_ == kValid; }

    // Access to the underlying device buffer storage. Requires this->ok().
    const std::shared_ptr<TrackedDeviceBuffer>& buffer() const {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_33(mht_33_v, 722, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "buffer");

      CHECK_EQ(state_, kValid);
      CHECK_NE(buffer_, nullptr);
      return buffer_;
    }
    TrackedDeviceBuffer* operator->() const { return buffer().get(); }
    const TrackedDeviceBuffer& operator*() const {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_34(mht_34_v, 731, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "*");
 return *buffer(); }

    // Converts the hold into a usage event. Only valid for holds of type
    // kUsage.
    //
    //   usage_stream:   the stream that the buffer was used on.
    //   event:          an event that has been recorded on usage_stream after
    //                   the buffer was used.
    //   reference_held: true if and only if the caller has caused a
    //                   reference to this->buffer() to stay live until after
    //                   the host is sure that the usage (transfer or execution)
    //                   has completed.
    void ConvertUsageHold(se::Stream* usage_stream,
                          std::shared_ptr<BufferSequencingEvent> event,
                          bool reference_held);

    // Confirms that the buffer was successfully donated to an execution.
    // Only valid for holds of type kDonation. Causes the buffer to become
    // invalid.
    void ConfirmDonation();

    // Adds the held device buffers in order to 'iterator'. Used to add the
    // buffers to an ExecutionInput. We require but do not verify that
    // 'iterator' when passed in is pointing to a sub-tuple of the
    // ExecutionInput whose on_device_shape matches that of the
    // TrackedDeviceBuffer. 'end' is used to check that 'iterator' doesn't run
    // out of bounds. Donates the device buffers if the hold type is kDonation,
    // otherwise retains ownership of the device buffers.
    void AddToInput(ShapeTree<MaybeOwningDeviceMemory>::iterator* iterator,
                    const ShapeTree<MaybeOwningDeviceMemory>::iterator& end,
                    ExecutionInput* execution_input,
                    se::DeviceMemoryAllocator* allocator) const;

   private:
    friend class PjRtStreamExecutorBuffer;
    friend class PjRtStreamExecutorClient;

    // Helper struct that makes it possible to move a ScopedHold through a
    // closure.
    using ForClosure = std::tuple<PjRtStreamExecutorBuffer*, Type, State,
                                  Status, std::shared_ptr<TrackedDeviceBuffer>>;

    ScopedHold(PjRtStreamExecutorBuffer* parent, Type type)
        : parent_(parent), type_(type), state_(kUninitialized) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_35(mht_35_v, 777, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "ScopedHold");
}
    explicit ScopedHold(const ForClosure& closure_helper)
        : parent_(std::get<0>(closure_helper)),
          type_(std::get<1>(closure_helper)),
          state_(std::get<2>(closure_helper)),
          status_(std::get<3>(closure_helper)),
          buffer_(std::get<4>(closure_helper)) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_36(mht_36_v, 786, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "ScopedHold");

      // Check the buffer is not in an error state.
      CHECK(status_.ok() && buffer_ != nullptr);
    }

    // Sets buffer state.
    void SetState(State state) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_37(mht_37_v, 795, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "SetState");
 state_ = state; }

    // Sets buffer_ and status_. Called by parent_ to initialize the hold.
    void Acquire(StatusOr<std::shared_ptr<TrackedDeviceBuffer>>&& buffer_or);
    // Releases the contents of *this, so *this can subsequently be
    // deleted without releasing the parent's hold. Should be passed to the
    // appropriate constructor of another ScopedHold, e.g., when a hold must be
    // passed through a closure that is incompatible with std::move.
    ForClosure ToClosure();

    PjRtStreamExecutorBuffer* const parent_;
    const Type type_;

    // There is an invariant that if ok() then
    // buffer_.ValueOrDie() != nullptr.
    State state_;
    Status status_;
    std::shared_ptr<TrackedDeviceBuffer> buffer_;
  };

  PjRtStreamExecutorBuffer(Shape on_device_shape,
                           std::shared_ptr<TrackedDeviceBuffer> device_buffer,
                           PjRtClient* client, PjRtDevice* device);
  ~PjRtStreamExecutorBuffer() override;

  PjRtStreamExecutorBuffer(const PjRtStreamExecutorBuffer&) = delete;
  PjRtStreamExecutorBuffer(PjRtStreamExecutorBuffer&&) = delete;
  PjRtStreamExecutorBuffer& operator=(const PjRtStreamExecutorBuffer&) = delete;
  PjRtStreamExecutorBuffer& operator=(PjRtStreamExecutorBuffer&&) = delete;

  const Shape& on_device_shape() const override {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_38(mht_38_v, 828, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "on_device_shape");
 return on_device_shape_; }
  StatusOr<Shape> logical_on_device_shape() override;
  PjRtStreamExecutorDevice* device() const override {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_39(mht_39_v, 833, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "device");
 return device_; }
  PjRtPlatformId platform_id() const {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_40(mht_40_v, 837, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "platform_id");
 return client_->platform_id(); }
  absl::string_view platform_name() const {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_41(mht_41_v, 841, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "platform_name");
 return client_->platform_name(); }
  PjRtStreamExecutorClient* client() const override {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_42(mht_42_v, 845, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "client");
 return client_; }
  bool IsEmptyTuple() const {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_43(mht_43_v, 849, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "IsEmptyTuple");

    return on_device_shape_.IsTuple() &&
           on_device_shape_.tuple_shapes_size() == 0;
  }

  StatusOr<std::unique_ptr<ExternalReference>> AcquireExternalReference()
      override;

  StatusOr<std::unique_ptr<ExternalReference>> ReleaseDeviceMemoryOwnership(
      bool wait_for_operations_to_complete) override;

  using PjRtBuffer::ToLiteralSync;
  PjRtFuture<Status> ToLiteral(MutableLiteralBase* literal) override;

  StatusOr<size_t> GetOnDeviceSizeInBytes() const override;

  Status CopyRawToHost(void* dst, int64_t offset, int64_t transfer_size,
                       std::function<void(Status)> on_ready) override;

  // Drops the buffer's reference to its associated device memory, leaving the
  // buffer in an invalid state. The memory will be freed lazily when all async
  // operations using the buffer have completed, according to the allocation
  // semantics of the underlying platform. Delete may briefly block if another
  // thread is in the process of enqueuing an operation on this buffer, but it
  // will never block for a stream operation to complete. If an external
  // framework holds a reference to the TrackedDeviceBuffer via
  // GetBufferWithExternalReference, the memory will not be freed until the
  // external framework drops the reference.
  void Delete() override;

  bool IsDeleted() override;

  // Returns a view of the PjRtBuffer device memory as a ShapedBuffer. The
  // PjRtBuffer retains ownership of the device buffers.
  StatusOr<ShapedBuffer> AsShapedBuffer() const;

  // Returns a hold on the TrackedDeviceBuffer holding the device
  // buffers. See comment on ScopedHold.
  ScopedHold GetBufferWithHold(ScopedHold::Type type);
  ScopedHold GetBufferWithUsageHold() {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_44(mht_44_v, 891, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "GetBufferWithUsageHold");

    return GetBufferWithHold(ScopedHold::kUsage);
  }
  ScopedHold GetBufferWithExternalReference() {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_45(mht_45_v, 897, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "GetBufferWithExternalReference");

    return GetBufferWithHold(ScopedHold::kExternalReference);
  }

  StatusOr<std::unique_ptr<PjRtBuffer>> CopyToDevice(
      PjRtDevice* dst_device) override;

  void CopyToRemoteDevice(absl::string_view serialized_descriptor,
                          RemoteSendCallback on_done) override;

  void CopyToRemoteDeviceScattered(
      absl::Span<const std::pair<std::string, RemoteSendCallback>>
          serialized_descriptors_and_callbacks,
      const ScatterDetails& scatter_details) override;

  PjRtFuture<Status> GetReadyFuture() override;

  bool IsOnCpu() const override;

  // Similar to Delete, drops the buffer's reference to its associated device
  // memory, leaving the buffer in an invalid state, but returns the
  // TrackedDeviceBuffer rather than freeing the device memory, so that another
  // framework can take ownership of it. The buffer returned from Release may
  // be safely dropped at any time even if it still has pending async
  // operations. The client should call GetReadyFuture()->Await() before calling
  // Release with wait_for_operations_to_complete=false, to ensure that the host
  // has synchronized past any outstanding write operations to the buffer. If
  // wait_for_operations_to_complete=true the host will block until any
  // potentially outstanding asynchronous operations have completed before
  // returning, in which case it is safe to read or mutate the returned buffer.
  // If the buffer was shared via an external reference it is the client's
  // responsibility that accesses via that reference do not interfere with
  // accesses via the buffer returned from Release.
  StatusOr<std::shared_ptr<TrackedDeviceBuffer>> Release(
      bool wait_for_operations_to_complete);

 private:
  friend class PjRtClient;

  // Blocks in mu_.Await until there are no more usage holds.
  void WaitForOutstandingUsageHolds() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Blocks in mu_.Await until there is no donation hold.
  void WaitForOutstandingDonationHold() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Adds a hold of 'type' and returns device_buffer_. Returns an error if
  // device_buffer_ is null, or if a donation hold was requested when there is
  // an outstanding external hold.
  // Requires holds_[kDonation] == 0 (i.e., WaitForOutstandingDonationHolds()
  // must be called first.)
  StatusOr<std::shared_ptr<TrackedDeviceBuffer>> GetBufferForHoldLocked(
      ScopedHold::Type type) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Adds a hold of hold->type() and initializes `hold` with device_buffer_.
  // Initializes hold with an error if device_buffer_ is null, or if a donation
  // hold was requested when there is an outstanding external hold.
  // Requires holds_[kDonation] == 0 (i.e., WaitForOutstandingDonationHolds()
  // must be called first.)
  void AcquireHoldLocked(ScopedHold* hold) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Drops a usage hold and calls device_buffer_->AddUsageEvent. Does a sanity
  // check that buffer==device_buffer_ or device_buffer_==nullptr. Called after
  // device_buffer_ was successfully enqueued on a stream.
  void ConvertUsageHold(TrackedDeviceBuffer* buffer, se::Stream* usage_stream,
                        std::shared_ptr<BufferSequencingEvent> event,
                        bool reference_held);

  // Drops a donation hold and makes *this invalid for further use. Does a
  // sanity check that buffer==device_buffer_. Called after device_buffer_ was
  // successfully donated to an execution.
  void ConfirmDonation(TrackedDeviceBuffer* device_buffer);

  // Drops a hold without taking any other action. Does a sanity check that
  // buffer==device_buffer_ or device_buffer_==nullptr.
  void DropHold(ScopedHold::Type type, TrackedDeviceBuffer* buffer);

  StatusOr<std::pair<std::unique_ptr<PjRtBuffer>,
                     std::shared_ptr<BufferSequencingEvent>>>
  CopyToDeviceHelper(PjRtDevice* dst_device, LocalDeviceState* dst_local_device,
                     LocalDeviceState* transfer_local_device,
                     se::Stream* transfer_stream,
                     std::shared_ptr<TrackedDeviceBuffer> src_device_buffer);

  PjRtStreamExecutorClient* const client_;
  const Shape on_device_shape_;
  PjRtStreamExecutorDevice* const device_;

  mutable absl::Mutex mu_;
  std::shared_ptr<TrackedDeviceBuffer> device_buffer_ ABSL_GUARDED_BY(mu_);
  // Count of holds on the buffer.
  std::array<int, ScopedHold::Type::kMaxValue> holds_ ABSL_GUARDED_BY(mu_);
  PjRtFuture<Status>::Promise definition_promise_ ABSL_GUARDED_BY(mu_);
};

// Wraps one or more XLA LocalExecutables (one per partition, as specified by
// the build options).
class PjRtStreamExecutorExecutable : public PjRtExecutable {
 public:
  PjRtStreamExecutorExecutable(
      std::vector<std::unique_ptr<LocalExecutable>> executables,
      bool parameter_is_tupled_arguments,
      std::shared_ptr<DeviceAssignment> device_assignment,
      std::vector<LogicalDeviceIds> addressable_device_logical_ids,
      std::vector<PjRtDevice*> addressable_devices,
      PjRtStreamExecutorClient* client);

  ~PjRtStreamExecutorExecutable() override = default;

  PjRtStreamExecutorClient* client() const override {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_46(mht_46_v, 1008, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "client");
 return client_; }

  absl::string_view name() const override;

  int num_replicas() const override {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_47(mht_47_v, 1015, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "num_replicas");

    return executables_[0]->build_options().num_replicas();
  }

  int num_partitions() const override {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_48(mht_48_v, 1022, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "num_partitions");

    return executables_[0]->build_options().num_partitions();
  }

  int64_t SizeOfGeneratedCodeInBytes() const override {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_49(mht_49_v, 1029, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "SizeOfGeneratedCodeInBytes");

    int64_t size = 0;
    for (auto& executable : executables_) {
      size += executable->executable()->SizeOfGeneratedCodeInBytes();
    }
    return size;
  }

  const DeviceAssignment& device_assignment() const override {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_50(mht_50_v, 1040, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "device_assignment");

    return *device_assignment_;
  }

  absl::Span<const LogicalDeviceIds> addressable_device_logical_ids()
      const override {
    return addressable_device_logical_ids_;
  }

  absl::Span<PjRtDevice* const> addressable_devices() const override {
    return addressable_devices_;
  }

  // Return an HloModule per partition.
  StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override;

  using PjRtExecutable::Execute;
  StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>> Execute(
      absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
      const ExecuteOptions& options,
      absl::optional<std::vector<PjRtFuture<Status>>>& returned_futures)
      override;

  using PjRtExecutable::ExecuteSharded;
  StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecuteSharded(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      absl::optional<PjRtFuture<Status>>& returned_future,
      bool fill_future) override;

  using PjRtExecutable::ExecutePortable;
  StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>> ExecutePortable(
      absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
      const ExecuteOptions& options,
      absl::optional<PjRtFuture<Status>>& returned_future,
      bool fill_future) override;

  void Delete() override {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_51(mht_51_v, 1081, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "Delete");
 executables_.clear(); }

  bool IsDeleted() override {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_52(mht_52_v, 1086, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "IsDeleted");
 return executables_.empty(); }

  absl::Span<const std::shared_ptr<LocalExecutable>> executables() const {
    return executables_;
  }

 protected:
  bool parameter_is_tupled_arguments() const {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPSpjrt_stream_executor_clientDTh mht_53(mht_53_v, 1096, "", "./tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h", "parameter_is_tupled_arguments");

    return parameter_is_tupled_arguments_;
  }

 private:
  friend class PjRtStreamExecutorClient;
  friend class PjRtTpuClient;
  friend class InternalPjRtTpuClient;
  // Initializes information about which arguments to which executables must be
  // donated due to aliases that were specified by the computation.
  Status SetUpDonation(bool tuple_inputs);

  // Returns a sorted list of the parameters that must be donated. Derived
  // classes may use custom logic.
  virtual absl::Span<int const> ParametersThatMustBeDonated(
      int executable_idx) const;

  virtual StatusOr<std::vector<ExecutionInput>>
  MakeExecutionInputsAndWaitForEvents(
      int device_ordinal, const ExecuteOptions& options,
      absl::Span<const Shape> executable_parameter_shapes,
      absl::Span<PjRtBuffer* const> argument_handles,
      absl::Span<const PjRtStreamExecutorBuffer::ScopedHold> device_buffers,
      absl::flat_hash_set<BufferSequencingEvent*>& events) const;

  StatusOr<ScopedShapedBuffer> EnqueueExecution(
      absl::Span<PjRtBuffer* const> argument_handles, int replica,
      int partition, int executable_idx, const RunId& run_id,
      const ExecuteOptions& options, PjRtDevice* device,
      std::vector<PjRtStreamExecutorBuffer::ScopedHold>* device_buffers,
      std::shared_ptr<DeviceAssignment> device_assignment,
      std::vector<std::function<void()>>& compute_callbacks) const;

  virtual std::vector<std::unique_ptr<PjRtBuffer>> MakeOutputBuffers(
      int device_ordinal, const ExecuteOptions& options,
      ScopedShapedBuffer result_buffer,
      std::shared_ptr<BufferSequencingEvent> definition_event,
      PjRtDevice* device, std::vector<std::function<void()>>& compute_callbacks,
      std::vector<std::shared_ptr<TrackedDeviceBuffer>>& buffers_to_release)
      const;

  StatusOr<Result> ExecuteHelper(absl::Span<PjRtBuffer* const> argument_handles,
                                 int replica, int partition,
                                 const RunId& run_id,
                                 const ExecuteOptions& options,
                                 bool fill_future,
                                 PjRtDevice* device = nullptr) const;

  // Create shared pointers so we can free them after the execution: with
  // asynchronous execution, the process being executed can outlive the
  // executable itself.
  PjRtStreamExecutorClient* const client_;
  // One executable per partition.
  std::vector<std::shared_ptr<LocalExecutable>> executables_;
  // On device shapes of the executable parameters.
  std::vector<std::vector<Shape>> on_device_executable_parameter_shapes_;
  // Per-executable sorted vector of parameters that have any aliased buffers
  // and thus must be donated when executing the computation.
  std::vector<std::vector<int>> parameters_that_must_be_donated_;
  std::shared_ptr<DeviceAssignment> device_assignment_;

  // True if the executables were compiled expecting arguments in a single
  // tuple.
  const bool parameter_is_tupled_arguments_;

  // The replica and partition indices of device_assignment_ to be run by this
  // client. On single-host platforms without partitioning, this is all replicas
  // (i.e. addressable_device_logical_ids_[i] = (i, 0)), but this may not be the
  // case on multi-host platforms. If there are 4 replicas and 2 partitions on a
  // single host platform, size of addressable_device_logical_ids_ is 4*2 = 8.
  std::vector<LogicalDeviceIds> addressable_device_logical_ids_;

  // addressable_devices_[i] is the Device to which
  // addressable_device_logical_ids_[i] is assigned. shared_ptrs instead of
  // unique_ptrs to play well with the Python bindings (see xla.cc).
  std::vector<PjRtDevice*> addressable_devices_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_PJRT_STREAM_EXECUTOR_CLIENT_H_
