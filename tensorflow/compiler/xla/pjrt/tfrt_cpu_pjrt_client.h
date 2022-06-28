/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_TFRT_CPU_PJRT_CLIENT_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_TFRT_CPU_PJRT_CLIENT_H_
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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh() {
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

#include "absl/base/thread_annotations.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/client/executable_build_options.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/layout.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_future.h"
#include "tensorflow/compiler/xla/pjrt/semaphore.h"
#include "tensorflow/compiler/xla/pjrt/tracked_tfrt_cpu_device_buffer.h"
#include "tensorflow/compiler/xla/pjrt/transpose.h"
#include "tensorflow/compiler/xla/pjrt/worker_thread.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/computation_placer.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_compiler.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_executable.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_module_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime

namespace xla {

class TfrtCpuDevice final : public PjRtDevice {
 public:
  TfrtCpuDevice(int id, bool asynchronous);

  void SetClient(PjRtClient* client) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_0(mht_0_v, 224, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "SetClient");

    CHECK(client_ == nullptr);
    client_ = client;
  }

  PjRtClient* client() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_1(mht_1_v, 232, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "client");
 return client_; }

  bool IsAddressable() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_2(mht_2_v, 237, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "IsAddressable");

    return process_index() == client()->process_index();
  }

  int id() const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_3(mht_3_v, 244, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "id");
 return id_; }

  int process_index() const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_4(mht_4_v, 249, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "process_index");
 return 0; }

  // Used as `device_ordinal`.
  int local_hardware_id() const override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_5(mht_5_v, 255, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "local_hardware_id");
 return id_; }

  absl::string_view device_kind() const override;

  std::string DebugString() const override;

  Status TransferToInfeed(const LiteralSlice& literal) override;

  Status TransferFromOutfeed(MutableBorrowingLiteral literal) override;

  // Returns a semaphore for admission control on inflight computations.
  Semaphore& max_inflight_computations_semaphore() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_6(mht_6_v, 269, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "max_inflight_computations_semaphore");

    return max_inflight_computations_semaphore_;
  }

 private:
  int id_;
  PjRtClient* client_ = nullptr;

  // TODO(zhangqiaorjc): Optimize semaphore related overhead.
  // Semaphore used to limit how many programs can be enqueued by the host
  // ahead of the device.
  Semaphore max_inflight_computations_semaphore_;
};

class TfrtCpuExecutable;

class TfrtCpuClient final : public PjRtClient {
 public:
  TfrtCpuClient(int process_index,
                std::vector<std::unique_ptr<TfrtCpuDevice>> devices,
                std::unique_ptr<tfrt::HostContext> host_ctx);

  int process_index() const override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_7(mht_7_v, 294, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "process_index");
 return process_index_; }

  int device_count() const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_8(mht_8_v, 299, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "device_count");
 return devices_.size(); }

  int addressable_device_count() const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_9(mht_9_v, 304, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "addressable_device_count");

    return addressable_devices_.size();
  }

  absl::Span<PjRtDevice* const> devices() const override { return devices_; }

  absl::Span<PjRtDevice* const> addressable_devices() const override {
    return addressable_devices_;
  }

  StatusOr<PjRtDevice*> LookupDevice(int device_id) const override;

  StatusOr<PjRtDevice*> LookupAddressableDevice(
      int local_hardware_id) const override;

  PjRtPlatformId platform_id() const override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_10(mht_10_v, 322, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "platform_id");

    return tensorflow::Fingerprint64(CpuName());
  }

  absl::string_view platform_name() const override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_11(mht_11_v, 329, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "platform_name");
 return CpuName(); }

  absl::string_view platform_version() const override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_12(mht_12_v, 334, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "platform_version");
 return "<unknown>"; }

  PjRtRuntimeType runtime_type() const override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_13(mht_13_v, 339, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "runtime_type");
 return kTfrt; }

  StatusOr<DeviceAssignment> GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const override;

  StatusOr<std::unique_ptr<HloCostAnalysis>> GetHloCostAnalysis() override;

  StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      const XlaComputation& computation, CompileOptions options) override;
  StatusOr<std::unique_ptr<PjRtExecutable>> Compile(
      mlir::ModuleOp module, CompileOptions options) override;

  StatusOr<absl::optional<std::string>> ExecutableFingerprint(
      const PjRtExecutable& executable) const override;

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

  StatusOr<std::unique_ptr<PjRtBuffer>> CreateUninitializedBuffer(
      const Shape& shape, PjRtDevice* device) override;

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
      PjRtCrossHostRecvNotifier&& notifier) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_14(mht_14_v, 390, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "MakeCrossHostReceiveBuffers");

    LOG(FATAL) << "MakeCrossHostReceiveBuffers not implemented.";
  }

  void MakeCrossHostReceiveBuffersForGather(
      absl::Span<const Shape> shapes, std::vector<GatherDetails> gather_details,
      PjRtDevice* device, PjRtCrossHostRecvNotifier&& notifier) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_15(mht_15_v, 399, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "MakeCrossHostReceiveBuffersForGather");

    LOG(FATAL) << "MakeCrossHostReceiveBuffersForGather not implemented.";
  }

  StatusOr<std::unique_ptr<PjRtBuffer>> CreateViewOfDeviceBuffer(
      void* device_ptr, const Shape& shape, PjRtDevice* device,
      std::function<void()> on_delete_callback) override;

  StatusOr<ChannelHandle> CreateChannelHandle() override {
    return Unimplemented("CreateChannelHandle not implemented.");
  }
  StatusOr<ChannelHandle> CreateDeviceToHostChannelHandle() override {
    return Unimplemented("CreateDeviceToHostChannelHandle not implemented.");
  }
  StatusOr<ChannelHandle> CreateHostToDeviceChannelHandle() override {
    return Unimplemented("CreateHostToDeviceChannelHandle not implemented.");
  }

  Status Defragment() override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_16(mht_16_v, 420, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "Defragment");

    return Unimplemented("Defragment not implemented.");
  }

  tfrt::HostContext* GetHostContext() const {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_17(mht_17_v, 427, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "GetHostContext");
 return host_ctx_.get(); }

  Eigen::ThreadPoolDevice* eigen_intraop_device() const {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_18(mht_18_v, 432, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "eigen_intraop_device");

    return eigen_intraop_device_.get();
  }

  tfrt::AsyncValueRef<CpuEvent> GetLastCollectiveLaunchEvent() {
    absl::MutexLock lock(&mu_);
    return last_collective_launch_event_.CopyRef();
  }

  void SetLastCollectiveLaunchEvent(tfrt::AsyncValueRef<CpuEvent> event) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_19(mht_19_v, 444, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "SetLastCollectiveLaunchEvent");

    absl::MutexLock lock(&mu_);
    last_collective_launch_event_ = std::move(event);
  }

 private:
  int process_index_;
  // Includes all devices, including non-addressable devices.
  std::vector<std::unique_ptr<TfrtCpuDevice>> owned_devices_;
  // Pointers to `owned_devices_`.
  std::vector<PjRtDevice*> devices_;
  // Maps Device::id() to the corresponding Device. Includes all devices.
  absl::flat_hash_map<int, TfrtCpuDevice*> id_to_device_;
  // Addressable devices indexed by core_id.
  std::vector<PjRtDevice*> addressable_devices_;
  std::unique_ptr<tfrt::HostContext> host_ctx_;
  std::unique_ptr<ComputationPlacer> computation_placer_;

  // TODO(zhangqiaorjc): Use tfrt::compat::EigenHostContextThreadPool.
  std::unique_ptr<tensorflow::thread::ThreadPool> eigen_intraop_pool_;
  std::unique_ptr<Eigen::ThreadPoolDevice> eigen_intraop_device_;

  // Launching collectives are prone to deadlock when we use fixed-sized
  // threadpools since ExecuteHelper will block until all replicas reach the
  // barrier. We ensure that
  // 1. Threadpool size is at least as large as device_count so one collective
  //    launch over all devices can succeed.
  // 2. Gang-schedule each collective by conservatively ensuring a total order
  //    of collectives and launching only one collective at a time to avoid
  //    having no active threads to make progress
  // TODO(zhangqiaorjc): Explore alternatives that allow multiple concurrent
  // collectives.
  mutable absl::Mutex mu_;
  tfrt::AsyncValueRef<CpuEvent> last_collective_launch_event_
      ABSL_GUARDED_BY(mu_);

  // A cache for transpose plans. We use transposes to convert
  // (possibly strided) buffers provided to BufferFromHostBuffer into dense
  // major-to-minor layout.
  absl::Mutex transpose_mu_;
  TransposePlanCache transpose_cache_ ABSL_GUARDED_BY(transpose_mu_);
};

class TfrtCpuBuffer final : public PjRtBuffer {
 public:
  // Helper class to retain a "hold" on a TfrtCpuBuffer. A ScopedHold may not
  // outlive its parent TfrtCpuBuffer.
  //
  // There are three types of hold, as follows:
  //
  // 1) Usage hold: a transient hold while an operation using the buffer is
  //    being enqueued to the runtime.
  // A client acquires a usage hold by calling
  // TfrtCpuBuffer::GetBufferWithHold(kUsage) or the convenience
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
  // TfrtCpuBuffer::GetBufferWithHold(kExternal) or the convenience
  // wrapper GetBufferWithExternalReference and releases it by deleting the
  // ScopedHold. The external framework should not modify the underlying buffer
  // unless it is confident via its own synchronization that modifications do
  // not race with reads from the TfrtCpuBuffer.
  //
  // 3) Donation hold: a transient hold while an execution that donates the
  //    buffer is being enqueued to the runtime.
  // A client acquires a donation hold by calling
  // TfrtCpuBuffer::GetBufferWithHold(kDonation). If the enqueue
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
  // Calls to TfrtCpuBuffer::ReleaseDeviceMemoryOwnership (and transitively to
  // TfrtCpuBuffer::Delete() and ~TfrtCpuBuffer()) will block until all usage
  // and donation holds are either deleted or converted/confirmed.
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
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_20(mht_20_v, 559, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "type");
 return type_; }

    Status status() const {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_21(mht_21_v, 564, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "status");

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
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_22(mht_22_v, 588, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "ok");
 return state_ == kValid; }

    // Access to the underlying device buffer storage. Requires this->ok().
    const std::shared_ptr<TrackedTfrtCpuDeviceBuffer>& buffer() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_23(mht_23_v, 594, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "buffer");

      CHECK_EQ(state_, kValid);
      CHECK_NE(buffer_, nullptr);
      return buffer_;
    }
    TrackedTfrtCpuDeviceBuffer* operator->() const { return buffer().get(); }
    const TrackedTfrtCpuDeviceBuffer& operator*() const {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_24(mht_24_v, 603, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "*");
 return *buffer(); }

    // Converts the hold into a usage event. Only valid for holds of type
    // kUsage.
    void ConvertUsageHold(absl::Span<tfrt::AsyncValueRef<CpuEvent>> events);

    // Confirms that the buffer was successfully donated to an execution.
    // Only valid for holds of type kDonation. Causes the buffer to become
    // invalid.
    void ConfirmDonation();

   private:
    friend class TfrtCpuClient;
    friend class TfrtCpuBuffer;

    // Helper struct that makes it possible to move a ScopedHold through a
    // closure.
    using ForClosure = std::tuple<TfrtCpuBuffer*, Type, State, Status,
                                  std::shared_ptr<TrackedTfrtCpuDeviceBuffer>>;

    ScopedHold(TfrtCpuBuffer* parent, Type type)
        : parent_(parent), type_(type), state_(kUninitialized) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_25(mht_25_v, 627, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "ScopedHold");
}
    explicit ScopedHold(const ForClosure& closure_helper)
        : parent_(std::get<0>(closure_helper)),
          type_(std::get<1>(closure_helper)),
          state_(std::get<2>(closure_helper)),
          status_(std::get<3>(closure_helper)),
          buffer_(std::get<4>(closure_helper)) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_26(mht_26_v, 636, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "ScopedHold");

      // Check the buffer is not in an error state.
      CHECK(status_.ok() && buffer_ != nullptr);
    }

    // Sets buffer state.
    void SetState(State state) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_27(mht_27_v, 645, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "SetState");
 state_ = state; }

    // Sets buffer_ and status_. Called by parent_ to initialize the hold.
    void Acquire(
        StatusOr<std::shared_ptr<TrackedTfrtCpuDeviceBuffer>>&& buffer_or);
    // Releases the contents of *this, so *this can subsequently be
    // deleted without releasing the parent's hold. Should be passed to the
    // appropriate constructor of another ScopedHold, e.g., when a hold must be
    // passed through a closure that is incompatible with std::move.
    ForClosure ToClosure();

    TfrtCpuBuffer* const parent_;
    const Type type_;

    // There is an invariant that if ok() then buffer_ != nullptr.
    State state_;
    Status status_;
    std::shared_ptr<TrackedTfrtCpuDeviceBuffer> buffer_;
  };

  TfrtCpuBuffer(
      Shape on_device_shape,
      std::shared_ptr<TrackedTfrtCpuDeviceBuffer> tracked_device_buffer,
      TfrtCpuClient* client, TfrtCpuDevice* device);
  ~TfrtCpuBuffer() override;

  TfrtCpuBuffer(const TfrtCpuBuffer&) = delete;
  TfrtCpuBuffer(TfrtCpuBuffer&&) = delete;
  TfrtCpuBuffer& operator=(const TfrtCpuBuffer&) = delete;
  TfrtCpuBuffer& operator=(TfrtCpuBuffer&&) = delete;

  const Shape& on_device_shape() const override {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_28(mht_28_v, 679, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "on_device_shape");
 return on_device_shape_; }
  TfrtCpuDevice* device() const override {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_29(mht_29_v, 683, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "device");
 return device_; }
  TfrtCpuClient* client() const override {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_30(mht_30_v, 687, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "client");
 return client_; }

  StatusOr<Shape> logical_on_device_shape() override;

  StatusOr<std::unique_ptr<ExternalReference>> AcquireExternalReference()
      override;

  StatusOr<std::unique_ptr<ExternalReference>> ReleaseDeviceMemoryOwnership(
      bool wait_for_operations_to_complete) override;

  using PjRtBuffer::ToLiteralSync;
  PjRtFuture<Status> ToLiteral(MutableLiteralBase* literal) override;

  StatusOr<size_t> GetOnDeviceSizeInBytes() const override;

  Status CopyRawToHost(void* dst, int64_t offset, int64_t transfer_size,
                       std::function<void(Status)> on_ready) override {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_31(mht_31_v, 706, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "CopyRawToHost");

    return Unimplemented("CopyRawToHost not implemented");
  }

  void Delete() override;

  bool IsDeleted() override;

  StatusOr<std::unique_ptr<PjRtBuffer>> CopyToDevice(
      PjRtDevice* dst_device) override;

  void CopyToRemoteDevice(absl::string_view serialized_descriptor,
                          RemoteSendCallback on_done) override {
   std::vector<std::string> mht_32_v;
   mht_32_v.push_back("serialized_descriptor: \"" + std::string(serialized_descriptor.data(), serialized_descriptor.size()) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_32(mht_32_v, 722, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "CopyToRemoteDevice");

    on_done(Unimplemented("CopyToRemoteDevice not implemented."),
            /*sends_were_enqueued=*/false);
  }

  void CopyToRemoteDeviceScattered(
      absl::Span<const std::pair<std::string, RemoteSendCallback>>
          serialized_descriptors_and_callbacks,
      const ScatterDetails& scatter_details) override {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_33(mht_33_v, 733, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "CopyToRemoteDeviceScattered");

    for (const auto& d_and_cb : serialized_descriptors_and_callbacks) {
      d_and_cb.second(
          Unimplemented("CopyToRemoteDeviceScattered not implemented."),
          /*sends_were_enqueued=*/false);
    }
  }

  PjRtFuture<Status> GetReadyFuture() override;

  bool IsOnCpu() const override {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_34(mht_34_v, 746, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "IsOnCpu");
 return true; }

  // Returns a hold on the TrackedTfrtCpuDeviceBuffer holding the device
  // buffers. See comment on ScopedHold.
  ScopedHold GetBufferWithHold(ScopedHold::Type type);
  ScopedHold GetBufferWithUsageHold() {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_35(mht_35_v, 754, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "GetBufferWithUsageHold");

    return GetBufferWithHold(ScopedHold::kUsage);
  }
  ScopedHold GetBufferWithExternalReference() {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_36(mht_36_v, 760, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "GetBufferWithExternalReference");

    return GetBufferWithHold(ScopedHold::kExternalReference);
  }

 private:
  bool IsEmptyTuple() const {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_37(mht_37_v, 768, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "IsEmptyTuple");

    return on_device_shape_.IsTuple() &&
           on_device_shape_.tuple_shapes_size() == 0;
  }

  StatusOr<tfrt::AsyncValueRef<Literal>> CopyToHostAsyncInternal(
      bool discard_cached_copy, absl::optional<xla::Layout> layout);

  // Requires holds_[kDonation] == 0 (i.e., WaitForOutstandingDonationHolds()
  // must be called first.)
  StatusOr<std::shared_ptr<TrackedTfrtCpuDeviceBuffer>> GetBufferForHoldLocked(
      ScopedHold::Type type) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Requires holds_[kDonation] == 0 (i.e., WaitForOutstandingDonationHolds()
  // must be called first.)
  void AcquireHoldLocked(ScopedHold* hold) ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  void ConvertUsageHold(TrackedTfrtCpuDeviceBuffer* buffer,
                        absl::Span<tfrt::AsyncValueRef<CpuEvent>> events);

  void ConfirmDonation(TrackedTfrtCpuDeviceBuffer* device_buffer);

  void DropHold(ScopedHold::Type type, TrackedTfrtCpuDeviceBuffer* buffer);

  void WaitForOutstandingUsageHolds() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void WaitForOutstandingDonationHold() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Similar to Delete, drops the buffer's reference to its associated device
  // memory, leaving the buffer in an invalid state, but returns the
  // TrackedTfrtCpuDeviceBuffer rather than freeing the device memory, so that
  // another framework can take ownership of it. The buffer returned from
  // Release may be safely dropped at any time even if it still has pending
  // async operations. The client should call Await before calling Release with
  // wait_for_operations_to_complete=false, to ensure that the host has
  // synchronized past any outstanding write operations to the buffer. If
  // wait_for_operations_to_complete=true the host will block until any
  // potentially outstanding asynchronous operations have completed before
  // returning, in which case it is safe to read or mutate the returned buffer.
  // If the buffer was shared via an external reference it is the client's
  // responsibility that accesses via that reference do not interfere with
  // accesses via the buffer returned from Release.
  StatusOr<std::shared_ptr<TrackedTfrtCpuDeviceBuffer>> Release(
      bool wait_for_operations_to_complete);

  TfrtCpuClient* client_;
  const Shape on_device_shape_;
  TfrtCpuDevice* const device_;

  mutable absl::Mutex mu_;
  std::shared_ptr<TrackedTfrtCpuDeviceBuffer> tracked_device_buffer_
      ABSL_GUARDED_BY(mu_);
  // Count of holds on the buffer.
  std::array<int, ScopedHold::Type::kMaxValue> holds_ ABSL_GUARDED_BY(mu_);
  // Cached definition event used for constructing PjRtFutures to wait on.
  tfrt::AsyncValueRef<Status> definition_event_ ABSL_GUARDED_BY(mu_);
};

class TfrtCpuExecutable final : public PjRtExecutable {
 public:
  TfrtCpuExecutable(
      int num_replicas, int num_partitions,
      std::shared_ptr<DeviceAssignment> device_assignment,
      bool parameter_is_tupled_arguments,
      std::unique_ptr<Executable> cpu_executable,
      BufferAllocation::Index result_buffer_index,
      absl::InlinedVector<BufferAllocation::Index, 4> result_buffer_indices,
      std::vector<LogicalDeviceIds> addressable_device_logical_ids,
      std::vector<PjRtDevice*> addressable_devices, TfrtCpuClient* client);

  ~TfrtCpuExecutable() override = default;

  TfrtCpuClient* client() const override {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_38(mht_38_v, 842, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "client");
 return client_; }

  absl::string_view name() const override {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_39(mht_39_v, 847, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "name");

    return cpu_executable_->shared_module()->name();
  }

  int num_replicas() const override {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_40(mht_40_v, 854, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "num_replicas");
 return num_replicas_; }

  int num_partitions() const override {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_41(mht_41_v, 859, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "num_partitions");
 return num_partitions_; }

  int64_t SizeOfGeneratedCodeInBytes() const override {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_42(mht_42_v, 864, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "SizeOfGeneratedCodeInBytes");

    return cpu_executable_->SizeOfGeneratedCodeInBytes();
  }

  const DeviceAssignment& device_assignment() const override {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpjrtPStfrt_cpu_pjrt_clientDTh mht_43(mht_43_v, 871, "", "./tensorflow/compiler/xla/pjrt/tfrt_cpu_pjrt_client.h", "device_assignment");

    return *device_assignment_;
  }

  absl::Span<const LogicalDeviceIds> addressable_device_logical_ids()
      const override {
    return addressable_device_logical_ids_;
  }

  absl::Span<PjRtDevice* const> addressable_devices() const override {
    return addressable_devices_;
  }

  StatusOr<std::vector<std::shared_ptr<HloModule>>> GetHloModules()
      const override {
    return std::vector<std::shared_ptr<HloModule>>{
        cpu_executable_->shared_module()};
  }

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

  void Delete() override;

  bool IsDeleted() override;

  StatusOr<absl::optional<std::string>> Fingerprint() const;

 private:
  friend class TfrtCpuClient;

  Status SetUpDonation(bool tuple_inputs);

  // Checks that the input buffers passed in by the user have the correct size
  // on device for the compiled program.
  Status CheckBufferCompatibilities(
      absl::Span<const std::shared_ptr<TrackedTfrtCpuDeviceBuffer>>
          input_buffers) const;

  StatusOr<Result> ExecuteHelper(
      absl::Span<PjRtBuffer* const> argument_handles, int replica,
      int partition, const RunId& run_id, const ExecuteOptions& options,
      tfrt::AsyncValueRef<CpuEvent> last_collective_launch_event,
      bool fill_future, TfrtCpuDevice* device = nullptr);

  TfrtCpuClient* client_;

  int num_replicas_;
  int num_partitions_;
  std::shared_ptr<DeviceAssignment> device_assignment_;
  bool parameter_is_tupled_arguments_;

  std::shared_ptr<Executable> cpu_executable_;

  // Caching `result_buffer_index_` and `result_buffer_indices_` to avoid lookup
  // HLO dataflow analysis data structures in program execution critical path.

  // Buffer allocation index corresponding to root buffer buffer.
  BufferAllocation::Index result_buffer_index_;
  // Buffer allocation indices corresponding to each result buffer leaf buffer.
  absl::InlinedVector<BufferAllocation::Index, 4> result_buffer_indices_;

  // Size on device of each leaf buffer of the compiled program, cached here
  // for performance reasons.
  std::vector<int64_t> input_buffer_sizes_in_bytes_;

  // A sorted vector of parameters that have any aliased buffers and thus must
  // be donated when executing the computation.
  std::vector<int> parameters_that_must_be_donated_;

  // The replica and partition indices of device_assignment_ to be run by this
  // client. On single-host platforms without partitioning, this is all
  // replicas (i.e. addressable_device_logical_ids_[i] = (i, 0)), but this may
  // not be the case on multi-host platforms. If there are 4 replicas and 2
  // partitions on a single host platform, size of
  // addressable_device_logical_ids_ is 4*2 = 8.
  std::vector<LogicalDeviceIds> addressable_device_logical_ids_;

  // addressable_devices_[i] is the Device to which
  // addressable_device_logical_ids_[i] is assigned. shared_ptrs instead of
  // unique_ptrs to play well with the Python bindings (see xla.cc).
  std::vector<PjRtDevice*> addressable_devices_;

  // Cached result of comparing HloCostAnalysis FLOP estimate for execute
  // critical path.
  bool cheap_computation_;
};

StatusOr<std::unique_ptr<PjRtClient>> GetTfrtCpuClient(bool asynchronous);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_TFRT_CPU_PJRT_CLIENT_H_
