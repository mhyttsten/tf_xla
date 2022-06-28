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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc() {
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

#include <complex>
#include <cstddef>
#include <functional>
#include <memory>

#include "absl/base/thread_annotations.h"
#include "absl/strings/strip.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "grpcpp/grpcpp.h"
#include "tensorflow/compiler/xla/python/tpu_driver/event_id.h"
#include "tensorflow/compiler/xla/python/tpu_driver/platform/external/compat.h"
#include "tensorflow/compiler/xla/python/tpu_driver/tpu_driver.h"
#include "tensorflow/compiler/xla/python/tpu_driver/tpu_driver.pb.h"
#include "tensorflow/compiler/xla/python/tpu_driver/tpu_service.grpc.pb.h"
#include "tensorflow/compiler/xla/util.h"

namespace tpu_driver {
namespace {

using xla::Status;

const int64_t kMaxStreamWriteSize = 10 * 1000 * 1000;
const absl::Duration kWriteEpochDuration = absl::Microseconds(10);

constexpr char kGrpcProtocol[] = "grpc://";

class GrpcTpuStream;
class GrpcTpuDriver;

class GrpcEvent : public Event {
 public:
  explicit GrpcEvent(EventId id, GrpcTpuStream* stream)
      : id_(id), stream_(stream) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_0(mht_0_v, 220, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcEvent");
}
  ~GrpcEvent() override;

  xla::Status Await() override;
  absl::optional<xla::Status> AwaitWithTimeout(
      absl::Duration duration) override;
  void AddCallback(std::function<void(Status)> callback) override;

  EventId id() const {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_1(mht_1_v, 231, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "id");
 return id_; }
  GrpcTpuStream* stream() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_2(mht_2_v, 235, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "stream");
 return stream_; }

 private:
  const EventId id_;
  GrpcTpuStream* stream_;
};

class ErrorEvent : public GrpcEvent {
 public:
  explicit ErrorEvent(Status status) : GrpcEvent(EventId{0, 0}, nullptr) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_3(mht_3_v, 247, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "ErrorEvent");

    status_ = status;
  }

  xla::Status Await() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_4(mht_4_v, 254, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "Await");
 return status_; }
  absl::optional<xla::Status> AwaitWithTimeout(
      absl::Duration duration) override {
    return status_;
  }
  void AddCallback(std::function<void(Status)> callback) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_5(mht_5_v, 262, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "AddCallback");

    callback(status_);
  }

 private:
  Status status_;
};

class GrpcBufferHandle : public BufferHandle {
 public:
  explicit GrpcBufferHandle(
      EventId id, std::shared_ptr<GrpcEvent> event, int64_t bytes,
      absl::optional<xla::ShapeProto> shape = absl::nullopt)
      : id_(id),
        stream_(event->stream()),
        event_(std::move(event)),
        bytes_(bytes),
        shape_(shape) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_6(mht_6_v, 282, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcBufferHandle");
}

  std::shared_ptr<Event> OnReady() override { return event_; }
  int64_t size_in_bytes() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_7(mht_7_v, 288, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "size_in_bytes");
 return bytes_; }

  EventId id() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_8(mht_8_v, 293, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "id");
 return id_; }
  GrpcTpuStream* stream() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_9(mht_9_v, 297, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "stream");
 return stream_; }

  absl::optional<xla::ShapeProto> shape() override { return shape_; }

 private:
  const EventId id_;
  GrpcTpuStream* stream_;
  std::shared_ptr<GrpcEvent> event_;
  int64_t bytes_;
  absl::optional<xla::ShapeProto> shape_;
};

class GrpcCompiledProgramHandle : public CompiledProgramHandle {
 public:
  explicit GrpcCompiledProgramHandle(EventId id,
                                     std::shared_ptr<GrpcEvent> event)
      : id_(id),
        stream_(event->stream()),
        event_(std::move(event)),
        metadata_(std::make_shared<CompiledProgramMetadata>()) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_10(mht_10_v, 319, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcCompiledProgramHandle");
}

  std::shared_ptr<Event> OnReady() override { return event_; }

  EventId id() const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_11(mht_11_v, 326, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "id");
 return id_; }
  GrpcTpuStream* stream() const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_12(mht_12_v, 330, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "stream");
 return stream_; }

  Status program_shape(xla::ProgramShapeProto* program_shape) override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_13(mht_13_v, 335, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "program_shape");

    auto opt_status = OnReady()->AwaitWithTimeout(absl::Hours(1));
    if (!opt_status.has_value()) {
      return xla::InternalError("Compile failed to finish within 1 hour.");
    }

    Status status = opt_status.value();
    if (!status.ok()) {
      return status;
    }
    *program_shape = metadata_->program_shape();
    return Status::OK();
  }

  std::shared_ptr<CompiledProgramMetadata> metadata() { return metadata_; }

 private:
  const EventId id_;
  GrpcTpuStream* stream_;
  std::shared_ptr<GrpcEvent> event_;

  // Using a shared pointer here because the program handle can go out of scope
  // before we get a response back, but we want a valid location to write things
  // into regardless.
  std::shared_ptr<CompiledProgramMetadata> metadata_;
};

class GrpcLoadedProgramHandle : public LoadedProgramHandle {
 public:
  explicit GrpcLoadedProgramHandle(EventId id, std::shared_ptr<GrpcEvent> event)
      : id_(id), stream_(event->stream()), event_(std::move(event)) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_14(mht_14_v, 368, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcLoadedProgramHandle");
}

  std::shared_ptr<Event> OnReady() override { return event_; }

  EventId id() const {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_15(mht_15_v, 375, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "id");
 return id_; }
  GrpcTpuStream* stream() const {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_16(mht_16_v, 379, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "stream");
 return stream_; }

 private:
  const EventId id_;
  GrpcTpuStream* stream_;
  std::shared_ptr<GrpcEvent> event_;
};

class GrpcTpuStream {
 public:
  explicit GrpcTpuStream(int32_t id, GrpcTpuDriver* driver,
                         std::unique_ptr<grpc::CloudTpuDriver::Stub> stub);
  virtual ~GrpcTpuStream();

  std::unique_ptr<BufferHandle> Allocate(int32_t core_id, MemoryRegion region,
                                         int64_t num_bytes,
                                         absl::Span<Event* const> wait_for);
  std::unique_ptr<BufferHandle> Allocate(int32_t core_id, MemoryRegion region,
                                         const xla::ShapeProto& shape,
                                         absl::Span<Event* const> wait_for);
  std::unique_ptr<BufferHandle> AllocateTuple(
      int32_t core_id, MemoryRegion region,
      absl::Span<BufferHandle* const> children,
      absl::Span<Event* const> wait_for);
  std::shared_ptr<Event> Deallocate(std::unique_ptr<BufferHandle> handle,
                                    absl::Span<Event* const> wait_for);

  std::shared_ptr<Event> TransferToDevice(const void* src, BufferHandle* dst,
                                          absl::Span<Event* const> wait_for);
  std::shared_ptr<Event> TransferFromDevice(const BufferHandle* src, void* dst,
                                            absl::Span<Event* const> wait_for);

  std::shared_ptr<Event> TransferFromDeviceToDevice(
      const BufferHandle* src, BufferHandle* dst,
      absl::Span<Event* const> wait_for);

  std::unique_ptr<CompiledProgramHandle> CompileProgram(
      const xla::HloProto& source, int32_t num_replicas,
      absl::Span<Event* const> wait_for);
  std::unique_ptr<LoadedProgramHandle> LoadProgram(
      int32_t core_id, const CompiledProgramHandle* handle,
      absl::Span<Event* const> wait_for);
  std::shared_ptr<Event> UnloadProgram(
      std::unique_ptr<LoadedProgramHandle> handle,
      absl::Span<Event* const> wait_for);
  std::shared_ptr<Event> ExecuteProgram(
      LoadedProgramHandle* program, absl::Span<BufferHandle* const> inputs,
      absl::Span<BufferHandle* const> outputs,
      const xla::DeviceAssignmentProto& device_assignment,
      absl::Span<Event* const> wait_for);

 private:
  friend class GrpcEvent;
  friend class GrpcTpuDriver;

  struct EventInfo {
    bool all_deps_done = false;
    bool done = false;     // response received
    bool deleted = false;  // deleted by the user
    Status status;
    absl::InlinedVector<std::function<void(Status)>, 1> callbacks;
    // Most events should have <= 2 requirement events.
    absl::InlinedVector<EventId, 2> deps;
  };

  struct TransferInfo {
    explicit TransferInfo(void* dst, int64_t num_bytes)
        : dst(dst), num_bytes(num_bytes) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_17(mht_17_v, 449, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "TransferInfo");
}

    void* const dst;
    const uint64_t num_bytes;
  };

  struct CompileMetadataInfo {
    explicit CompileMetadataInfo(
        std::shared_ptr<CompiledProgramMetadata> metadata) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_18(mht_18_v, 460, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "CompileMetadataInfo");

      compiled_metadata = metadata;
    }
    std::shared_ptr<CompiledProgramMetadata> compiled_metadata;
  };

  // Every public method above should call this first.
  void InitializeRequest(StreamRequest::Entry* req,
                         absl::Span<Event* const> wait_for)
      ABSL_LOCKS_EXCLUDED(events_mutex_);

  // The first update to an event marks it done and calls registered callbacks.
  // All subsequent updates must have the same OK-ness as the first update.
  // Among non-OK updates, only the first error status is remembered.
  void UpdateEventStatus(EventId id, Status status)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(events_mutex_);

  // To ensure callbacks are still triggered, after this is called, we do not
  // remove the event from the event mapping until a response is received from
  // the server.
  void DeleteEvent(EventId id) ABSL_LOCKS_EXCLUDED(events_mutex_);

  // Wait at most `duration` for event `id` to complete. Returns the event
  // status or an empty optional if the event does not complete in time.
  absl::optional<Status> WaitForEvent(EventId id, absl::Duration duration)
      ABSL_LOCKS_EXCLUDED(events_mutex_);

  void AddEventCallback(EventId id, std::function<void(Status)> callback)
      ABSL_LOCKS_EXCLUDED(events_mutex_);

  void AddWriteRequest(std::unique_ptr<StreamRequest::Entry> req) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_19(mht_19_v, 493, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "AddWriteRequest");

    absl::MutexLock m(&request_lock_);
    VLOG(2) << "Adding request: " << req->DebugString();
    requests_.push_back(std::move(req));
  }

  // Unique identifier for this stream.
  int32_t id_;
  // The parent driver that created this stream.
  GrpcTpuDriver* driver_;

  std::unique_ptr<grpc::CloudTpuDriver::Stub> stub_;
  ::grpc::ClientContext ctx_;
  std::unique_ptr<
      ::grpc::ClientReaderWriterInterface<StreamRequest, StreamResponse>>
      stream_;

  absl::Mutex request_lock_;
  std::deque<std::unique_ptr<StreamRequest::Entry>> requests_
      ABSL_GUARDED_BY(request_lock_);
  int64_t num_pending_requests_ ABSL_GUARDED_BY(request_lock_) = 0;

  bool shutting_down_ ABSL_GUARDED_BY(request_lock_) = false;

  void StreamWriterFn();
  Thread writer_thread_;

  void StreamReaderFn();
  Thread reader_thread_;

  // Map from operation ID to event information.
  absl::Mutex events_mutex_;
  absl::flat_hash_map<EventId, EventInfo> events_
      ABSL_GUARDED_BY(events_mutex_);

  // Map from operation ID to transfer information.
  // When a D2H transfer completes, received data is copied into the `dst`
  // pointer in `TransferInfo`.
  absl::Mutex transfers_mutex_;
  absl::flat_hash_map<EventId, TransferInfo> transfers_
      ABSL_GUARDED_BY(transfers_mutex_);

  absl::Mutex compiles_mutex_;
  absl::flat_hash_map<EventId, CompileMetadataInfo> compiles_
      ABSL_GUARDED_BY(compiles_mutex_);
};

class GrpcTpuDriver : public TpuDriver {
 public:
  explicit GrpcTpuDriver(const TpuDriverConfig& config,
                         std::shared_ptr<::grpc::ChannelCredentials> creds,
                         int32_t client_id)
      : config_(config), creds_(creds), client_id_(client_id) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_20(mht_20_v, 548, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcTpuDriver");

    SystemInfo system_info;
    QuerySystemInfo(&system_info);
    for (auto& chip_info : system_info.tpu_chip()) {
      for (auto& core_info : chip_info.core()) {
        int32_t core_id = core_info.id();
        // We have one stream per core, so use core ID as stream ID.
        streams_[core_id] = AllocateStream(core_id);
      }
    }
    CHECK_GT(streams_.size(), 0) << "Can't find any TPU chip in the system.";

    host_stream_ = AllocateStream(-1);
  }

  ~GrpcTpuDriver() override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_21(mht_21_v, 566, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "~GrpcTpuDriver");

    if (closed_) {
      return;
    }
    auto status = Close();
    if (!status.ok()) {
      LOG(ERROR) << status;
    }
  }

  void QuerySystemInfo(SystemInfo* system_info) override;
  Status Reset() override;

  std::unique_ptr<BufferHandle> Allocate(
      int32_t core_id, MemoryRegion region, int64_t num_bytes,
      absl::Span<Event* const> wait_for) override {
    return streams_[core_id]->Allocate(core_id, region, num_bytes, wait_for);
  }
  std::unique_ptr<BufferHandle> Allocate(
      int32_t core_id, MemoryRegion region, const xla::ShapeProto& shape,
      absl::Span<Event* const> wait_for) override {
    return streams_[core_id]->Allocate(core_id, region, shape, wait_for);
  }
  std::unique_ptr<BufferHandle> AllocateTuple(
      int32_t core_id, MemoryRegion region,
      absl::Span<BufferHandle* const> children,
      absl::Span<Event* const> wait_for) override {
    return streams_[core_id]->AllocateTuple(core_id, region, children,
                                            wait_for);
  }
  std::shared_ptr<Event> Deallocate(
      std::unique_ptr<BufferHandle> handle,
      absl::Span<Event* const> wait_for) override {
    auto* stream = static_cast<GrpcBufferHandle*>(handle.get())->stream();
    return stream->Deallocate(std::move(handle), wait_for);
  }

  std::shared_ptr<Event> TransferToDevice(
      const void* src, BufferHandle* dst,
      absl::Span<Event* const> wait_for) override {
    auto* stream = static_cast<GrpcBufferHandle*>(dst)->stream();
    return stream->TransferToDevice(src, dst, wait_for);
  }
  std::shared_ptr<Event> TransferFromDevice(
      const BufferHandle* src, void* dst,
      absl::Span<Event* const> wait_for) override {
    auto* stream = static_cast<const GrpcBufferHandle*>(src)->stream();
    return stream->TransferFromDevice(src, dst, wait_for);
  }

  std::shared_ptr<Event> TransferFromDeviceToDevice(
      const BufferHandle* src, BufferHandle* dst,
      absl::Span<Event* const> wait_for) override {
    auto* stream = static_cast<const GrpcBufferHandle*>(src)->stream();
    return stream->TransferFromDeviceToDevice(src, dst, wait_for);
  }

  std::unique_ptr<CompiledProgramHandle> CompileProgram(
      const xla::HloProto& source, int32_t num_replicas,
      absl::Span<Event* const> wait_for) override {
    // Always compile using the first/default core's stream.
    return streams_[0]->CompileProgram(source, num_replicas, wait_for);
  }
  std::unique_ptr<LoadedProgramHandle> LoadProgram(
      int32_t core_id, const CompiledProgramHandle* handle,
      absl::Span<Event* const> wait_for) override {
    return streams_[core_id]->LoadProgram(core_id, handle, wait_for);
  }
  std::shared_ptr<Event> UnloadProgram(
      std::unique_ptr<LoadedProgramHandle> handle,
      absl::Span<Event* const> wait_for) override {
    auto* stream =
        static_cast<const GrpcLoadedProgramHandle*>(handle.get())->stream();
    return stream->UnloadProgram(std::move(handle), wait_for);
  }
  std::shared_ptr<Event> ExecuteProgram(
      LoadedProgramHandle* program, absl::Span<BufferHandle* const> inputs,
      absl::Span<BufferHandle* const> outputs,
      const xla::DeviceAssignmentProto& device_assignment,
      absl::Span<Event* const> wait_for) override {
    auto* stream =
        static_cast<const GrpcLoadedProgramHandle*>(program)->stream();
    return stream->ExecuteProgram(program, inputs, outputs, device_assignment,
                                  wait_for);
  }

  EventId NewOperationId() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_22(mht_22_v, 655, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "NewOperationId");
 return EventId{client_id_, ++operation_id_}; }

  static std::unique_ptr<grpc::CloudTpuDriver::Stub> CreateTpuDriverStub(
      const TpuDriverConfig& config,
      std::shared_ptr<::grpc::ChannelCredentials> creds);

  uint32_t client_id() const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_23(mht_23_v, 664, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "client_id");
 return client_id_; }

 private:
  Status Close();
  std::unique_ptr<GrpcTpuStream> AllocateStream(int32_t core_id);

  const TpuDriverConfig config_;
  std::shared_ptr<::grpc::ChannelCredentials> creds_;
  const uint32_t client_id_;
  // Map from stream IDs to streams.
  absl::flat_hash_map<int32_t, std::unique_ptr<GrpcTpuStream>> streams_;
  std::unique_ptr<GrpcTpuStream> host_stream_;
  // Shared by all streams.
  std::atomic<uint64_t> operation_id_{0};
  std::atomic<bool> closed_{false};
};  // namespace

GrpcEvent::~GrpcEvent() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_24(mht_24_v, 684, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcEvent::~GrpcEvent");
 stream_->DeleteEvent(id_); }

Status GrpcEvent::Await() {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_25(mht_25_v, 689, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcEvent::Await");

  auto opt_status = stream_->WaitForEvent(id_, absl::InfiniteDuration());
  return opt_status.value();
}

absl::optional<Status> GrpcEvent::AwaitWithTimeout(absl::Duration duration) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_26(mht_26_v, 697, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcEvent::AwaitWithTimeout");

  return stream_->WaitForEvent(id_, duration);
}

void GrpcEvent::AddCallback(std::function<void(Status)> callback) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_27(mht_27_v, 704, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcEvent::AddCallback");

  stream_->AddEventCallback(id_, std::move(callback));
}

GrpcTpuStream::GrpcTpuStream(int32_t id, GrpcTpuDriver* driver,
                             std::unique_ptr<grpc::CloudTpuDriver::Stub> stub)
    : id_(id),
      driver_(driver),
      stub_(std::move(stub)),
      stream_(stub_->StreamExecute(&ctx_)),
      writer_thread_(&GrpcTpuStream::StreamWriterFn, this),
      reader_thread_(&GrpcTpuStream::StreamReaderFn, this) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_28(mht_28_v, 718, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcTpuStream::GrpcTpuStream");
}

GrpcTpuStream::~GrpcTpuStream() {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_29(mht_29_v, 723, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcTpuStream::~GrpcTpuStream");

  {
    absl::MutexLock lock(&request_lock_);
    shutting_down_ = true;
  }

  VLOG(1) << "Shutting down stream.";
  {
    // Mark all remaining events invalid.
    absl::MutexLock lock(&events_mutex_);
    for (const auto& e : events_) {
      if (!e.second.done) {
        LOG(ERROR) << "Resetting: " << e.first;
        UpdateEventStatus(e.first, xla::Status(tensorflow::error::Code::ABORTED,
                                               "Driver was closed."));
      }
    }
  }
  VLOG(1) << "Closing stream.";
  stream_->WritesDone();
  stream_->Finish().IgnoreError();
  VLOG(1) << "Waiting for writer.";
  writer_thread_.join();
  VLOG(1) << "Waiting for reader.";
  reader_thread_.join();
}

void GrpcTpuStream::InitializeRequest(StreamRequest::Entry* req,
                                      absl::Span<Event* const> wait_for) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_30(mht_30_v, 754, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcTpuStream::InitializeRequest");

  auto operation_id = driver_->NewOperationId();
  EventInfo event_info;

  req->set_operation_id(operation_id.AsInt());
  if (wait_for.empty()) {
    event_info.all_deps_done = true;
  } else {
    event_info.deps.reserve(wait_for.size());
    for (auto* event : wait_for) {
      auto grpc_event = static_cast<const GrpcEvent*>(event);
      req->add_wait_for_id(grpc_event->id().AsInt());
      event_info.deps.push_back(grpc_event->id());
    }
  }

  absl::MutexLock lock(&events_mutex_);
  events_[operation_id] = event_info;
}

void GrpcTpuStream::UpdateEventStatus(EventId id, Status status) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_31(mht_31_v, 777, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcTpuStream::UpdateEventStatus");

  auto it = events_.find(id);

  // These should only happen when the server shuts down, and our local event
  // cancellation interleaves with server responses. It should be safe to ignore
  // the second updates in these situations.
  if (it == events_.end()) {
    VLOG(1) << "Received a status update: " << status
            << ", but cannot find GrpcEvent " << id;
    return;
  }
  if (it->second.done) {
    // Done and deleted events must have already been removed.
    CHECK(!it->second.deleted);
    VLOG(1) << "Received a second status update: " << status.error_message()
            << ", for GrpcEvent " << id << " already done with status: "
            << it->second.status.error_message();
    return;
  }

  // This is the first time this event finishes. Remember the results and call
  // the callbacks.
  VLOG(1) << "Response received for GrpcEvent " << id << ". "
          << status.ToString() << ". Firing " << it->second.callbacks.size()
          << " callbacks.";
  it->second.done = true;
  it->second.status = status;
  for (const auto& callback : it->second.callbacks) {
    callback(status);
  }

  // Truly remove the event if it's both done and deleted.
  if (it->second.deleted) {
    events_.erase(it);
  }
}

void GrpcTpuStream::DeleteEvent(EventId id) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_32(mht_32_v, 817, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcTpuStream::DeleteEvent");

  absl::MutexLock lock(&events_mutex_);
  auto it = events_.find(id);
  CHECK(it != events_.end());
  CHECK(!it->second.deleted);
  it->second.deleted = true;
  // Truly remove the event if it's both done and deleted.
  if (it->second.done) {
    events_.erase(it);
  }
}

absl::optional<Status> GrpcTpuStream::WaitForEvent(EventId id,
                                                   absl::Duration duration) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_33(mht_33_v, 833, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcTpuStream::WaitForEvent");

  events_mutex_.Lock();
  auto it = events_.find(id);

  if (it == events_.end()) {
    // This event has already been marked as done and deleted. Assume success.
    events_mutex_.Unlock();
    return Status::OK();
  }

  if (!it->second.all_deps_done) {
    absl::InlinedVector<EventId, 2> deps = it->second.deps;
    events_mutex_.Unlock();
    for (auto dep : deps) {
      // If a requirement event timed out, no point in any further waiting.
      if (!WaitForEvent(dep, duration)) {
        return absl::nullopt;
      }
    }
    events_mutex_.Lock();
  }

  // Set the flag here, as we're guaranteed they have all completed at this
  // point. This helps terminate recursion on a chain of completed events as
  // soon as possible, at this event.
  it = events_.find(id);
  if (it != events_.end()) {
    it->second.all_deps_done = true;
  }

  auto done = [this, id]() {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_34(mht_34_v, 866, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "lambda");

    events_mutex_.AssertHeld();
    return !events_.contains(id) || events_[id].done;
  };
  if (events_mutex_.AwaitWithTimeout(absl::Condition(&done), duration)) {
    auto status = events_.contains(id) ? events_[id].status : Status::OK();
    events_mutex_.Unlock();
    return status;
  }
  events_mutex_.Unlock();
  return absl::nullopt;
}

void GrpcTpuStream::AddEventCallback(EventId id,
                                     std::function<void(Status)> callback) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_35(mht_35_v, 883, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcTpuStream::AddEventCallback");

  absl::MutexLock lock(&events_mutex_);
  auto it = events_.find(id);
  if (it == events_.end()) {
    callback(Status());
    return;
  }
  if (it->second.done) {
    callback(it->second.status);
    return;
  }
  it->second.callbacks.push_back(std::move(callback));
}

static bool ShouldBeginWriting(int64_t* pending_requests) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_36(mht_36_v, 900, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "ShouldBeginWriting");

  return *pending_requests > 32;
}

void GrpcTpuStream::StreamWriterFn() {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_37(mht_37_v, 907, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcTpuStream::StreamWriterFn");

  while (true) {
    request_lock_.LockWhenWithTimeout(
        absl::Condition(&ShouldBeginWriting, &num_pending_requests_),
        kWriteEpochDuration);
    if (shutting_down_) {
      request_lock_.Unlock();
      return;
    }

    if (requests_.empty()) {
      request_lock_.Unlock();
      continue;
    }

    std::vector<StreamRequest> reqs;
    int64_t request_bytes = 0;
    while (!requests_.empty()) {
      StreamRequest::Entry* e = requests_.front().release();
      requests_.pop_front();
      const int64_t entry_bytes = e->ByteSizeLong();
      if (reqs.empty() || request_bytes + entry_bytes > kMaxStreamWriteSize) {
        reqs.push_back(StreamRequest());
        request_bytes = 0;
      }
      VLOG(1) << "Sending request: " << EventId::FromInt(e->operation_id());
      VLOG(2) << "Sending request: " << e->DebugString();
      reqs.back().mutable_entry()->AddAllocated(e);
    }
    num_pending_requests_ = 0;
    request_lock_.Unlock();

    for (const auto& r : reqs) {
      TraceMe activity("GrpcTpuStream::Send ");
      ::grpc::WriteOptions opts;
      opts.set_no_compression().clear_buffer_hint();
      stream_->Write(r, opts);
    }
  }
}

void GrpcTpuStream::StreamReaderFn() {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_38(mht_38_v, 951, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcTpuStream::StreamReaderFn");

  StreamResponse resp;
  while (stream_->Read(&resp)) {
    VLOG(2) << "Received response: " << resp.DebugString();
    for (const StreamResponse::Entry& entry : resp.entry()) {
      EventId event_id = EventId::FromInt(entry.operation_id());
      VLOG(1) << "Received response for: " << event_id;

      TraceMe activity("GrpcTpuStream::RequestComplete");
      if (entry.has_transfer_from()) {
        TraceMe activity("GrpcTpuStream::TransferFromComplete");
        absl::MutexLock lock(&transfers_mutex_);
        auto it = transfers_.find(event_id);
        CHECK(it != transfers_.end());
        VLOG(1) << "Copying: " << it->second.num_bytes << " to position "
                << it->second.dst;
        if (entry.transfer_from().data().size() != it->second.num_bytes) {
          absl::MutexLock lock(&events_mutex_);
          UpdateEventStatus(
              event_id,
              Status(
                  tensorflow::error::Code::DATA_LOSS,
                  absl::StrCat("Expected ", it->second.num_bytes, " received ",
                               entry.transfer_from().data().size())));
          continue;
        }
        memcpy(it->second.dst, entry.transfer_from().data().data(),
               it->second.num_bytes);
      }

      if (entry.has_compile()) {
        TraceMe activity("GrpcTpuStream::CompileComplete");
        absl::MutexLock lock(&compiles_mutex_);
        auto it = compiles_.find(event_id);
        CHECK(it != compiles_.end());
        *it->second.compiled_metadata = entry.compile().metadata();
      }

      absl::MutexLock lock(&events_mutex_);
      if (entry.status().code() != tensorflow::error::Code::OK) {
        UpdateEventStatus(
            event_id,
            Status(static_cast<tensorflow::error::Code>(entry.status().code()),
                   entry.status().message()));
      } else {
        UpdateEventStatus(event_id, Status::OK());
      }
    }
  }
}

std::unique_ptr<BufferHandle> GrpcTpuStream::Allocate(
    int32_t core_id, MemoryRegion region, int64_t num_bytes,
    absl::Span<Event* const> wait_for) {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_39(mht_39_v, 1007, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcTpuStream::Allocate");

  auto req = absl::make_unique<StreamRequest::Entry>();
  InitializeRequest(req.get(), wait_for);
  TraceMe activity("GrpcTpuStream::Allocate(num_bytes)");
  req->mutable_alloc()->set_core_id(core_id);
  req->mutable_alloc()->set_region(region);
  req->mutable_alloc()->set_num_bytes(num_bytes);
  auto event =
      std::make_shared<GrpcEvent>(EventId::FromInt(req->operation_id()), this);
  AddWriteRequest(std::move(req));
  return absl::make_unique<GrpcBufferHandle>(event->id(), std::move(event),
                                             num_bytes);
}

std::unique_ptr<BufferHandle> GrpcTpuStream::Allocate(
    int32_t core_id, MemoryRegion region, const xla::ShapeProto& shape,
    absl::Span<Event* const> wait_for) {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_40(mht_40_v, 1026, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcTpuStream::Allocate");

  auto req = absl::make_unique<StreamRequest::Entry>();
  InitializeRequest(req.get(), wait_for);
  TraceMe activity("GrpcTpuStream::Allocate(shape)");
  req->mutable_alloc()->set_core_id(core_id);
  req->mutable_alloc()->set_region(region);
  *req->mutable_alloc()->mutable_shape() = shape;
  auto event =
      std::make_shared<GrpcEvent>(EventId::FromInt(req->operation_id()), this);
  AddWriteRequest(std::move(req));
  return absl::make_unique<GrpcBufferHandle>(
      event->id(), std::move(event), ComputeBytesFromShape(shape), shape);
}

std::unique_ptr<BufferHandle> GrpcTpuStream::AllocateTuple(
    int32_t core_id, MemoryRegion region,
    absl::Span<BufferHandle* const> children,
    absl::Span<Event* const> wait_for) {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_41(mht_41_v, 1046, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcTpuStream::AllocateTuple");

  auto req = absl::make_unique<StreamRequest::Entry>();
  InitializeRequest(req.get(), wait_for);
  TraceMe activity("GrpcTpuStream::AllocateTuple");
  req->mutable_alloc_tuple()->set_core_id(core_id);
  req->mutable_alloc_tuple()->set_region(region);
  for (auto child : children) {
    auto grpc_child = static_cast<GrpcBufferHandle*>(child);
    req->mutable_alloc_tuple()->add_children(grpc_child->id().AsInt());
  }
  auto event =
      std::make_shared<GrpcEvent>(EventId::FromInt(req->operation_id()), this);
  AddWriteRequest(std::move(req));
  return absl::make_unique<GrpcBufferHandle>(event->id(), std::move(event), 0);
}

std::shared_ptr<Event> GrpcTpuStream::Deallocate(
    std::unique_ptr<BufferHandle> handle, absl::Span<Event* const> wait_for) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_42(mht_42_v, 1066, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcTpuStream::Deallocate");

  auto req = absl::make_unique<StreamRequest::Entry>();
  InitializeRequest(req.get(), wait_for);
  TraceMe activity("GrpcTpuStream::Deallocate");
  auto grpc_handle = static_cast<GrpcBufferHandle*>(handle.get());
  req->mutable_dealloc()->set_handle(grpc_handle->id().AsInt());
  auto event =
      std::make_shared<GrpcEvent>(EventId::FromInt(req->operation_id()), this);
  AddWriteRequest(std::move(req));
  return event;
}

std::shared_ptr<Event> GrpcTpuStream::TransferToDevice(
    const void* src, BufferHandle* dst, absl::Span<Event* const> wait_for) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_43(mht_43_v, 1082, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcTpuStream::TransferToDevice");

  auto req = absl::make_unique<StreamRequest::Entry>();
  InitializeRequest(req.get(), wait_for);
  TraceMe activity("GrpcTpuStream::TransferToDevice");
  req->mutable_transfer_to()->mutable_data()->assign(
      static_cast<const char*>(src), dst->size_in_bytes());
  req->mutable_transfer_to()->set_target_handle(
      static_cast<GrpcBufferHandle*>(dst)->id().AsInt());
  auto event =
      std::make_shared<GrpcEvent>(EventId::FromInt(req->operation_id()), this);
  AddWriteRequest(std::move(req));
  return event;
}

std::shared_ptr<Event> GrpcTpuStream::TransferFromDevice(
    const BufferHandle* src, void* dst, absl::Span<Event* const> wait_for) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_44(mht_44_v, 1100, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcTpuStream::TransferFromDevice");

  auto req = absl::make_unique<StreamRequest::Entry>();
  InitializeRequest(req.get(), wait_for);
  TraceMe activity("GrpcTpuStream::TransferFromDevice");
  req->mutable_transfer_from()->set_source_handle(
      static_cast<const GrpcBufferHandle*>(src)->id().AsInt());
  EventId event_id = EventId::FromInt(req->operation_id());
  {
    absl::MutexLock lock(&transfers_mutex_);
    TransferInfo info(dst, const_cast<BufferHandle*>(src)->size_in_bytes());
    transfers_.insert(std::make_pair(event_id, info));
  }
  auto event = std::make_shared<GrpcEvent>(event_id, this);
  AddWriteRequest(std::move(req));
  return event;
}

std::shared_ptr<Event> GrpcTpuStream::TransferFromDeviceToDevice(
    const BufferHandle* src, BufferHandle* dst,
    absl::Span<Event* const> wait_for) {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_45(mht_45_v, 1122, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcTpuStream::TransferFromDeviceToDevice");

  auto req = absl::make_unique<StreamRequest::Entry>();
  InitializeRequest(req.get(), wait_for);
  TraceMe activity([&req] {
    return absl::StrCat("GrpcTpuStream::TransferFromDeviceToDevice",
                        req->operation_id());
  });

  req->mutable_transfer_from_to()->set_source_handle(
      static_cast<const GrpcBufferHandle*>(src)->id().AsInt());
  req->mutable_transfer_from_to()->set_target_handle(
      static_cast<const GrpcBufferHandle*>(dst)->id().AsInt());
  EventId event_id = EventId::FromInt(req->operation_id());
  auto event = std::make_shared<GrpcEvent>(event_id, this);
  AddWriteRequest(std::move(req));
  return event;
}

std::unique_ptr<CompiledProgramHandle> GrpcTpuStream::CompileProgram(
    const xla::HloProto& source, int32_t num_replicas,
    absl::Span<Event* const> wait_for) {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_46(mht_46_v, 1145, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcTpuStream::CompileProgram");

  auto req = absl::make_unique<StreamRequest::Entry>();
  InitializeRequest(req.get(), wait_for);
  TraceMe activity("GrpcTpuStream::CompileProgram");
  *req->mutable_compile()->mutable_hlo_program() = source;
  req->mutable_compile()->set_num_replicas(num_replicas);
  EventId event_id = EventId::FromInt(req->operation_id());

  auto event =
      std::make_shared<GrpcEvent>(EventId::FromInt(req->operation_id()), this);

  auto handle = absl::make_unique<GrpcCompiledProgramHandle>(event->id(),
                                                             std::move(event));
  {
    absl::MutexLock lock(&compiles_mutex_);
    CompileMetadataInfo info(handle->metadata());
    compiles_.insert(std::make_pair(event_id, info));
  }

  AddWriteRequest(std::move(req));
  return std::move(handle);
}

std::unique_ptr<LoadedProgramHandle> GrpcTpuStream::LoadProgram(
    int32_t core_id, const CompiledProgramHandle* handle,
    absl::Span<Event* const> wait_for) {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_47(mht_47_v, 1173, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcTpuStream::LoadProgram");

  auto req = absl::make_unique<StreamRequest::Entry>();
  InitializeRequest(req.get(), wait_for);
  TraceMe activity("GrpcTpuStream::LoadProgram");
  req->mutable_load()->set_core_id(core_id);
  auto grpc_handle = static_cast<const GrpcCompiledProgramHandle*>(handle);
  if (grpc_handle->id().client_id != driver_->client_id()) {
    auto event = std::make_shared<ErrorEvent>(
        xla::InvalidArgument("Invalid program handle (wrong client id). Did "
                             "you restart the server or use a stale handle?"));
    return absl::make_unique<GrpcLoadedProgramHandle>(event->id(),
                                                      std::move(event));
  }
  req->mutable_load()->set_compiled_program_handle(grpc_handle->id().AsInt());
  auto event =
      std::make_shared<GrpcEvent>(EventId::FromInt(req->operation_id()), this);
  AddWriteRequest(std::move(req));
  return absl::make_unique<GrpcLoadedProgramHandle>(event->id(),
                                                    std::move(event));
}

std::shared_ptr<Event> GrpcTpuStream::UnloadProgram(
    std::unique_ptr<LoadedProgramHandle> handle,
    absl::Span<Event* const> wait_for) {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_48(mht_48_v, 1199, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcTpuStream::UnloadProgram");

  auto req = absl::make_unique<StreamRequest::Entry>();
  InitializeRequest(req.get(), wait_for);
  TraceMe activity("GrpcTpuStream::UnloadProgram");
  req->mutable_unload()->set_loaded_program_handle(
      static_cast<GrpcLoadedProgramHandle*>(handle.get())->id().AsInt());
  auto event =
      std::make_shared<GrpcEvent>(EventId::FromInt(req->operation_id()), this);
  AddWriteRequest(std::move(req));
  return event;
}

std::shared_ptr<Event> GrpcTpuStream::ExecuteProgram(
    LoadedProgramHandle* program, absl::Span<BufferHandle* const> inputs,
    absl::Span<BufferHandle* const> outputs,
    const xla::DeviceAssignmentProto& device_assignment,
    absl::Span<Event* const> wait_for) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_49(mht_49_v, 1218, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcTpuStream::ExecuteProgram");

  auto req = absl::make_unique<StreamRequest::Entry>();
  InitializeRequest(req.get(), wait_for);
  auto program_handle = static_cast<GrpcLoadedProgramHandle*>(program);
  if (program_handle->id().client_id != driver_->client_id()) {
    return std::make_shared<ErrorEvent>(
        xla::InvalidArgument("Invalid program handle (wrong client id). Did "
                             "you restart the server or use a stale handle?"));
  }

  req->mutable_execute()->set_loaded_program_handle(
      program_handle->id().AsInt());

  for (BufferHandle* input : inputs) {
    auto* grpc_handle = static_cast<GrpcBufferHandle*>(input);
    if (grpc_handle->id().client_id != driver_->client_id()) {
      return std::make_shared<ErrorEvent>(xla::InvalidArgument(
          "Invalid input buffer (wrong client id). Did you restart the server "
          "or use a stale handle?"));
    }
    req->mutable_execute()->add_input_handle(grpc_handle->id().AsInt());
  }

  for (BufferHandle* output : outputs) {
    auto* grpc_handle = static_cast<GrpcBufferHandle*>(output);
    if (grpc_handle->id().client_id != driver_->client_id()) {
      return std::make_shared<ErrorEvent>(xla::InvalidArgument(
          "Invalid output buffer (wrong client id). Did you restart the server "
          "or use a stale handle?"));
    }
    req->mutable_execute()->add_output_handle(
        static_cast<GrpcBufferHandle*>(output)->id().AsInt());
  }
  // Only pass along device_assignment if it's not default constructed.
  if (!(device_assignment.replica_count() == 0 &&
        device_assignment.computation_count() == 0)) {
    *req->mutable_execute()->mutable_device_assignment() = device_assignment;
  }
  auto event =
      std::make_shared<GrpcEvent>(EventId::FromInt(req->operation_id()), this);
  AddWriteRequest(std::move(req));
  return event;
}

/*static*/ std::unique_ptr<grpc::CloudTpuDriver::Stub>
GrpcTpuDriver::CreateTpuDriverStub(
    const TpuDriverConfig& config,
    std::shared_ptr<::grpc::ChannelCredentials> creds) {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_50(mht_50_v, 1268, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcTpuDriver::CreateTpuDriverStub");

  ::grpc::ChannelArguments args;
  args.SetMaxReceiveMessageSize(std::numeric_limits<int>::max());
  args.SetMaxSendMessageSize(std::numeric_limits<int>::max());

  // Send at least 20 keep-alives before giving up.
  int keepalive_timeout_ms = config.grpc().keepalive_timeout_secs() * 1000;
  int keepalive_interval_ms = keepalive_timeout_ms / 20;

  grpc_arg client_arg_vals[] = {
      {.type = GRPC_ARG_INTEGER,
       .key = const_cast<char*>(
           GRPC_ARG_HTTP2_MIN_RECV_PING_INTERVAL_WITHOUT_DATA_MS),
       .value = {.integer = keepalive_interval_ms}},
      {.type = GRPC_ARG_INTEGER,
       .key = const_cast<char*>(GRPC_ARG_HTTP2_MAX_PINGS_WITHOUT_DATA),
       .value = {.integer = 0}},  // unlimited
      {.type = GRPC_ARG_INTEGER,
       .key = const_cast<char*>(GRPC_ARG_KEEPALIVE_TIME_MS),
       .value = {.integer = keepalive_interval_ms}},
      {.type = GRPC_ARG_INTEGER,
       .key = const_cast<char*>(GRPC_ARG_KEEPALIVE_TIMEOUT_MS),
       .value = {.integer = keepalive_timeout_ms}},
      {.type = GRPC_ARG_INTEGER,
       .key = const_cast<char*>(GRPC_ARG_KEEPALIVE_PERMIT_WITHOUT_CALLS),
       .value = {.integer = 1}},
      {.type = GRPC_ARG_INTEGER,
       .key = const_cast<char*>(GRPC_ARG_HTTP2_WRITE_BUFFER_SIZE),
       .value = {.integer = 64 * 1000 * 1000}}};

  grpc_channel_args client_args = {.num_args = 6, .args = client_arg_vals};
  args.SetChannelArgs(&client_args);

  // strips out 'grpc://'
  auto worker_addr = absl::StripPrefix(config.worker(), kGrpcProtocol);
  std::shared_ptr<::grpc::Channel> channel =
      ::grpc::CreateCustomChannel(std::string(worker_addr), creds, args);
  return grpc::CloudTpuDriver::NewStub(channel);
}

std::unique_ptr<GrpcTpuStream> GrpcTpuDriver::AllocateStream(int32_t id) {
   std::vector<std::string> mht_51_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_51(mht_51_v, 1311, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcTpuDriver::AllocateStream");

  auto stub = CreateTpuDriverStub(config_, creds_);
  ::grpc::ClientContext ctx;
  ctx.set_fail_fast(false);
  ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
  return absl::make_unique<GrpcTpuStream>(id, this, std::move(stub));
}

void GrpcTpuDriver::QuerySystemInfo(SystemInfo* system_info) {
   std::vector<std::string> mht_52_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_52(mht_52_v, 1322, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcTpuDriver::QuerySystemInfo");

  auto stub = CreateTpuDriverStub(config_, creds_);
  ::grpc::ClientContext ctx;
  ctx.set_fail_fast(false);
  ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));

  QuerySystemInfoRequest req;
  QuerySystemInfoResponse resp;
  ::grpc::Status status = stub->QuerySystemInfo(&ctx, req, &resp);
  if (!status.ok()) {
    LOG(ERROR) << "QuerySystemInfo request failed: " << status.error_code()
               << ": " << status.error_message() << ": "
               << status.error_details();
    return;
  }
  *system_info = resp.system_info();
}

Status GrpcTpuDriver::Reset() {
   std::vector<std::string> mht_53_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_53(mht_53_v, 1343, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcTpuDriver::Reset");

  auto stub = CreateTpuDriverStub(config_, creds_);
  ::grpc::ClientContext ctx;
  ctx.set_fail_fast(false);
  ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
  ResetRequest req;
  ResetResponse resp;
  ::grpc::Status status = stub->Reset(&ctx, req, &resp);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to reset the gRPC driver: " << status.error_code()
               << ": " << status.error_message() << ": "
               << status.error_details();
    return xla::Status(tensorflow::error::Code(status.error_code()),
                       absl::StrCat("Failed to reset TPU driver. Error was: ",
                                    status.error_message(),
                                    ". Details: ", status.error_details()));
  }
  streams_.clear();
  host_stream_.reset();
  return Close();
}

Status GrpcTpuDriver::Close() {
   std::vector<std::string> mht_54_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSgrpc_tpu_driverDTcc mht_54(mht_54_v, 1368, "", "./tensorflow/compiler/xla/python/tpu_driver/grpc_tpu_driver.cc", "GrpcTpuDriver::Close");

  auto stub = CreateTpuDriverStub(config_, creds_);
  ::grpc::ClientContext ctx;
  ctx.set_fail_fast(false);
  ctx.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(10));
  CloseRequest req;
  req.set_client_id(client_id_);
  CloseResponse resp;
  ::grpc::Status status = stub->Close(&ctx, req, &resp);
  if (!status.ok()) {
    return xla::Status(tensorflow::error::Code(status.error_code()),
                       absl::StrCat("Failed to close TPU driver. Error was: ",
                                    status.error_message(),
                                    ". Details: ", status.error_details()));
  }
  closed_ = true;
  return Status::OK();
}
}  // namespace

xla::StatusOr<std::unique_ptr<TpuDriver>> CreateGrpcTpuDriver(
    const TpuDriverConfig& config,
    std::shared_ptr<::grpc::ChannelCredentials> creds) {
  auto stub = GrpcTpuDriver::CreateTpuDriverStub(config, creds);
  ::grpc::ClientContext ctx;
  ctx.set_fail_fast(false);
  ctx.set_deadline(
      std::chrono::system_clock::now() +
      std::chrono::seconds(config.grpc().connection_timeout_secs()));
  OpenRequest req;
  OpenResponse resp;
  ::grpc::Status status = stub->Open(&ctx, req, &resp);
  if (!status.ok()) {
    LOG(ERROR) << "Failed to open the gRPC driver: " << status.error_code()
               << ": " << status.error_message() << ": "
               << status.error_details();
    return xla::Status(
        tensorflow::error::Code(status.error_code()),
        absl::StrCat(
            "Failed to connect to remote server at address: ", config.worker(),
            ". Error from gRPC: ", status.error_message(),
            ". Details: ", status.error_details()));
  }
  return std::unique_ptr<TpuDriver>(
      new GrpcTpuDriver(config, creds, resp.client_id()));
}

REGISTER_TPU_DRIVER(
    "grpc://",
    [](const TpuDriverConfig& config)
        -> xla::StatusOr<std::unique_ptr<TpuDriver>> {
      if (absl::StartsWith(config.worker(), "grpc://localhost")) {
        LOG(INFO) << "Using local credentials for localhost: connection.";
        return CreateGrpcTpuDriver(
            config, ::grpc::experimental::LocalCredentials(LOCAL_TCP));
      } else {
        return CreateGrpcTpuDriver(config,
                                   ::grpc::InsecureChannelCredentials());
      }
    });

}  // namespace tpu_driver
