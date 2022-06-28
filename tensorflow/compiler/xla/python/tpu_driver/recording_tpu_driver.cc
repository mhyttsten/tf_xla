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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSrecording_tpu_driverDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSrecording_tpu_driverDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSrecording_tpu_driverDTcc() {
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
// =============================================================================
#include <atomic>
#include <functional>

#include "absl/base/internal/sysinfo.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/python/tpu_driver/platform/external/compat.h"
#include "tensorflow/compiler/xla/python/tpu_driver/tpu_driver.h"
#include "tensorflow/compiler/xla/python/tpu_driver/tpu_driver.pb.h"
#include "tensorflow/compiler/xla/python/tpu_driver/tpu_service.grpc.pb.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/threadpool.h"

/*
 * The ReplayDriver wraps a concrete TpuDriver implementation and records the
 * stream of operations to a log file. This log can be later replayed and
 * analyzed for debugging.
 */

namespace tpu_driver {
namespace {

static std::atomic<int64_t> id_counter(0);

using xla::Status;

class RecordingTpuDriver;

class RecordingEvent : public Event {
 public:
  explicit RecordingEvent(std::shared_ptr<Event> event)
      : shared_event_(std::move(event)), id_(id_counter++) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSrecording_tpu_driverDTcc mht_0(mht_0_v, 216, "", "./tensorflow/compiler/xla/python/tpu_driver/recording_tpu_driver.cc", "RecordingEvent");
}

  explicit RecordingEvent(std::shared_ptr<Event> event, int64_t id)
      : shared_event_(event), id_(id) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSrecording_tpu_driverDTcc mht_1(mht_1_v, 222, "", "./tensorflow/compiler/xla/python/tpu_driver/recording_tpu_driver.cc", "RecordingEvent");
}

  ~RecordingEvent() override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSrecording_tpu_driverDTcc mht_2(mht_2_v, 227, "", "./tensorflow/compiler/xla/python/tpu_driver/recording_tpu_driver.cc", "~RecordingEvent");
}

  xla::Status Await() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSrecording_tpu_driverDTcc mht_3(mht_3_v, 232, "", "./tensorflow/compiler/xla/python/tpu_driver/recording_tpu_driver.cc", "Await");
 return shared_event_->Await(); }

  absl::optional<xla::Status> AwaitWithTimeout(
      absl::Duration duration) override {
    return shared_event_->AwaitWithTimeout(duration);
  }

  void AddCallback(std::function<void(xla::Status)> callback) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSrecording_tpu_driverDTcc mht_4(mht_4_v, 242, "", "./tensorflow/compiler/xla/python/tpu_driver/recording_tpu_driver.cc", "AddCallback");

    return shared_event_->AddCallback(callback);
  }

 private:
  std::shared_ptr<Event> shared_event_;

  int64_t id_;
  friend class RecordingTpuDriver;
};

class RecordingBufferHandle : public BufferHandle {
 public:
  explicit RecordingBufferHandle(std::unique_ptr<BufferHandle> handle)
      : handle_(std::move(handle)),
        id_(id_counter++),
        event_(std::make_shared<RecordingEvent>(handle_->OnReady(), id_)) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSrecording_tpu_driverDTcc mht_5(mht_5_v, 261, "", "./tensorflow/compiler/xla/python/tpu_driver/recording_tpu_driver.cc", "RecordingBufferHandle");
}
  std::shared_ptr<Event> OnReady() override { return event_; }
  int64_t size_in_bytes() override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSrecording_tpu_driverDTcc mht_6(mht_6_v, 266, "", "./tensorflow/compiler/xla/python/tpu_driver/recording_tpu_driver.cc", "size_in_bytes");
 return handle_->size_in_bytes(); }
  absl::optional<xla::ShapeProto> shape() override { return handle_->shape(); }

 private:
  std::unique_ptr<BufferHandle> handle_;
  int64_t id_;
  std::shared_ptr<RecordingEvent> event_;
  friend class RecordingTpuDriver;
};

class RecordingCompiledProgramHandle : public CompiledProgramHandle {
 public:
  explicit RecordingCompiledProgramHandle(
      std::unique_ptr<CompiledProgramHandle> handle)
      : handle_(std::move(handle)),
        id_(id_counter++),
        event_(std::make_shared<RecordingEvent>(handle_->OnReady(), id_)) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSrecording_tpu_driverDTcc mht_7(mht_7_v, 285, "", "./tensorflow/compiler/xla/python/tpu_driver/recording_tpu_driver.cc", "RecordingCompiledProgramHandle");
}
  std::shared_ptr<Event> OnReady() override { return event_; }
  int64_t size_in_bytes() override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSrecording_tpu_driverDTcc mht_8(mht_8_v, 290, "", "./tensorflow/compiler/xla/python/tpu_driver/recording_tpu_driver.cc", "size_in_bytes");
 return handle_->size_in_bytes(); }
  xla::Status program_shape(xla::ProgramShapeProto* program_shape) override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSrecording_tpu_driverDTcc mht_9(mht_9_v, 294, "", "./tensorflow/compiler/xla/python/tpu_driver/recording_tpu_driver.cc", "program_shape");

    return handle_->program_shape(program_shape);
  }

 private:
  std::unique_ptr<CompiledProgramHandle> handle_;
  int64_t id_;
  std::shared_ptr<RecordingEvent> event_;
  friend class RecordingTpuDriver;
};

class RecordingLoadedProgramHandle : public LoadedProgramHandle {
 public:
  explicit RecordingLoadedProgramHandle(
      std::unique_ptr<LoadedProgramHandle> handle)
      : handle_(std::move(handle)),
        id_(id_counter++),
        event_(std::make_shared<RecordingEvent>(handle_->OnReady(), id_)) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSrecording_tpu_driverDTcc mht_10(mht_10_v, 314, "", "./tensorflow/compiler/xla/python/tpu_driver/recording_tpu_driver.cc", "RecordingLoadedProgramHandle");
}
  std::shared_ptr<Event> OnReady() override { return event_; }
  int64_t size_in_bytes() override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSrecording_tpu_driverDTcc mht_11(mht_11_v, 319, "", "./tensorflow/compiler/xla/python/tpu_driver/recording_tpu_driver.cc", "size_in_bytes");
 return handle_->size_in_bytes(); }

 private:
  std::unique_ptr<LoadedProgramHandle> handle_;
  int64_t id_;
  std::shared_ptr<RecordingEvent> event_;
  friend class RecordingTpuDriver;
};

class RecordingTpuDriver : public TpuDriver {
 public:
  explicit RecordingTpuDriver(std::unique_ptr<TpuDriver> driver,
                              const std::string recording_path,
                              const bool flush)
      : driver_(std::move(driver)),
        recording_path_(recording_path),
        flush_(flush) {
   std::vector<std::string> mht_12_v;
   mht_12_v.push_back("recording_path: \"" + recording_path + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSrecording_tpu_driverDTcc mht_12(mht_12_v, 339, "", "./tensorflow/compiler/xla/python/tpu_driver/recording_tpu_driver.cc", "RecordingTpuDriver");

    auto file_status = tensorflow::Env::Default()->NewAppendableFile(
        recording_path_, &log_file_);
    if (!file_status.ok()) {
      LOG(FATAL) << "Unable to open " << recording_path_
                 << " for appending. Error: " << file_status.ToString();
    }
  }
  ~RecordingTpuDriver() override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSrecording_tpu_driverDTcc mht_13(mht_13_v, 350, "", "./tensorflow/compiler/xla/python/tpu_driver/recording_tpu_driver.cc", "~RecordingTpuDriver");

    {
      log_file_->Flush().IgnoreError();
      log_file_->Close().IgnoreError();
      log_file_ = nullptr;
    }
  }

  void QuerySystemInfo(SystemInfo* system_info) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSrecording_tpu_driverDTcc mht_14(mht_14_v, 361, "", "./tensorflow/compiler/xla/python/tpu_driver/recording_tpu_driver.cc", "QuerySystemInfo");

    // TODO(frankchn): Should we even save this event, since it is out-of-band.
    driver_->QuerySystemInfo(system_info);
  }

  Status Reset() override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSrecording_tpu_driverDTcc mht_15(mht_15_v, 369, "", "./tensorflow/compiler/xla/python/tpu_driver/recording_tpu_driver.cc", "Reset");
 return driver_->Reset(); }

  std::unique_ptr<BufferHandle> Allocate(
      int32_t core_id, MemoryRegion region, int64_t num_bytes,
      absl::Span<Event* const> wait_for) override {
    auto unwrapped_wait_for = UnwrapWaitFor(wait_for);

    auto thread_id = GetCurrentThreadId();
    auto handle =
        driver_->Allocate(core_id, region, num_bytes, unwrapped_wait_for);
    auto recording_handle =
        std::make_unique<RecordingBufferHandle>(std::move(handle));
    auto handle_id = recording_handle->id_;

    {
      StreamRequest::Entry r;
      r.mutable_alloc()->set_core_id(core_id);
      r.mutable_alloc()->set_region(region);
      r.mutable_alloc()->set_num_bytes(num_bytes);

      PopulateAndSaveEntry(&r, wait_for, handle_id, thread_id);
    }

    return recording_handle;
  }

  std::unique_ptr<BufferHandle> Allocate(
      int32_t core_id, MemoryRegion region, const xla::ShapeProto& shape,
      absl::Span<Event* const> wait_for) override {
    auto unwrapped_wait_for = UnwrapWaitFor(wait_for);

    auto thread_id = GetCurrentThreadId();
    auto handle = driver_->Allocate(core_id, region, shape, unwrapped_wait_for);
    auto recording_handle =
        std::make_unique<RecordingBufferHandle>(std::move(handle));
    auto handle_id = recording_handle->id_;

    {
      StreamRequest::Entry r;
      r.mutable_alloc()->set_core_id(core_id);
      r.mutable_alloc()->set_region(region);
      *(r.mutable_alloc()->mutable_shape()) = shape;

      PopulateAndSaveEntry(&r, wait_for, handle_id, thread_id);
    }

    return recording_handle;
  }

  std::unique_ptr<BufferHandle> AllocateTuple(
      int32_t core_id, MemoryRegion region,
      absl::Span<BufferHandle* const> children,
      absl::Span<Event* const> wait_for) override {
    auto unwrapped_wait_for = UnwrapWaitFor(wait_for);

    std::vector<BufferHandle*> unwrapped_children;
    std::vector<int64_t> child_ids;
    const auto children_size = children.size();
    unwrapped_children.reserve(children_size);
    child_ids.reserve(children_size);
    for (auto child : children) {
      BufferHandle* unwrapped_child =
          static_cast<const RecordingBufferHandle*>(child)->handle_.get();
      unwrapped_children.push_back(unwrapped_child);
      child_ids.push_back(
          static_cast<const RecordingBufferHandle*>(child)->id_);
    }

    auto thread_id = GetCurrentThreadId();
    auto handle = driver_->AllocateTuple(core_id, region, unwrapped_children,
                                         unwrapped_wait_for);
    auto recording_handle =
        std::make_unique<RecordingBufferHandle>(std::move(handle));
    auto handle_id = recording_handle->id_;

    {
      StreamRequest::Entry r;
      r.mutable_alloc_tuple()->set_core_id(core_id);
      r.mutable_alloc_tuple()->set_region(region);

      for (auto child : child_ids) {
        r.mutable_alloc_tuple()->add_children(child);
      }

      PopulateAndSaveEntry(&r, wait_for, handle_id, thread_id);
    }

    return recording_handle;
  }

  std::shared_ptr<Event> Deallocate(
      std::unique_ptr<BufferHandle> handle,
      absl::Span<Event* const> wait_for) override {
    auto unwrapped_wait_for = UnwrapWaitFor(wait_for);

    auto thread_id = GetCurrentThreadId();
    auto recording_handle = static_cast<RecordingBufferHandle*>(handle.get());
    int64_t recording_handle_id = recording_handle->id_;
    auto event = driver_->Deallocate(std::move(recording_handle->handle_),
                                     unwrapped_wait_for);
    auto recording_event = std::make_shared<RecordingEvent>(std::move(event));
    int64_t event_id = recording_event->id_;

    {
      StreamRequest::Entry r;
      r.mutable_dealloc()->set_handle(recording_handle_id);
      PopulateAndSaveEntry(&r, wait_for, event_id, thread_id);
    }

    return recording_event;
  }

  std::shared_ptr<Event> TransferToDevice(
      const void* src, BufferHandle* dst,
      absl::Span<Event* const> wait_for) override {
    int64_t num_bytes = dst->size_in_bytes();
    auto unwrapped_wait_for = UnwrapWaitFor(wait_for);

    auto thread_id = GetCurrentThreadId();
    auto recording_handle = static_cast<RecordingBufferHandle*>(dst);
    int64_t recording_handle_id = recording_handle->id_;
    auto recording_event =
        std::make_shared<RecordingEvent>(driver_->TransferToDevice(
            src, static_cast<RecordingBufferHandle*>(dst)->handle_.get(),
            unwrapped_wait_for));
    int64_t event_id = recording_event->id_;

    {
      StreamRequest::Entry r;
      r.mutable_transfer_to()->set_target_handle(recording_handle_id);
      if (num_bytes > 0) {
        r.mutable_transfer_to()->mutable_data()->assign(
            static_cast<const char*>(src), num_bytes);
      } else {
        *r.mutable_transfer_to()->mutable_data() = "";
      }
      PopulateAndSaveEntry(&r, wait_for, event_id, thread_id);
    }

    return recording_event;
  }

  std::shared_ptr<Event> TransferFromDevice(
      const BufferHandle* src, void* dst,
      absl::Span<Event* const> wait_for) override {
    auto unwrapped_wait_for = UnwrapWaitFor(wait_for);

    auto thread_id = GetCurrentThreadId();
    auto src_handle_id = static_cast<const RecordingBufferHandle*>(src)->id_;
    auto recording_event =
        std::make_shared<RecordingEvent>(driver_->TransferFromDevice(
            static_cast<const RecordingBufferHandle*>(src)->handle_.get(), dst,
            unwrapped_wait_for));
    auto event_id = recording_event->id_;

    {
      StreamRequest::Entry r;
      r.mutable_transfer_from()->set_source_handle(src_handle_id);
      PopulateAndSaveEntry(&r, wait_for, event_id, thread_id);
    }

    return recording_event;
  }

  std::shared_ptr<Event> TransferFromDeviceToDevice(
      const BufferHandle* src, BufferHandle* dst,
      absl::Span<Event* const> wait_for) override {
    auto unwrapped_wait_for = UnwrapWaitFor(wait_for);

    auto thread_id = GetCurrentThreadId();
    auto src_handle_id = static_cast<const RecordingBufferHandle*>(src)->id_;
    auto dst_handle_id = static_cast<const RecordingBufferHandle*>(dst)->id_;
    auto recording_event =
        std::make_shared<RecordingEvent>(driver_->TransferFromDeviceToDevice(
            static_cast<const RecordingBufferHandle*>(src)->handle_.get(),
            static_cast<const RecordingBufferHandle*>(dst)->handle_.get(),
            unwrapped_wait_for));
    auto event_id = recording_event->id_;

    {
      StreamRequest::Entry r;
      r.mutable_transfer_from_to()->set_source_handle(src_handle_id);
      r.mutable_transfer_from_to()->set_target_handle(dst_handle_id);
      PopulateAndSaveEntry(&r, wait_for, event_id, thread_id);
    }

    return recording_event;
  }

  std::unique_ptr<CompiledProgramHandle> CompileProgram(
      const xla::HloProto& source, int32_t num_replicas,
      absl::Span<Event* const> wait_for) override {
    auto unwrapped_wait_for = UnwrapWaitFor(wait_for);

    auto thread_id = GetCurrentThreadId();
    auto recording_handle = std::make_unique<RecordingCompiledProgramHandle>(
        driver_->CompileProgram(source, num_replicas, unwrapped_wait_for));
    auto handle_id = recording_handle->id_;

    {
      StreamRequest::Entry r;
      *r.mutable_compile()->mutable_hlo_program() = source;
      r.mutable_compile()->set_num_replicas(num_replicas);
      PopulateAndSaveEntry(&r, wait_for, handle_id, thread_id);
    }

    return recording_handle;
  }

  std::unique_ptr<LoadedProgramHandle> LoadProgram(
      int32_t core_id, const CompiledProgramHandle* handle,
      absl::Span<Event* const> wait_for) override {
    auto unwrapped_wait_for = UnwrapWaitFor(wait_for);

    auto thread_id = GetCurrentThreadId();
    auto compiled_handle_id =
        static_cast<const RecordingCompiledProgramHandle*>(handle)->id_;
    auto recording_handle =
        std::make_unique<RecordingLoadedProgramHandle>(driver_->LoadProgram(
            core_id,
            static_cast<const RecordingCompiledProgramHandle*>(handle)
                ->handle_.get(),
            unwrapped_wait_for));
    auto handle_id = recording_handle->id_;
    {
      StreamRequest::Entry r;
      r.mutable_load()->set_core_id(core_id);
      r.mutable_load()->set_compiled_program_handle(compiled_handle_id);
      PopulateAndSaveEntry(&r, wait_for, handle_id, thread_id);
    }

    return recording_handle;
  }

  std::shared_ptr<Event> UnloadProgram(
      std::unique_ptr<LoadedProgramHandle> handle,
      absl::Span<Event* const> wait_for) override {
    auto unwrapped_wait_for = UnwrapWaitFor(wait_for);

    auto thread_id = GetCurrentThreadId();
    auto loaded_handle_id =
        static_cast<RecordingLoadedProgramHandle*>(handle.get())->id_;
    auto recording_event =
        std::make_shared<RecordingEvent>(driver_->UnloadProgram(
            std::move(static_cast<RecordingLoadedProgramHandle*>(handle.get())
                          ->handle_),
            unwrapped_wait_for));
    auto event_id = recording_event->id_;

    {
      StreamRequest::Entry r;
      r.mutable_unload()->set_loaded_program_handle(loaded_handle_id);
      PopulateAndSaveEntry(&r, wait_for, event_id, thread_id);
    }

    return recording_event;
  }

  std::shared_ptr<Event> ExecuteProgram(
      LoadedProgramHandle* program, absl::Span<BufferHandle* const> inputs,
      absl::Span<BufferHandle* const> outputs,
      const xla::DeviceAssignmentProto& device_assignment,
      absl::Span<Event* const> wait_for) override {
    auto unwrapped_wait_for = UnwrapWaitFor(wait_for);

    auto thread_id = GetCurrentThreadId();
    auto program_handle_id =
        static_cast<RecordingLoadedProgramHandle*>(program)->id_;

    std::vector<BufferHandle*> unwrapped_inputs;
    std::vector<int64_t> input_ids;
    const auto inputs_size = inputs.size();
    unwrapped_inputs.reserve(inputs_size);
    input_ids.reserve(inputs_size);
    for (auto input : inputs) {
      BufferHandle* unwrapped_input =
          static_cast<const RecordingBufferHandle*>(input)->handle_.get();
      unwrapped_inputs.push_back(unwrapped_input);
      input_ids.push_back(
          static_cast<const RecordingBufferHandle*>(input)->id_);
    }

    std::vector<BufferHandle*> unwrapped_outputs;
    std::vector<int64_t> output_ids;
    const auto output_size = outputs.size();
    unwrapped_outputs.reserve(output_size);
    output_ids.reserve(output_size);
    for (auto output : outputs) {
      BufferHandle* unwrapped_output =
          static_cast<const RecordingBufferHandle*>(output)->handle_.get();
      unwrapped_outputs.push_back(unwrapped_output);
      output_ids.push_back(
          static_cast<const RecordingBufferHandle*>(output)->id_);
    }

    auto recording_event =
        std::make_shared<RecordingEvent>(driver_->ExecuteProgram(
            static_cast<RecordingLoadedProgramHandle*>(program)->handle_.get(),
            unwrapped_inputs, unwrapped_outputs, device_assignment,
            unwrapped_wait_for));
    auto event_id = recording_event->id_;

    {
      StreamRequest::Entry r;
      r.mutable_execute()->set_loaded_program_handle(program_handle_id);
      for (auto input_id : input_ids) {
        r.mutable_execute()->add_input_handle(input_id);
      }
      for (auto output_id : output_ids) {
        r.mutable_execute()->add_output_handle(output_id);
      }
      *r.mutable_execute()->mutable_device_assignment() = device_assignment;

      PopulateAndSaveEntry(&r, wait_for, event_id, thread_id);
    }

    return recording_event;
  }

  std::unique_ptr<TpuLinearizer> GetLinearizer() override {
    return driver_->GetLinearizer();
  }

 private:
  std::unique_ptr<TpuDriver> driver_;
  const std::string recording_path_;
  const bool flush_;

  std::unique_ptr<tensorflow::WritableFile> log_file_;

  void PopulateAndSaveEntry(StreamRequest::Entry* r,
                            absl::Span<Event* const> wait_for,
                            int64_t handle_id, int64_t thread_id) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSrecording_tpu_driverDTcc mht_16(mht_16_v, 704, "", "./tensorflow/compiler/xla/python/tpu_driver/recording_tpu_driver.cc", "PopulateAndSaveEntry");

    for (auto event : wait_for) {
      auto recording_event = static_cast<const RecordingEvent*>(event);
      r->add_wait_for_id(recording_event->id_);
    }
    r->set_operation_id(handle_id);
    r->set_thread_id(thread_id);

    uint64_t data_size = r->ByteSizeLong();
    std::vector<char> buffer;
    buffer.resize(sizeof(data_size) + data_size);
    memcpy(buffer.data(), &data_size, sizeof(data_size));
    r->SerializeToArray(buffer.data() + sizeof(data_size), data_size);

    {
      if (log_file_ == nullptr) {
        LOG(WARNING) << "The TPU driver has been shut down before all logging "
                        "has been written.";
        return;
      }

      absl::string_view buffer_sp(buffer.data(), buffer.size());
      auto data_status = log_file_->Append(buffer_sp);
      if (!data_status.ok()) {
        LOG(WARNING) << "Unable to write data to log file. File possibly "
                        "corrupt. Error: "
                     << data_status.ToString();
      }

      if (flush_) {
        auto flush_status = log_file_->Flush();
        if (!flush_status.ok()) {
          LOG(WARNING) << "Unable to flush data to log file. File possibly "
                          "corrupt. Error: "
                       << flush_status.ToString();
        }

        auto sync_status = log_file_->Sync();
        if (!sync_status.ok()) {
          LOG(WARNING) << "Unable to sync log file. File possibly "
                          "corrupt. Error: "
                       << sync_status.ToString();
        }
      }
    }
  }

  std::vector<Event*> UnwrapWaitFor(absl::Span<Event* const> wait_for) {
    std::vector<Event*> unwrapped_events;
    for (auto event : wait_for) {
      Event* unwrapped_event =
          static_cast<RecordingEvent*>(event)->shared_event_.get();
      unwrapped_events.push_back(unwrapped_event);
    }
    return unwrapped_events;
  }

  int64_t GetCurrentThreadId() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSrecording_tpu_driverDTcc mht_17(mht_17_v, 764, "", "./tensorflow/compiler/xla/python/tpu_driver/recording_tpu_driver.cc", "GetCurrentThreadId");
 return absl::base_internal::GetTID(); }
};

xla::StatusOr<std::unique_ptr<TpuDriver>> RegisterRecordingTpuDriver(
    const TpuDriverConfig& config) {
  std::vector<std::string> configs = absl::StrSplit(config.worker(), '|');

  std::string file;
  std::string worker;
  bool flush = false;

  for (const auto& config : configs) {
    std::vector<std::string> kv =
        absl::StrSplit(config, absl::MaxSplits('=', 1));
    if (kv[0] == "file") {
      file = kv[1];
    }
    if (kv[0] == "worker") {
      worker = kv[1];
    }
    if (kv[0] == "flush") {
      if (kv[1] == "true" || kv[1] == "1") {
        flush = true;
      }
    }
  }

  TpuDriverConfig worker_config;
  worker_config.set_worker(worker);

  auto driver_status = TpuDriverRegistry::Open(worker_config);
  if (!driver_status.ok()) return driver_status.status();
  auto driver = driver_status.ConsumeValueOrDie();

  return std::unique_ptr<TpuDriver>(
      new RecordingTpuDriver(std::move(driver), file, flush));
}

// To record a sequence of operations, set the worker configuration string to
// record://|file=<filename>|worker=grpc://1.2.3.4:8470 (for GRPC).
REGISTER_TPU_DRIVER("record://", RegisterRecordingTpuDriver);

}  // namespace
}  // namespace tpu_driver
