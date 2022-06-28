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
class MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSdirect_tpu_driverDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSdirect_tpu_driverDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSdirect_tpu_driverDTcc() {
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

#include <dlfcn.h>

#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "tensorflow/compiler/xla/python/tpu_driver/client/libtpu.h"
#include "tensorflow/compiler/xla/python/tpu_driver/tpu_driver.h"
#include "tensorflow/compiler/xla/python/tpu_driver/tpu_driver.pb.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace tpu_driver {
namespace {

// Enable the macro by default in the Google internal environment where the
// libtpu.so is linked in statically.
#ifdef PLATFORM_GOOGLE
#define TPU_SHARED_LIBRARY_COMPILE_LINK 1
#endif

xla::Status CreateXlaStatus(::TpuStatus* status) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSdirect_tpu_driverDTcc mht_0(mht_0_v, 205, "", "./tensorflow/compiler/xla/python/tpu_driver/direct_tpu_driver.cc", "CreateXlaStatus");

  if (status->code == tensorflow::error::OK) {
    return xla::Status::OK();
  } else {
    return xla::Status(tensorflow::error::Code(status->code),
                       absl::StrFormat("%s", status->msg));
  }
}

constexpr char kDirectProtocol[] = "direct://";

::TpuAllocationShape GetTpuAllocationShape(const xla::ShapeProto& shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSdirect_tpu_driverDTcc mht_1(mht_1_v, 219, "", "./tensorflow/compiler/xla/python/tpu_driver/direct_tpu_driver.cc", "GetTpuAllocationShape");

  ::TpuAllocationShape shape_;
  shape_.size = shape.ByteSizeLong();
  shape_.bytes = malloc(shape_.size);
  if (!shape.SerializeToArray(shape_.bytes, shape_.size)) {
    LOG(ERROR) << "Unable to serialize shape to array.";
    free(shape_.bytes);
    shape_.size = 0;
    shape_.bytes = nullptr;
  }
  return shape_;
}

class DirectTpuDriver;

class DirectEvent : public Event {
 public:
  explicit DirectEvent(::TpuDriverFn* driver_fn, ::TpuEvent* event)
      : driver_fn_(driver_fn), event_(event) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSdirect_tpu_driverDTcc mht_2(mht_2_v, 240, "", "./tensorflow/compiler/xla/python/tpu_driver/direct_tpu_driver.cc", "DirectEvent");
}

  ~DirectEvent() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSdirect_tpu_driverDTcc mht_3(mht_3_v, 245, "", "./tensorflow/compiler/xla/python/tpu_driver/direct_tpu_driver.cc", "~DirectEvent");
 driver_fn_->TpuDriver_FreeEvent(event_); }

  xla::Status Await() override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSdirect_tpu_driverDTcc mht_4(mht_4_v, 250, "", "./tensorflow/compiler/xla/python/tpu_driver/direct_tpu_driver.cc", "Await");

    auto tpu_status = driver_fn_->TpuDriver_EventAwait(event_, -1);
    auto ret = CreateXlaStatus(tpu_status);
    driver_fn_->TpuDriver_FreeStatus(tpu_status);
    return ret;
  }

  absl::optional<xla::Status> AwaitWithTimeout(
      absl::Duration duration) override {
    auto tpu_status_or = driver_fn_->TpuDriver_EventAwait(
        event_, absl::ToInt64Microseconds(duration));
    if (tpu_status_or == nullptr) {
      return absl::nullopt;
    } else {
      auto ret = CreateXlaStatus(tpu_status_or);
      driver_fn_->TpuDriver_FreeStatus(tpu_status_or);
      return ret;
    }
  }

  void AddCallback(std::function<void(xla::Status)> callback) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSdirect_tpu_driverDTcc mht_5(mht_5_v, 273, "", "./tensorflow/compiler/xla/python/tpu_driver/direct_tpu_driver.cc", "AddCallback");

    // We have to create a new copy of the fn on the heap to make it persist.
    std::function<void(xla::Status)>* callback_addr =
        new std::function<void(xla::Status)>(callback);

    // Using the callback_addr instead of capturing because C++11 lambdas with
    // variable captures cannot be converted to C function pointers.
    driver_fn_->TpuDriver_EventAddCallback(
        event_,
        [](struct TpuStatus* status, void* additional_info) {
          auto callback_addr =
              static_cast<std::function<void(xla::Status)>*>(additional_info);
          auto xla_status = CreateXlaStatus(status);
          (*callback_addr)(xla_status);
          delete callback_addr;
        },
        callback_addr);
  }

 private:
  ::TpuDriverFn* driver_fn_;
  ::TpuEvent* event_;

  friend DirectTpuDriver;
};

class DirectBufferHandle : public BufferHandle {
 public:
  explicit DirectBufferHandle(::TpuDriverFn* driver_fn,
                              ::TpuBufferHandle* handle)
      : handle_(handle), event_(new DirectEvent(driver_fn, handle->event)) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSdirect_tpu_driverDTcc mht_6(mht_6_v, 306, "", "./tensorflow/compiler/xla/python/tpu_driver/direct_tpu_driver.cc", "DirectBufferHandle");
}

  std::shared_ptr<Event> OnReady() override { return event_; }

  int64_t size_in_bytes() override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSdirect_tpu_driverDTcc mht_7(mht_7_v, 313, "", "./tensorflow/compiler/xla/python/tpu_driver/direct_tpu_driver.cc", "size_in_bytes");
 return handle_->size_in_bytes; }

  absl::optional<xla::ShapeProto> shape() override {
    LOG(FATAL) << "Unimplemented.";
    return absl::nullopt;
  }

 private:
  ::TpuBufferHandle* handle_;
  std::shared_ptr<DirectEvent> event_;

  friend DirectTpuDriver;
};

class DirectCompiledProgramHandle : public CompiledProgramHandle {
 public:
  explicit DirectCompiledProgramHandle(::TpuDriverFn* driver_fn,
                                       ::TpuCompiledProgramHandle* handle)
      : handle_(handle),
        driver_fn_(driver_fn),
        event_(new DirectEvent(driver_fn, handle->event)) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSdirect_tpu_driverDTcc mht_8(mht_8_v, 336, "", "./tensorflow/compiler/xla/python/tpu_driver/direct_tpu_driver.cc", "DirectCompiledProgramHandle");
}

  ~DirectCompiledProgramHandle() override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSdirect_tpu_driverDTcc mht_9(mht_9_v, 341, "", "./tensorflow/compiler/xla/python/tpu_driver/direct_tpu_driver.cc", "~DirectCompiledProgramHandle");

    driver_fn_->TpuDriver_FreeCompiledProgramHandle(handle_);
  }

  std::shared_ptr<Event> OnReady() override { return event_; }

  int64_t size_in_bytes() override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSdirect_tpu_driverDTcc mht_10(mht_10_v, 350, "", "./tensorflow/compiler/xla/python/tpu_driver/direct_tpu_driver.cc", "size_in_bytes");

    LOG(FATAL) << "Unimplemented.";
    return 0;
  }

  xla::Status program_shape(xla::ProgramShapeProto* program_shape) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSdirect_tpu_driverDTcc mht_11(mht_11_v, 358, "", "./tensorflow/compiler/xla/python/tpu_driver/direct_tpu_driver.cc", "program_shape");

    struct CompiledProgramShape* shape =
        driver_fn_->TpuDriver_GetCompiledProgramShape(handle_);
    program_shape->ParseFromArray(shape->bytes, shape->size);

    auto status = CreateXlaStatus(shape->status);
    driver_fn_->TpuDriver_FreeCompiledProgramShape(shape);
    return status;
  }

 private:
  ::TpuCompiledProgramHandle* handle_;
  ::TpuDriverFn* driver_fn_;
  std::shared_ptr<DirectEvent> event_;

  friend DirectTpuDriver;
};

class DirectLoadedProgramHandle : public LoadedProgramHandle {
 public:
  explicit DirectLoadedProgramHandle(::TpuDriverFn* driver_fn,
                                     ::TpuLoadedProgramHandle* handle)
      : handle_(handle), event_(new DirectEvent(driver_fn, handle->event)) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSdirect_tpu_driverDTcc mht_12(mht_12_v, 383, "", "./tensorflow/compiler/xla/python/tpu_driver/direct_tpu_driver.cc", "DirectLoadedProgramHandle");
}
  std::shared_ptr<Event> OnReady() override { return event_; }

  int64_t size_in_bytes() override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSdirect_tpu_driverDTcc mht_13(mht_13_v, 389, "", "./tensorflow/compiler/xla/python/tpu_driver/direct_tpu_driver.cc", "size_in_bytes");

    LOG(FATAL) << "Unimplemented.";
    return 0;
  }

 private:
  ::TpuLoadedProgramHandle* handle_;
  std::shared_ptr<DirectEvent> event_;

  friend DirectTpuDriver;
};

class DirectTpuLinearizer : public TpuLinearizer {
 public:
  explicit DirectTpuLinearizer(::TpuDriver* driver, ::TpuDriverFn* driver_fn)
      : driver_(driver), driver_fn_(driver_fn) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSdirect_tpu_driverDTcc mht_14(mht_14_v, 407, "", "./tensorflow/compiler/xla/python/tpu_driver/direct_tpu_driver.cc", "DirectTpuLinearizer");
}

  int64_t ComputeLinearizedBytesFromShape(
      const xla::ShapeProto& shape) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSdirect_tpu_driverDTcc mht_15(mht_15_v, 413, "", "./tensorflow/compiler/xla/python/tpu_driver/direct_tpu_driver.cc", "ComputeLinearizedBytesFromShape");

    ::TpuAllocationShape shape_ = GetTpuAllocationShape(shape);
    uint64_t size =
        driver_fn_->TpuDriver_ComputeLinearizedBytesFromShape(driver_, shape_);
    free(shape_.bytes);
    return size;
  }

  xla::Status LinearizeShape(void* dst, const void* src,
                             const xla::ShapeProto& shape) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSdirect_tpu_driverDTcc mht_16(mht_16_v, 425, "", "./tensorflow/compiler/xla/python/tpu_driver/direct_tpu_driver.cc", "LinearizeShape");

    ::TpuAllocationShape shape_ = GetTpuAllocationShape(shape);

    auto tpu_status =
        driver_fn_->TpuDriver_LinearizeShape(driver_, dst, src, shape_);
    auto status = CreateXlaStatus(tpu_status);
    driver_fn_->TpuDriver_FreeStatus(tpu_status);
    free(shape_.bytes);
    return status;
  }

  xla::Status DelinearizeShape(void* dst, const void* src,
                               const xla::ShapeProto& shape) override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSdirect_tpu_driverDTcc mht_17(mht_17_v, 440, "", "./tensorflow/compiler/xla/python/tpu_driver/direct_tpu_driver.cc", "DelinearizeShape");

    ::TpuAllocationShape shape_ = GetTpuAllocationShape(shape);

    auto tpu_status =
        driver_fn_->TpuDriver_DelinearizeShape(driver_, dst, src, shape_);
    auto status = CreateXlaStatus(tpu_status);
    driver_fn_->TpuDriver_FreeStatus(tpu_status);
    free(shape_.bytes);
    return status;
  }

 private:
  ::TpuDriver* driver_;
  ::TpuDriverFn* driver_fn_;
};

class DirectTpuDriver : public TpuDriver {
 public:
  explicit DirectTpuDriver(const std::string& so_path) {
   std::vector<std::string> mht_18_v;
   mht_18_v.push_back("so_path: \"" + so_path + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSdirect_tpu_driverDTcc mht_18(mht_18_v, 462, "", "./tensorflow/compiler/xla/python/tpu_driver/direct_tpu_driver.cc", "DirectTpuDriver");

    void* handle;
    handle = dlopen(so_path.c_str(), RTLD_NOW);
    if (!handle) {
      LOG(FATAL) << "Unable to load shared library: " << dlerror();
    }

    PrototypeTpuDriver_Initialize* initialize_fn;
    *reinterpret_cast<void**>(&initialize_fn) =
        dlsym(handle, "TpuDriver_Initialize");
    initialize_fn(&driver_fn_, /*initialize=*/true);

    driver_ = driver_fn_.TpuDriver_Open("local://");
  }

#ifdef TPU_SHARED_LIBRARY_COMPILE_LINK
  DirectTpuDriver() {
    TpuDriver_Initialize(&driver_fn_, /*initialize=*/false);
    driver_ = driver_fn_.TpuDriver_Open("local://");
  }
#endif

  ~DirectTpuDriver() override { driver_fn_.TpuDriver_Close(driver_); }

  void QuerySystemInfo(SystemInfo* system_info) override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSdirect_tpu_driverDTcc mht_19(mht_19_v, 489, "", "./tensorflow/compiler/xla/python/tpu_driver/direct_tpu_driver.cc", "QuerySystemInfo");

    ::TpuSystemInfo* info = driver_fn_.TpuDriver_QuerySystemInfo(driver_);
    system_info->ParseFromArray(info->bytes, info->size);
    driver_fn_.TpuDriver_FreeSystemInfo(info);
  }

  xla::Status Reset() override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSdirect_tpu_driverDTcc mht_20(mht_20_v, 498, "", "./tensorflow/compiler/xla/python/tpu_driver/direct_tpu_driver.cc", "Reset");

    auto tpu_status = driver_fn_.TpuDriver_Reset(driver_);
    auto status = CreateXlaStatus(tpu_status);
    driver_fn_.TpuDriver_FreeStatus(tpu_status);
    return status;
  }

  std::unique_ptr<BufferHandle> Allocate(
      int32_t core_id, MemoryRegion region, int64_t num_bytes,
      absl::Span<Event* const> wait_for) override {
    auto tpu_events = MakeEventArray(wait_for);
    auto bh = absl::make_unique<DirectBufferHandle>(
        &driver_fn_,
        driver_fn_.TpuDriver_Allocate(driver_, core_id, region, num_bytes,
                                      wait_for.size(), tpu_events));
    delete[] tpu_events;
    return bh;
  }

  std::unique_ptr<BufferHandle> Allocate(
      int32_t core_id, MemoryRegion region, const xla::ShapeProto& shape,
      absl::Span<Event* const> wait_for) override {
    auto tpu_events = MakeEventArray(wait_for);

    ::TpuAllocationShape shape_ = GetTpuAllocationShape(shape);
    auto bh = absl::make_unique<DirectBufferHandle>(
        &driver_fn_,
        driver_fn_.TpuDriver_AllocateShape(driver_, core_id, region, shape_,
                                           wait_for.size(), tpu_events));

    free(shape_.bytes);
    delete[] tpu_events;
    return bh;
  }

  std::unique_ptr<BufferHandle> AllocateTuple(
      int32_t core_id, MemoryRegion region,
      absl::Span<BufferHandle* const> children,
      absl::Span<Event* const> wait_for) override {
    auto tpu_events = MakeEventArray(wait_for);

    ::TpuBufferHandle** childbuf = new ::TpuBufferHandle*[children.size()];
    for (int i = 0; i < children.size(); i++) {
      childbuf[i] =
          static_cast<DirectBufferHandle* const>(children[i])->handle_;
    }

    auto bh = absl::make_unique<DirectBufferHandle>(
        &driver_fn_, driver_fn_.TpuDriver_AllocateTuple(
                         driver_, core_id, region, children.size(), childbuf,
                         wait_for.size(), tpu_events));
    delete[] tpu_events;
    delete[] childbuf;

    return bh;
  }

  std::shared_ptr<Event> Deallocate(
      std::unique_ptr<BufferHandle> handle,
      absl::Span<Event* const> wait_for) override {
    auto tpu_events = MakeEventArray(wait_for);
    auto* direct_bh = static_cast<DirectBufferHandle*>(handle.get());
    auto event = std::make_shared<DirectEvent>(
        &driver_fn_,
        driver_fn_.TpuDriver_Deallocate(driver_, direct_bh->handle_,
                                        wait_for.size(), tpu_events));
    delete[] tpu_events;
    return event;
  }

  std::shared_ptr<Event> TransferToDevice(
      const void* src, BufferHandle* dst,
      absl::Span<Event* const> wait_for) override {
    auto tpu_events = MakeEventArray(wait_for);
    auto event = std::make_shared<DirectEvent>(
        &driver_fn_,
        driver_fn_.TpuDriver_TransferToDevice(
            driver_, src, static_cast<DirectBufferHandle*>(dst)->handle_,
            wait_for.size(), tpu_events));
    delete[] tpu_events;
    return event;
  }

  std::shared_ptr<Event> TransferFromDevice(
      const BufferHandle* src, void* dst,
      absl::Span<Event* const> wait_for) override {
    auto tpu_events = MakeEventArray(wait_for);
    auto event = std::make_shared<DirectEvent>(
        &driver_fn_,
        driver_fn_.TpuDriver_TransferFromDevice(
            driver_, static_cast<const DirectBufferHandle*>(src)->handle_, dst,
            wait_for.size(), tpu_events));
    delete[] tpu_events;
    return event;
  }

  std::shared_ptr<Event> TransferFromDeviceToDevice(
      const BufferHandle* src, BufferHandle* dst,
      absl::Span<Event* const> wait_for) override {
    auto tpu_events = MakeEventArray(wait_for);
    auto event = std::make_shared<DirectEvent>(
        &driver_fn_,
        driver_fn_.TpuDriver_TransferFromDeviceToDevice(
            driver_, static_cast<const DirectBufferHandle*>(src)->handle_,
            static_cast<DirectBufferHandle*>(dst)->handle_, wait_for.size(),
            tpu_events));
    delete[] tpu_events;
    return event;
  }

  std::unique_ptr<CompiledProgramHandle> CompileProgram(
      const xla::HloProto& source, int32_t num_replicas,
      absl::Span<Event* const> wait_for) override {
    auto tpu_events = MakeEventArray(wait_for);

    struct HloProto hlo;
    hlo.size = source.ByteSizeLong();
    hlo.buffer = malloc(hlo.size);
    if (!source.SerializeToArray(hlo.buffer, hlo.size)) {
      LOG(ERROR) << "Unable to serialize HLO to array.";
      return nullptr;
    }

    auto handle = absl::make_unique<DirectCompiledProgramHandle>(
        &driver_fn_,
        driver_fn_.TpuDriver_CompileProgram(driver_, hlo, num_replicas,
                                            wait_for.size(), tpu_events));

    free(hlo.buffer);
    delete[] tpu_events;
    return handle;
  }
  std::unique_ptr<LoadedProgramHandle> LoadProgram(
      int32_t core_id, const CompiledProgramHandle* handle,
      absl::Span<Event* const> wait_for) override {
    auto tpu_events = MakeEventArray(wait_for);

    auto loaded_handle = absl::make_unique<DirectLoadedProgramHandle>(
        &driver_fn_,
        driver_fn_.TpuDriver_LoadProgram(
            driver_, core_id,
            static_cast<const DirectCompiledProgramHandle*>(handle)->handle_,
            wait_for.size(), tpu_events));

    delete[] tpu_events;
    return loaded_handle;
  }

  std::shared_ptr<Event> UnloadProgram(
      std::unique_ptr<LoadedProgramHandle> handle,
      absl::Span<Event* const> wait_for) override {
    auto tpu_events = MakeEventArray(wait_for);
    auto* direct_lph = static_cast<DirectLoadedProgramHandle*>(handle.get());
    auto event = std::make_shared<DirectEvent>(
        &driver_fn_,
        driver_fn_.TpuDriver_UnloadProgram(driver_, direct_lph->handle_,
                                           wait_for.size(), tpu_events));
    delete[] tpu_events;
    return event;
  }

  std::shared_ptr<Event> ExecuteProgram(
      LoadedProgramHandle* program, absl::Span<BufferHandle* const> inputs,
      absl::Span<BufferHandle* const> outputs,
      const xla::DeviceAssignmentProto& device_assignment,
      absl::Span<Event* const> wait_for) override {
    auto tpu_events = MakeEventArray(wait_for);

    std::vector<::TpuBufferHandle*> inputv;
    inputv.reserve(inputs.size());
    for (int i = 0; i < inputs.size(); i++) {
      inputv.push_back(
          static_cast<DirectBufferHandle* const>(inputs[i])->handle_);
    }
    std::vector<::TpuBufferHandle*> outputv;
    outputv.reserve(outputs.size());
    for (int i = 0; i < outputs.size(); i++) {
      outputv.push_back(
          static_cast<DirectBufferHandle* const>(outputs[i])->handle_);
    }

    struct DeviceAssignment da;
    da.size = device_assignment.ByteSizeLong();
    da.bytes = malloc(da.size);
    device_assignment.SerializeToArray(da.bytes, da.size);

    auto event = std::make_shared<DirectEvent>(
        &driver_fn_,
        driver_fn_.TpuDriver_ExecuteProgram(
            driver_, static_cast<DirectLoadedProgramHandle*>(program)->handle_,
            inputs.size(), inputv.data(), outputs.size(), outputv.data(), da,
            wait_for.size(), tpu_events));

    free(da.bytes);
    delete[] tpu_events;
    return event;
  }

  std::unique_ptr<TpuLinearizer> GetLinearizer() override {
    return std::make_unique<DirectTpuLinearizer>(driver_, &driver_fn_);
  }

 private:
  ::TpuDriverFn driver_fn_;
  ::TpuDriver* driver_;

  ::TpuEvent** MakeEventArray(absl::Span<Event* const> wait_for) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSpythonPStpu_driverPSdirect_tpu_driverDTcc mht_21(mht_21_v, 707, "", "./tensorflow/compiler/xla/python/tpu_driver/direct_tpu_driver.cc", "MakeEventArray");

    if (wait_for.empty()) return nullptr;
    ::TpuEvent** ret = new ::TpuEvent*[wait_for.size()];
    for (int i = 0; i < wait_for.size(); i++) {
      ret[i] = static_cast<DirectEvent* const>(wait_for[i])->event_;
    }
    return ret;
  }
};

xla::StatusOr<std::unique_ptr<TpuDriver>> RegisterDirectTpuDriver(
    const TpuDriverConfig& config) {
  std::string shared_lib = config.worker().substr(strlen(kDirectProtocol));
  if (shared_lib == "internal") {
#ifdef TPU_SHARED_LIBRARY_COMPILE_LINK
    return xla::StatusOr<std::unique_ptr<TpuDriver>>(
        absl::make_unique<DirectTpuDriver>());
#else
    LOG(FATAL) << "Request to use compile-time linked TPU library, but did not "
               << "link in appropriate library at compile time.";
#endif
  }
  return xla::StatusOr<std::unique_ptr<TpuDriver>>(
      absl::make_unique<DirectTpuDriver>(shared_lib));
}

REGISTER_TPU_DRIVER(kDirectProtocol, RegisterDirectTpuDriver);

}  // namespace
}  // namespace tpu_driver
