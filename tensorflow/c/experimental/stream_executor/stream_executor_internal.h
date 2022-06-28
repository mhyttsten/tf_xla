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
// Classes and utilities that work with StreamExecutor C API for internal use.
// This includes functions used for device registration and interfaces needed
// for testing.
#ifndef TENSORFLOW_C_EXPERIMENTAL_STREAM_EXECUTOR_STREAM_EXECUTOR_INTERNAL_H_
#define TENSORFLOW_C_EXPERIMENTAL_STREAM_EXECUTOR_STREAM_EXECUTOR_INTERNAL_H_
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
class MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_internalDTh {
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
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_internalDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_internalDTh() {
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


#include "tensorflow/c/experimental/stream_executor/stream_executor.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/stream_executor/executor_cache.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/platform.h"

namespace stream_executor {

// Plugin initialization function that a device plugin
// must define.
typedef void (*SEInitPluginFn)(SE_PlatformRegistrationParams* const,
                               TF_Status* const);

// Registers StreamExecutor platform. `device_type` and `platform_name` are
// output parameters.
port::Status InitStreamExecutorPlugin(void* dso_handle,
                                      std::string* device_type,
                                      std::string* platform_name);

// Allow registering a StreamExecutor plugin using a function (used for
// testing).
port::Status InitStreamExecutorPlugin(SEInitPluginFn init_fn,
                                      std::string* device_type,
                                      std::string* platform_name);

// This file implements core stream executor base classes in terms of
// the C API defined in stream_executor.h. A class "CSomething" represents a
// "Something" that can be manipulated via calls in the C interface.
class CPlatform : public Platform {
 public:
  explicit CPlatform(SP_Platform platform,
                     void (*destroy_platform)(SP_Platform*),
                     SP_PlatformFns platform_fns,
                     void (*destroy_platform_fns)(SP_PlatformFns*),
                     SP_DeviceFns device_fns, SP_StreamExecutor stream_executor,
                     SP_TimerFns timer_fns);
  ~CPlatform() override;

  Id id() const override {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_internalDTh mht_0(mht_0_v, 228, "", "./tensorflow/c/experimental/stream_executor/stream_executor_internal.h", "id");
 return const_cast<int*>(&plugin_id_value_); }
  const std::string& Name() const override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_internalDTh mht_1(mht_1_v, 232, "", "./tensorflow/c/experimental/stream_executor/stream_executor_internal.h", "Name");
 return name_; }
  int VisibleDeviceCount() const override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_internalDTh mht_2(mht_2_v, 236, "", "./tensorflow/c/experimental/stream_executor/stream_executor_internal.h", "VisibleDeviceCount");

    int visible_device_count = 0;
    tensorflow::TF_StatusPtr c_status(TF_NewStatus());
    platform_fns_.get_device_count(&platform_, &visible_device_count,
                                   c_status.get());
    if (TF_GetCode(c_status.get()) != TF_OK) {
      LOG(ERROR) << TF_Message(c_status.get());
      return 0;
    }
    return visible_device_count;
  }
  bool UseBfcAllocator() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_internalDTh mht_3(mht_3_v, 250, "", "./tensorflow/c/experimental/stream_executor/stream_executor_internal.h", "UseBfcAllocator");
 return platform_.use_bfc_allocator; }
  bool ForceMemoryGrowth() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_internalDTh mht_4(mht_4_v, 254, "", "./tensorflow/c/experimental/stream_executor/stream_executor_internal.h", "ForceMemoryGrowth");
 return platform_.force_memory_growth; }
  port::StatusOr<std::unique_ptr<DeviceDescription>> DescriptionForDevice(
      int ordinal) const override;
  port::StatusOr<StreamExecutor*> ExecutorForDevice(int ordinal) override;
  port::StatusOr<StreamExecutor*> ExecutorForDeviceWithPluginConfig(
      int ordinal, const PluginConfig& plugin_config) override;
  port::StatusOr<StreamExecutor*> GetExecutor(
      const StreamExecutorConfig& config) override;
  port::StatusOr<std::unique_ptr<StreamExecutor>> GetUncachedExecutor(
      const StreamExecutorConfig& config) override;

  // Trace listener is not supported
  void RegisterTraceListener(std::unique_ptr<TraceListener> listener) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_internalDTh mht_5(mht_5_v, 269, "", "./tensorflow/c/experimental/stream_executor/stream_executor_internal.h", "RegisterTraceListener");

    LOG(FATAL) << "RegisterTraceListener is not supported by pluggable device";
  }
  void UnregisterTraceListener(TraceListener* listener) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_internalDTh mht_6(mht_6_v, 275, "", "./tensorflow/c/experimental/stream_executor/stream_executor_internal.h", "UnregisterTraceListener");
}

  void DestroyAllExecutors() {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_internalDTh mht_7(mht_7_v, 280, "", "./tensorflow/c/experimental/stream_executor/stream_executor_internal.h", "DestroyAllExecutors");
 executor_cache_.DestroyAllExecutors(); }

 private:
  SP_Platform platform_;
  void (*destroy_platform_)(SP_Platform*);
  SP_PlatformFns platform_fns_;
  void (*destroy_platform_fns_)(SP_PlatformFns*);
  SP_DeviceFns device_fns_;
  SP_StreamExecutor stream_executor_;
  SP_TimerFns timer_fns_;
  const std::string name_;
  int plugin_id_value_;
  stream_executor::ExecutorCache executor_cache_;
};

class CStream : public internal::StreamInterface {
 public:
  CStream(SP_Device* device, SP_StreamExecutor* stream_executor)
      : device_(device),
        stream_executor_(stream_executor),
        stream_handle_(nullptr) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_internalDTh mht_8(mht_8_v, 303, "", "./tensorflow/c/experimental/stream_executor/stream_executor_internal.h", "CStream");
}
  ~CStream() override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_internalDTh mht_9(mht_9_v, 307, "", "./tensorflow/c/experimental/stream_executor/stream_executor_internal.h", "~CStream");
 Destroy(); }

  port::Status Create() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_internalDTh mht_10(mht_10_v, 312, "", "./tensorflow/c/experimental/stream_executor/stream_executor_internal.h", "Create");

    tensorflow::TF_StatusPtr c_status(TF_NewStatus());
    stream_executor_->create_stream(device_, &stream_handle_, c_status.get());
    port::Status s = tensorflow::StatusFromTF_Status(c_status.get());
    return s;
  }

  void Destroy() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_internalDTh mht_11(mht_11_v, 322, "", "./tensorflow/c/experimental/stream_executor/stream_executor_internal.h", "Destroy");

    if (stream_handle_ != nullptr) {
      stream_executor_->destroy_stream(device_, stream_handle_);
      stream_handle_ = nullptr;
    }
  }

  SP_Stream Handle() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_internalDTh mht_12(mht_12_v, 332, "", "./tensorflow/c/experimental/stream_executor/stream_executor_internal.h", "Handle");
 return stream_handle_; }

 private:
  SP_Device* device_;
  SP_StreamExecutor* stream_executor_;
  SP_Stream stream_handle_;
};

class CEvent : public internal::EventInterface {
 public:
  CEvent(SP_Device* device, SP_StreamExecutor* stream_executor)
      : device_(device),
        stream_executor_(stream_executor),
        event_handle_(nullptr) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_internalDTh mht_13(mht_13_v, 348, "", "./tensorflow/c/experimental/stream_executor/stream_executor_internal.h", "CEvent");
}
  ~CEvent() override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_internalDTh mht_14(mht_14_v, 352, "", "./tensorflow/c/experimental/stream_executor/stream_executor_internal.h", "~CEvent");
 Destroy(); }

  port::Status Create() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_internalDTh mht_15(mht_15_v, 357, "", "./tensorflow/c/experimental/stream_executor/stream_executor_internal.h", "Create");

    tensorflow::TF_StatusPtr c_status(TF_NewStatus());
    stream_executor_->create_event(device_, &event_handle_, c_status.get());
    return tensorflow::StatusFromTF_Status(c_status.get());
  }

  port::Status Record(SP_Stream stream_handle) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_internalDTh mht_16(mht_16_v, 366, "", "./tensorflow/c/experimental/stream_executor/stream_executor_internal.h", "Record");

    tensorflow::TF_StatusPtr c_status(TF_NewStatus());
    stream_executor_->record_event(device_, stream_handle, event_handle_,
                                   c_status.get());
    return tensorflow::StatusFromTF_Status(c_status.get());
  }

  void Destroy() {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_internalDTh mht_17(mht_17_v, 376, "", "./tensorflow/c/experimental/stream_executor/stream_executor_internal.h", "Destroy");

    if (event_handle_ != nullptr) {
      stream_executor_->destroy_event(device_, event_handle_);
      event_handle_ = nullptr;
    }
  }

  SP_Event Handle() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_internalDTh mht_18(mht_18_v, 386, "", "./tensorflow/c/experimental/stream_executor/stream_executor_internal.h", "Handle");
 return event_handle_; }

 private:
  SP_Device* device_;
  SP_StreamExecutor* stream_executor_;
  SP_Event event_handle_;
};

class CTimer : public internal::TimerInterface {
 public:
  CTimer(SP_Device* device, SP_StreamExecutor* stream_executor,
         SP_TimerFns* timer_fns)
      : device_(device),
        stream_executor_(stream_executor),
        timer_handle_(nullptr),
        timer_fns_(timer_fns) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_internalDTh mht_19(mht_19_v, 404, "", "./tensorflow/c/experimental/stream_executor/stream_executor_internal.h", "CTimer");
}
  ~CTimer() override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_internalDTh mht_20(mht_20_v, 408, "", "./tensorflow/c/experimental/stream_executor/stream_executor_internal.h", "~CTimer");
 Destroy(); }

  port::Status Create() {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_internalDTh mht_21(mht_21_v, 413, "", "./tensorflow/c/experimental/stream_executor/stream_executor_internal.h", "Create");

    tensorflow::TF_StatusPtr c_status(TF_NewStatus());
    stream_executor_->create_timer(device_, &timer_handle_, c_status.get());
    return tensorflow::StatusFromTF_Status(c_status.get());
  }

  void Destroy() {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_internalDTh mht_22(mht_22_v, 422, "", "./tensorflow/c/experimental/stream_executor/stream_executor_internal.h", "Destroy");

    if (timer_handle_ != nullptr) {
      stream_executor_->destroy_timer(device_, timer_handle_);
      timer_handle_ = nullptr;
    }
  }

  SP_Timer Handle() {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_internalDTh mht_23(mht_23_v, 432, "", "./tensorflow/c/experimental/stream_executor/stream_executor_internal.h", "Handle");
 return timer_handle_; }

  uint64 Microseconds() const override {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_internalDTh mht_24(mht_24_v, 437, "", "./tensorflow/c/experimental/stream_executor/stream_executor_internal.h", "Microseconds");

    return timer_fns_->nanoseconds(timer_handle_) / 1000;
  }

  uint64 Nanoseconds() const override {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScPSexperimentalPSstream_executorPSstream_executor_internalDTh mht_25(mht_25_v, 444, "", "./tensorflow/c/experimental/stream_executor/stream_executor_internal.h", "Nanoseconds");

    return timer_fns_->nanoseconds(timer_handle_);
  }

 private:
  SP_Device* device_;
  SP_StreamExecutor* stream_executor_;
  SP_Timer timer_handle_;
  SP_TimerFns* timer_fns_;
};

}  // namespace stream_executor
#endif  // TENSORFLOW_C_EXPERIMENTAL_STREAM_EXECUTOR_STREAM_EXECUTOR_INTERNAL_H_
