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

#ifndef TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_PLATFORM_H_
#define TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_PLATFORM_H_
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
class MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTh {
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
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTh() {
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


#include <memory>

#include "absl/container/flat_hash_map.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/executor_cache.h"
#include "tensorflow/stream_executor/platform.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"
#include "tensorflow/stream_executor/tpu/tpu_executor_c_api.h"
#include "tensorflow/stream_executor/tpu/tpu_platform_interface.h"

namespace tensorflow {
namespace tpu {

class TpuPlatform : public ::tensorflow::tpu::TpuPlatformInterface {
 public:
  using StreamMap =
      absl::flat_hash_map<stream_executor::internal::StreamInterface*,
                          SE_Stream*>;
  using EventMap =
      absl::flat_hash_map<stream_executor::internal::EventInterface*,
                          SE_Event*>;

  static const ::stream_executor::Platform::Id kId;

  using Status = ::stream_executor::port::Status;
  template <typename T>
  using StatusOr = ::stream_executor::port::StatusOr<T>;

  TpuPlatform();

  ~TpuPlatform() override;

  static TpuPlatform* GetRegisteredPlatform();

  Id id() const override;

  const std::string& Name() const override;

  int VisibleDeviceCount() const override;

  int64_t TpuMemoryLimit() override;

  bool ShouldRegisterTpuDeviceToDeviceCopy() override;

  const tensorflow::tpu::TpuTopologyPtr GetTopologyPtr() override;

  const tensorflow::tpu::TpuHostLocationExternal GetTpuHostLocation()
      const override;

  TpuRuntimeVersion version() const override;

  bool Initialized() const override;

  Status Initialize(
      const std::map<std::string, std::string>& platform_options) override;

  Status Reset(bool only_tear_down, absl::string_view reason) override {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("reason: \"" + std::string(reason.data(), reason.size()) + "\"");
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTh mht_0(mht_0_v, 246, "", "./tensorflow/stream_executor/tpu/tpu_platform.h", "Reset");

    LOG(FATAL) << "Not yet implemented";
  }

  StatusOr<std::unique_ptr<::stream_executor::DeviceDescription>>
  DescriptionForDevice(int ordinal) const override {
    LOG(FATAL) << "Not yet implemented";
  }

  StatusOr<::stream_executor::StreamExecutor*> ExecutorForDevice(
      int ordinal) override {
    stream_executor::StreamExecutorConfig config;
    config.ordinal = ordinal;
    return GetExecutor(config);
  }

  StatusOr<::stream_executor::StreamExecutor*>
  ExecutorForDeviceWithPluginConfig(
      int ordinal,
      const ::stream_executor::PluginConfig& plugin_config) override {
    stream_executor::StreamExecutorConfig config;
    config.ordinal = ordinal;
    config.plugin_config = plugin_config;
    return GetExecutor(config);
  }

  StatusOr<::stream_executor::StreamExecutor*> GetExecutor(
      const ::stream_executor::StreamExecutorConfig& config) override;

  StatusOr<std::unique_ptr<::stream_executor::StreamExecutor>>
  GetUncachedExecutor(
      const ::stream_executor::StreamExecutorConfig& config) override;

  void RegisterTraceListener(
      std::unique_ptr<stream_executor::TraceListener> listener) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTh mht_1(mht_1_v, 283, "", "./tensorflow/stream_executor/tpu/tpu_platform.h", "RegisterTraceListener");

    LOG(FATAL) << "Not yet implemented";
  }

  void UnregisterTraceListener(
      stream_executor::TraceListener* listener) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTh mht_2(mht_2_v, 291, "", "./tensorflow/stream_executor/tpu/tpu_platform.h", "UnregisterTraceListener");

    LOG(FATAL) << "Not yet implemented";
  }

  StreamMap* stream_map() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTh mht_3(mht_3_v, 298, "", "./tensorflow/stream_executor/tpu/tpu_platform.h", "stream_map");
 return &stream_map_; }

  void InsertEvent(stream_executor::internal::EventInterface* key,
                   SE_Event* val);
  SE_Event* LookupEvent(stream_executor::internal::EventInterface* key);
  SE_Stream* LookupStream(stream_executor::internal::StreamInterface* key) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTh mht_4(mht_4_v, 306, "", "./tensorflow/stream_executor/tpu/tpu_platform.h", "LookupStream");

    mutex().lock();
    auto stream = stream_map_.at(key);
    mutex().unlock();
    return stream;
  }
  void EraseEvent(stream_executor::internal::EventInterface* key);

  SE_Platform* se_platform() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTh mht_5(mht_5_v, 317, "", "./tensorflow/stream_executor/tpu/tpu_platform.h", "se_platform");
 return platform_; }

  // Returns the number of TPUs per host.
  static Status TpusPerHost(int* tpus);

  // Returns the memory capacity of the TPUs on this host.
  static Status TpuMemoryLimit(int64_t* memory_limit);

  tensorflow::mutex& mutex() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTh mht_6(mht_6_v, 328, "", "./tensorflow/stream_executor/tpu/tpu_platform.h", "mutex");
 return event_map_mu_; }

 private:
  mutable SE_Platform* platform_;
  std::string name_;
  stream_executor::ExecutorCache executor_cache_;
  StreamMap stream_map_;
  EventMap event_map_;
  tensorflow::mutex event_map_mu_;
};

bool RegisterTpuPlatform();

}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_STREAM_EXECUTOR_TPU_TPU_PLATFORM_H_
