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
class MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTcc {
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
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTcc() {
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

#include "tensorflow/stream_executor/tpu/tpu_platform.h"

#include "tensorflow/c/tf_status.h"
#include "tensorflow/c/tf_status_helper.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/stream_executor/tpu/status_helper.h"
#include "tensorflow/stream_executor/tpu/tpu_executor.h"
#include "tensorflow/stream_executor/tpu/tpu_platform_id.h"

namespace tensorflow {
namespace tpu {

const ::stream_executor::Platform::Id TpuPlatform::kId = GetTpuPlatformId();
TpuPlatform* tpu_registered_platform = nullptr;

using Status = ::stream_executor::port::Status;
template <typename T>
using StatusOr = ::stream_executor::port::StatusOr<T>;

TpuPlatform::TpuPlatform() : name_("TPU") {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTcc mht_0(mht_0_v, 204, "", "./tensorflow/stream_executor/tpu/tpu_platform.cc", "TpuPlatform::TpuPlatform");

  platform_ = tpu::ExecutorApiFn()->TpuPlatform_NewFn();
  CHECK(platform_ != nullptr);
}

TpuPlatform* TpuPlatform::GetRegisteredPlatform() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTcc mht_1(mht_1_v, 212, "", "./tensorflow/stream_executor/tpu/tpu_platform.cc", "TpuPlatform::GetRegisteredPlatform");

  return tpu_registered_platform;
}

Status TpuPlatform::Initialize(
    const std::map<std::string, std::string>& platform_options) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTcc mht_2(mht_2_v, 220, "", "./tensorflow/stream_executor/tpu/tpu_platform.cc", "TpuPlatform::Initialize");

  StatusHelper status;

  size_t options_size = platform_options.size();
  const char** options_key =
      static_cast<const char**>(malloc(sizeof(const char*) * options_size));
  const char** options_value =
      static_cast<const char**>(malloc(sizeof(const char*) * options_size));

  size_t i = 0;
  for (const auto& option : platform_options) {
    options_key[i] = option.first.c_str();
    options_value[i] = option.second.c_str();
    i++;
  }

  tpu::ExecutorApiFn()->TpuPlatform_InitializeFn(
      platform_, options_size, options_key, options_value, status.c_status);

  free(options_key);
  free(options_value);

  return status.status();
}

bool TpuPlatform::Initialized() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTcc mht_3(mht_3_v, 248, "", "./tensorflow/stream_executor/tpu/tpu_platform.cc", "TpuPlatform::Initialized");

  return tpu::ExecutorApiFn()->TpuPlatform_InitializedFn(platform_);
}

TpuPlatform::~TpuPlatform() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTcc mht_4(mht_4_v, 255, "", "./tensorflow/stream_executor/tpu/tpu_platform.cc", "TpuPlatform::~TpuPlatform");

  tpu::ExecutorApiFn()->TpuPlatform_FreeFn(platform_);
}

int TpuPlatform::VisibleDeviceCount() const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTcc mht_5(mht_5_v, 262, "", "./tensorflow/stream_executor/tpu/tpu_platform.cc", "TpuPlatform::VisibleDeviceCount");

  return tpu::ExecutorApiFn()->TpuPlatform_VisibleDeviceCountFn(platform_);
}

StatusOr<::stream_executor::StreamExecutor*> TpuPlatform::GetExecutor(
    const ::stream_executor::StreamExecutorConfig& config) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTcc mht_6(mht_6_v, 270, "", "./tensorflow/stream_executor/tpu/tpu_platform.cc", "TpuPlatform::GetExecutor");

  return executor_cache_.GetOrCreate(
      config, [&]() { return GetUncachedExecutor(config); });
}

StatusOr<std::unique_ptr<::stream_executor::StreamExecutor>>
TpuPlatform::GetUncachedExecutor(
    const ::stream_executor::StreamExecutorConfig& config) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTcc mht_7(mht_7_v, 280, "", "./tensorflow/stream_executor/tpu/tpu_platform.cc", "TpuPlatform::GetUncachedExecutor");

  SE_StreamExecutorConfig* c_config =
      tpu::ExecutorApiFn()->TpuStreamExecutorConfig_DefaultFn();

  tpu::ExecutorApiFn()->TpuStreamExecutorConfig_SetOrdinalFn(c_config,
                                                             config.ordinal);

  StatusHelper status;
  SE_StreamExecutor* executor = tpu::ExecutorApiFn()->TpuPlatform_GetExecutorFn(
      platform_, c_config, status.c_status);
  tpu::ExecutorApiFn()->TpuStreamExecutorConfig_FreeFn(c_config);
  if (!status.ok()) {
    return status.status();
  }
  return std::make_unique<stream_executor::StreamExecutor>(
      this, std::make_unique<TpuExecutor>(this, executor), config.ordinal);
}

::stream_executor::Platform::Id TpuPlatform::id() const {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTcc mht_8(mht_8_v, 301, "", "./tensorflow/stream_executor/tpu/tpu_platform.cc", "TpuPlatform::id");

  return TpuPlatform::kId;
}

const std::string& TpuPlatform::Name() const {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTcc mht_9(mht_9_v, 308, "", "./tensorflow/stream_executor/tpu/tpu_platform.cc", "TpuPlatform::Name");
 return name_; }

int64_t TpuPlatform::TpuMemoryLimit() {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTcc mht_10(mht_10_v, 313, "", "./tensorflow/stream_executor/tpu/tpu_platform.cc", "TpuPlatform::TpuMemoryLimit");

  return tpu::ExecutorApiFn()->TpuPlatform_TpuMemoryLimitFn(platform_);
}

bool TpuPlatform::ShouldRegisterTpuDeviceToDeviceCopy() {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTcc mht_11(mht_11_v, 320, "", "./tensorflow/stream_executor/tpu/tpu_platform.cc", "TpuPlatform::ShouldRegisterTpuDeviceToDeviceCopy");

  return tpu::ExecutorApiFn()
      ->TpuPlatform_ShouldRegisterTpuDeviceToDeviceCopyFn(platform_);
}

const tensorflow::tpu::TpuTopologyPtr TpuPlatform::GetTopologyPtr() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTcc mht_12(mht_12_v, 328, "", "./tensorflow/stream_executor/tpu/tpu_platform.cc", "TpuPlatform::GetTopologyPtr");

  return tpu::ExecutorApiFn()->TpuPlatform_GetTopologyPtrFn(platform_);
}

const tensorflow::tpu::TpuHostLocationExternal TpuPlatform::GetTpuHostLocation()
    const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTcc mht_13(mht_13_v, 336, "", "./tensorflow/stream_executor/tpu/tpu_platform.cc", "TpuPlatform::GetTpuHostLocation");

  return tpu::TpuHostLocationExternal(
      tpu::ExecutorApiFn()->TpuPlatform_GetHostLocationFn(platform_));
}

TpuRuntimeVersion TpuPlatform::version() const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTcc mht_14(mht_14_v, 344, "", "./tensorflow/stream_executor/tpu/tpu_platform.cc", "TpuPlatform::version");

  return tpu::ExecutorApiFn()->TpuPlatform_GetRuntimeVersionFn(platform_);
}

void TpuPlatform::InsertEvent(stream_executor::internal::EventInterface* key,
                              SE_Event* val) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTcc mht_15(mht_15_v, 352, "", "./tensorflow/stream_executor/tpu/tpu_platform.cc", "TpuPlatform::InsertEvent");

  tensorflow::mutex_lock lock(event_map_mu_);
  event_map_[key] = val;
}

SE_Event* TpuPlatform::LookupEvent(
    stream_executor::internal::EventInterface* key) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTcc mht_16(mht_16_v, 361, "", "./tensorflow/stream_executor/tpu/tpu_platform.cc", "TpuPlatform::LookupEvent");

  tensorflow::tf_shared_lock lock(event_map_mu_);
  return event_map_.at(key);
}

void TpuPlatform::EraseEvent(stream_executor::internal::EventInterface* key) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTcc mht_17(mht_17_v, 369, "", "./tensorflow/stream_executor/tpu/tpu_platform.cc", "TpuPlatform::EraseEvent");

  tensorflow::mutex_lock lock(event_map_mu_);
  event_map_.erase(key);
}

Status TpuPlatform::TpusPerHost(int* tpus) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTcc mht_18(mht_18_v, 377, "", "./tensorflow/stream_executor/tpu/tpu_platform.cc", "TpuPlatform::TpusPerHost");

  TF_Status* status = TF_NewStatus();

  if (tpu::OpsApiFn()->TpuConfigurationApi_TpusPerHostFn == nullptr) {
    *tpus = 0;
    return Status::OK();
  }

  tpu::OpsApiFn()->TpuConfigurationApi_TpusPerHostFn(tpus, status);
  auto ret_status = StatusFromTF_Status(status);
  TF_DeleteStatus(status);
  return ret_status;
}

Status TpuPlatform::TpuMemoryLimit(int64_t* memory_limit) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTcc mht_19(mht_19_v, 394, "", "./tensorflow/stream_executor/tpu/tpu_platform.cc", "TpuPlatform::TpuMemoryLimit");

  TF_Status* status = TF_NewStatus();

  if (tpu::OpsApiFn()->TpuConfigurationApi_TpuMemoryLimitFn == nullptr) {
    *memory_limit = 0;
    return Status::OK();
  }

  tpu::OpsApiFn()->TpuConfigurationApi_TpuMemoryLimitFn(
      reinterpret_cast<int64_t*>(memory_limit), status);
  auto ret_status = StatusFromTF_Status(status);
  TF_DeleteStatus(status);
  return ret_status;
}

bool RegisterTpuPlatform() {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPSstream_executorPStpuPStpu_platformDTcc mht_20(mht_20_v, 412, "", "./tensorflow/stream_executor/tpu/tpu_platform.cc", "RegisterTpuPlatform");

  // Silently bail if the underlying TPU C API isn't initialized. This is useful
  // for code that unconditionally calls RegisterTpuPlatform() but doesn't link
  // in the underlying TPU library when not running on TPU.
  if (!tpu::IsStreamExecutorEnabled(tpu::ExecutorApiFn())) {
    return true;
  }
  static bool tpu_platform_registered = false;
  if (!tpu_platform_registered) {
    tpu_registered_platform = new TpuPlatform();
    std::unique_ptr<stream_executor::Platform> platform(
        tpu_registered_platform);
    SE_CHECK_OK(stream_executor::MultiPlatformManager::RegisterPlatform(
        std::move(platform)));
    tpu_platform_registered = true;
  }
  return true;
}

}  // namespace tpu
}  // namespace tensorflow
