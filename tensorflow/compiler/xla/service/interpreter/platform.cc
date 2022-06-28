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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSplatformDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSplatformDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSplatformDTcc() {
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

#include "tensorflow/compiler/xla/service/interpreter/platform.h"

#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/service/interpreter/executor.h"
#include "tensorflow/stream_executor/device_options.h"
#include "tensorflow/stream_executor/lib/initialize.h"
#include "tensorflow/stream_executor/lib/status.h"
#include "tensorflow/stream_executor/lib/status_macros.h"
#include "tensorflow/stream_executor/multi_platform_manager.h"
#include "tensorflow/stream_executor/platform.h"

namespace stream_executor {
namespace interpreter {

XlaInterpreterPlatform::XlaInterpreterPlatform(const std::string& name,
                                               const Platform::Id& id)
    : name_(name), id_(id) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSplatformDTcc mht_0(mht_0_v, 205, "", "./tensorflow/compiler/xla/service/interpreter/platform.cc", "XlaInterpreterPlatform::XlaInterpreterPlatform");
}

XlaInterpreterPlatform::~XlaInterpreterPlatform() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSplatformDTcc mht_1(mht_1_v, 210, "", "./tensorflow/compiler/xla/service/interpreter/platform.cc", "XlaInterpreterPlatform::~XlaInterpreterPlatform");
}

Platform::Id XlaInterpreterPlatform::id() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSplatformDTcc mht_2(mht_2_v, 215, "", "./tensorflow/compiler/xla/service/interpreter/platform.cc", "XlaInterpreterPlatform::id");
 return id_; }

int XlaInterpreterPlatform::VisibleDeviceCount() const {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSplatformDTcc mht_3(mht_3_v, 220, "", "./tensorflow/compiler/xla/service/interpreter/platform.cc", "XlaInterpreterPlatform::VisibleDeviceCount");
 return 1; }

const std::string& XlaInterpreterPlatform::Name() const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSplatformDTcc mht_4(mht_4_v, 225, "", "./tensorflow/compiler/xla/service/interpreter/platform.cc", "XlaInterpreterPlatform::Name");
 return name_; }

port::StatusOr<std::unique_ptr<DeviceDescription>>
XlaInterpreterPlatform::DescriptionForDevice(int ordinal) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSplatformDTcc mht_5(mht_5_v, 231, "", "./tensorflow/compiler/xla/service/interpreter/platform.cc", "XlaInterpreterPlatform::DescriptionForDevice");

  return XlaInterpreterExecutor::CreateDeviceDescription(ordinal);
}

port::StatusOr<StreamExecutor*> XlaInterpreterPlatform::ExecutorForDevice(
    int ordinal) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSplatformDTcc mht_6(mht_6_v, 239, "", "./tensorflow/compiler/xla/service/interpreter/platform.cc", "XlaInterpreterPlatform::ExecutorForDevice");

  StreamExecutorConfig config;
  config.ordinal = ordinal;
  config.plugin_config = PluginConfig();
  config.device_options = DeviceOptions::Default();
  return GetExecutor(config);
}

port::StatusOr<StreamExecutor*>
XlaInterpreterPlatform::ExecutorForDeviceWithPluginConfig(
    int device_ordinal, const PluginConfig& plugin_config) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSplatformDTcc mht_7(mht_7_v, 252, "", "./tensorflow/compiler/xla/service/interpreter/platform.cc", "XlaInterpreterPlatform::ExecutorForDeviceWithPluginConfig");

  StreamExecutorConfig config;
  config.ordinal = device_ordinal;
  config.plugin_config = plugin_config;
  config.device_options = DeviceOptions::Default();
  return GetExecutor(config);
}

port::StatusOr<StreamExecutor*> XlaInterpreterPlatform::GetExecutor(
    const StreamExecutorConfig& config) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSplatformDTcc mht_8(mht_8_v, 264, "", "./tensorflow/compiler/xla/service/interpreter/platform.cc", "XlaInterpreterPlatform::GetExecutor");

  return executor_cache_.GetOrCreate(
      config, [&]() { return GetUncachedExecutor(config); });
}

port::StatusOr<std::unique_ptr<StreamExecutor>>
XlaInterpreterPlatform::GetUncachedExecutor(
    const StreamExecutorConfig& config) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSplatformDTcc mht_9(mht_9_v, 274, "", "./tensorflow/compiler/xla/service/interpreter/platform.cc", "XlaInterpreterPlatform::GetUncachedExecutor");

  auto executor = absl::make_unique<StreamExecutor>(
      this, absl::make_unique<XlaInterpreterExecutor>(config.plugin_config),
      config.ordinal);
  auto init_status = executor->Init(config.device_options);
  if (!init_status.ok()) {
    return port::Status{
        port::error::INTERNAL,
        absl::StrFormat(
            "failed initializing StreamExecutor for device ordinal %d: %s",
            config.ordinal, init_status.ToString())};
  }

  return std::move(executor);
}

void XlaInterpreterPlatform::RegisterTraceListener(
    std::unique_ptr<TraceListener> listener) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSplatformDTcc mht_10(mht_10_v, 294, "", "./tensorflow/compiler/xla/service/interpreter/platform.cc", "XlaInterpreterPlatform::RegisterTraceListener");

  LOG(FATAL) << "not yet implemented: register executor trace listener";
}

void XlaInterpreterPlatform::UnregisterTraceListener(TraceListener* listener) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSplatformDTcc mht_11(mht_11_v, 301, "", "./tensorflow/compiler/xla/service/interpreter/platform.cc", "XlaInterpreterPlatform::UnregisterTraceListener");

  LOG(FATAL) << "not yet implemented: unregister executor trace listener";
}

static void InitializeXlaInterpreterPlatform() {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSinterpreterPSplatformDTcc mht_12(mht_12_v, 308, "", "./tensorflow/compiler/xla/service/interpreter/platform.cc", "InitializeXlaInterpreterPlatform");

  std::unique_ptr<Platform> platform(new XlaInterpreterPlatform);
  SE_CHECK_OK(MultiPlatformManager::RegisterPlatform(std::move(platform)));
}

}  // namespace interpreter
}  // namespace stream_executor

REGISTER_MODULE_INITIALIZER(
    interpreter_platform,
    stream_executor::interpreter::InitializeXlaInterpreterPlatform());

// Note that module initialization sequencing is not supported in the
// open-source project, so this will be a no-op there.
REGISTER_MODULE_INITIALIZER_SEQUENCE(interpreter_platform,
                                     multi_platform_manager);
REGISTER_MODULE_INITIALIZER_SEQUENCE(multi_platform_manager_listener,
                                     interpreter_platform);
