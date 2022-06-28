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
class MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_plugin_initDTcc {
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
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_plugin_initDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_plugin_initDTcc() {
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

#include "tensorflow/c/experimental/grappler/grappler_internal.h"
#include "tensorflow/c/experimental/pluggable_profiler/pluggable_profiler_internal.h"
#include "tensorflow/c/experimental/stream_executor/stream_executor_internal.h"
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.h"
#include "tensorflow/core/common_runtime/pluggable_device/pluggable_device_util.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {

static Status InitDeviceAndGraphModule(void* dso_handle) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_plugin_initDTcc mht_0(mht_0_v, 199, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device_plugin_init.cc", "InitDeviceAndGraphModule");

  void* dso_symbol_se;
  void* dso_symbol_graph;
  tensorflow::Env* env = tensorflow::Env::Default();

  Status status_se =
      env->GetSymbolFromLibrary(dso_handle, "SE_InitPlugin", &dso_symbol_se);
  Status status_graph =
      env->GetSymbolFromLibrary(dso_handle, "TF_InitGraph", &dso_symbol_graph);

  // Raise error if neither device nor graph is found.
  if (errors::IsNotFound(status_se) && errors::IsNotFound(status_graph)) {
    return errors::NotFound(status_se.error_message() + " " +
                            status_graph.error_message());
  }

  if (status_se == Status::OK()) {
    auto init_fn =
        reinterpret_cast<stream_executor::SEInitPluginFn>(dso_symbol_se);

    string device_type, platform_name;
    TF_RETURN_IF_ERROR(stream_executor::InitStreamExecutorPlugin(
        init_fn, &device_type, &platform_name));

    DeviceFactory::Register(
        device_type,
        std::make_unique<PluggableDeviceFactory>(device_type, platform_name),
        /*priority=*/220, /*is_pluggable_device=*/true);

    TF_RETURN_IF_ERROR(CopyTensor::Register(
        DeviceType(device_type), DeviceType(device_type),
        PluggableDeviceUtil::DeviceToDeviceCopy,
        /*is_pluggable_device=*/true));  // Register the Copy tensor.
  }

  if (status_graph == Status::OK()) {
    auto init_fn =
        reinterpret_cast<grappler::TFInitGraphPluginFn>(dso_symbol_graph);
    TF_RETURN_IF_ERROR(grappler::InitGraphPlugin(init_fn));
  }

  return Status::OK();
}

typedef void (*TFKernelInitFn)();
static Status InitKernelModule(void* dso_handle) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_plugin_initDTcc mht_1(mht_1_v, 247, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device_plugin_init.cc", "InitKernelModule");

  void* dso_symbol;
  tensorflow::Env* env = tensorflow::Env::Default();

  TF_RETURN_IF_ERROR(
      env->GetSymbolFromLibrary(dso_handle, "TF_InitKernel", &dso_symbol));
  auto init_fn = reinterpret_cast<TFKernelInitFn>(dso_symbol);
  init_fn();
  return Status::OK();
}

static Status InitProfilerModule(void* dso_handle) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_plugin_initDTcc mht_2(mht_2_v, 261, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device_plugin_init.cc", "InitProfilerModule");

  void* dso_symbol;
  tensorflow::Env* env = tensorflow::Env::Default();

  TF_RETURN_IF_ERROR(
      env->GetSymbolFromLibrary(dso_handle, "TF_InitProfiler", &dso_symbol));
  auto init_fn = reinterpret_cast<profiler::TFInitProfilerFn>(dso_symbol);
  TF_RETURN_IF_ERROR(profiler::InitPluginProfiler(init_fn));
  return Status::OK();
}

Status RegisterPluggableDevicePlugin(void* dso_handle) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePScommon_runtimePSpluggable_devicePSpluggable_device_plugin_initDTcc mht_3(mht_3_v, 275, "", "./tensorflow/core/common_runtime/pluggable_device/pluggable_device_plugin_init.cc", "RegisterPluggableDevicePlugin");

  // Step 1 Init Device/Graph Module.
  TF_RETURN_IF_ERROR(InitDeviceAndGraphModule(dso_handle));

  // Step 2 Init Kernel Module.
  TF_RETURN_IF_ERROR(InitKernelModule(dso_handle));

  // Step 3 Init Profiler Module. (Profiler support is optional.)
  Status status = InitProfilerModule(dso_handle);
  if (!status.ok()) {
    VLOG(1) << "Failed to load pluggable profiler module due to "
            << status.error_message();
  }
  return Status::OK();
}

}  // namespace tensorflow
