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
class MHTracer_DTPStensorflowPScorePStfrtPSeagerPSbackendsPSgpuPSgpu_registrationDTcc {
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
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSbackendsPSgpuPSgpu_registrationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStfrtPSeagerPSbackendsPSgpuPSgpu_registrationDTcc() {
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

#include "tensorflow/core/common_runtime/device/device_id.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/gpu/gpu_id_manager.h"
#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_op_handler.h"
#include "tensorflow/core/tfrt/eager/core_runtime/op_handler_registry.h"
#include "tfrt/gpu/core_runtime/gpu_op_handler.h"  // from @tf_runtime
#include "tfrt/gpu/device/device.h"  // from @tf_runtime
#include "tfrt/gpu/device/device_util.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/host_context/resource_context.h"  // from @tf_runtime

namespace tfrt {
namespace tf {
namespace gpu {

using ::tfrt::CoreRuntime;

static void RegisterGpuOpHandler(CoreRuntime* core_runtime,
                                 ResourceContext* resource_context,
                                 const DeviceMgr* device_mgr) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStfrtPSeagerPSbackendsPSgpuPSgpu_registrationDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/tfrt/eager/backends/gpu/gpu_registration.cc", "RegisterGpuOpHandler");

  for (auto& device : device_mgr->ListDevices()) {
    auto& parsed_name = device->parsed_name();
    assert(parsed_name.has_id && parsed_name.has_type);
    if (parsed_name.type == "GPU") {
      // Please see the difference between tf_device_id and platform_device_id
      // here in tensorflow/core/common_runtime/device/device_id.h
      tensorflow::TfDeviceId tf_device_id(parsed_name.id);
      tensorflow::PlatformDeviceId platform_device_id;
      tensorflow::Status s = tensorflow::GpuIdManager::TfToPlatformDeviceId(
          tf_device_id, &platform_device_id);
      if (!s.ok()) {
        LOG(ERROR) << "Failed to convert gpu device [" << device->name()
                   << "] to platform device id due to error: "
                   << s.error_message();
        continue;
      }
      auto gpu = tfrt::gpu::GetOrCreateGpuDevice(
          device->name(), platform_device_id.value(),
          core_runtime->GetHostContext());
      if (!gpu) {
        LOG(ERROR) << "Failed to create gpu device [" << device->name()
                   << "]. Error: " << StrCat(gpu.takeError());
        continue;
      }
      LOG(INFO) << "Found a GPU device: " << device->name();
      auto expected_fallback_op_handler =
          tensorflow::tfd::CreateRuntimeFallbackOpHandler(core_runtime,
                                                          device->name());
      assert(expected_fallback_op_handler);

      auto expected_gpu_op_handler =
          ::tfrt::gpu::CreateGpuOpHandler(core_runtime, std::move(gpu.get()),
                                          expected_fallback_op_handler.get());
      assert(expected_gpu_op_handler);

      core_runtime->RegisterOpHandler(device->name(),
                                      expected_gpu_op_handler.get());

      // TODO(fishx): Remove this when lowering pass can use full device name.
      if (parsed_name.id == 0) {
        core_runtime->RegisterOpHandler("gpu", expected_gpu_op_handler.get());
      }
    }
  }
}

static OpHandlerRegistration register_gpu(RegisterGpuOpHandler);

}  // namespace gpu
}  // namespace tf
}  // namespace tfrt
