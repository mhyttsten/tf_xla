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
class MHTracer_DTPStensorflowPScorePStfrtPSutilsPSutilsDTcc {
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
   MHTracer_DTPStensorflowPScorePStfrtPSutilsPSutilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStfrtPSutilsPSutilsDTcc() {
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
#include "tensorflow/core/tfrt/utils/utils.h"

#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/device.h"
#include "tensorflow/core/tfrt/eager/virtual_device.h"
#include "tensorflow/core/tpu/virtual_device.h"
#include "tfrt/bef_executor/bef_file.h"  // from @tf_runtime
#include "tfrt/core_runtime/core_runtime.h"  // from @tf_runtime
#include "tfrt/host_context/chain.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/function.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/error_util.h"  // from @tf_runtime

namespace tfrt {

using ::tensorflow::StatusOr;

Expected<const char*> ConvertTfDeviceNameToTfrt(
    const char* device_name, tensorflow::EagerContext* eager_context) {
  // NOTE(fishx): We need to get tf_device first because DeviceMgr in current TF
  // allows us get the device with simplified name like "CPU:0". However, TFRT
  // DeviceManager only allows get device via its fullname.
  tensorflow::Device* tf_device;
  tensorflow::Status s =
      eager_context->FindDeviceFromName(device_name, &tf_device);
  if (!s.ok()) {
    return MakeStringError(s.error_message());
  }
  return tf_device->name().c_str();
}

DType ConvertTfDTypeToTfrtDType(tensorflow::DataType dtype) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStfrtPSutilsPSutilsDTcc mht_0(mht_0_v, 217, "", "./tensorflow/core/tfrt/utils/utils.cc", "ConvertTfDTypeToTfrtDType");

  switch (dtype) {
#define DTYPE(TFRT_DTYPE, TF_DTYPE) \
  case tensorflow::TF_DTYPE:        \
    return DType(DType::TFRT_DTYPE);
#include "tensorflow/core/tfrt/utils/dtype.def"  // NOLINT
    default:
      return DType();
  }
}

tensorflow::Status RunRuntimeInitializer(const tfrt::ExecutionContext& exec_ctx,
                                         tfrt::BEFFile* bef_file,
                                         absl::string_view fallback_init_func) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("fallback_init_func: \"" + std::string(fallback_init_func.data(), fallback_init_func.size()) + "\"");
   MHTracer_DTPStensorflowPScorePStfrtPSutilsPSutilsDTcc mht_1(mht_1_v, 234, "", "./tensorflow/core/tfrt/utils/utils.cc", "RunRuntimeInitializer");

  auto* host = exec_ctx.host();

  auto* func = bef_file->GetFunction(
      {fallback_init_func.data(), fallback_init_func.size()});
  if (func == nullptr) return tensorflow::Status::OK();

  auto ready_chain = GetReadyChain();

  DCHECK_EQ(func->argument_types().size(), 1);

  llvm::SmallVector<RCReference<AsyncValue>, 1> results;
  results.resize(func->result_types().size());
  DCHECK_EQ(results.size(), 1);

  func->Execute(exec_ctx, ready_chain.GetAsyncValue(), results);

  host->Await(results);

  if (auto* error = results[0]->GetErrorIfPresent()) {
    return tensorflow::errors::Internal(error->message);
  }

  return tensorflow::Status::OK();
}

void CreateDummyTfDevices(
    const std::vector<std::string>& device_names,
    std::vector<std::unique_ptr<tensorflow::Device>>* dummy_tf_devices) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStfrtPSutilsPSutilsDTcc mht_2(mht_2_v, 265, "", "./tensorflow/core/tfrt/utils/utils.cc", "CreateDummyTfDevices");

  for (const auto& name : device_names) {
    tensorflow::DeviceAttributes device_attrs =
        tensorflow::Device::BuildDeviceAttributes(
            name, tensorflow::DEVICE_TPU_SYSTEM, tensorflow::Bytes(16ULL << 30),
            tensorflow::DeviceLocality(), "device: TFRT TPU SYSTEM device");
    dummy_tf_devices->push_back(std::make_unique<tensorflow::VirtualDevice>(
        tensorflow::Env::Default(), device_attrs));
  }
}

void AddDummyTfrtDevices(const std::vector<std::string>& device_names,
                         HostContext* host_ctx) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStfrtPSutilsPSutilsDTcc mht_3(mht_3_v, 280, "", "./tensorflow/core/tfrt/utils/utils.cc", "AddDummyTfrtDevices");

  for (const auto& name : device_names) {
    host_ctx->GetDeviceManager()->MaybeAddDevice(
        TakeRef(new tfrt::VirtualDevice(name)));
  }
}

StatusOr<RCReference<tfrt::BEFFile>> CreateBefFileFromBefBuffer(
    const tensorflow::tfrt_stub::Runtime& runtime, const tfrt::BefBuffer& bef) {
  auto* core_runtime = runtime.core_runtime();
  DCHECK(core_runtime);
  auto* host_context = core_runtime->GetHostContext();
  DCHECK(host_context);
  auto bef_file =
      BEFFile::Open(bef, host_context->GetKernelRegistry(),
                    host_context->diag_handler(), host_context->allocator());
  TF_RET_CHECK(bef_file) << "failed to open BEF";
  return bef_file;
}

int64_t GetUniqueInt() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStfrtPSutilsPSutilsDTcc mht_4(mht_4_v, 303, "", "./tensorflow/core/tfrt/utils/utils.cc", "GetUniqueInt");

  static std::atomic<int64_t> id(0);
  return id.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace tfrt
