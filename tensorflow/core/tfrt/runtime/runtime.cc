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
class MHTracer_DTPStensorflowPScorePStfrtPSruntimePSruntimeDTcc {
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
   MHTracer_DTPStensorflowPScorePStfrtPSruntimePSruntimeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStfrtPSruntimePSruntimeDTcc() {
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
#include "tensorflow/core/tfrt/runtime/runtime.h"

#include <string>
#include <utility>

#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_op_handler.h"
#include "tensorflow/core/runtime_fallback/kernel/kernel_fallback_tensor.h"
#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_op_handler.h"
#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_tensor.h"
#include "tfrt/cpu/core_runtime/cpu_op_handler.h"  // from @tf_runtime
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime
#include "tfrt/tensor/scalar_host_tensor.h"  // from @tf_runtime

#ifdef GOOGLE_CUDA
#include "tfrt/gpu/core_runtime/gpu_op_handler.h"  // from @tf_runtime
#include "tfrt/gpu/device/device.h"  // from @tf_runtime
#include "tfrt/gpu/device/device_util.h"  // from @tf_runtime
#include "tfrt/gpu/tensor/dense_gpu_tensor.h"  // from @tf_runtime
#endif  // GOOGLE_CUDA

constexpr char const kDefaultHostDeviceName[] =
    "/job:localhost/replica:0/task:0/device:CPU:0";

namespace tensorflow {
namespace tfrt_stub {
namespace {

tensorflow::Status InitializeOpHandlers(tfrt::CoreRuntime* corert) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStfrtPSruntimePSruntimeDTcc mht_0(mht_0_v, 215, "", "./tensorflow/core/tfrt/runtime/runtime.cc", "InitializeOpHandlers");

  // TODO(b/196962112): Make default device configurable through Runtime.
  std::string default_device = kDefaultHostDeviceName;

  DeviceNameUtils::ParsedName device_parsed_name;
  if (!DeviceNameUtils::ParseFullName(default_device, &device_parsed_name) ||
      !device_parsed_name.has_type) {
    return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                              "Invalid device name");
  }

  if (device_parsed_name.type == DEVICE_CPU) {
    default_device = kDefaultHostDeviceName;
  } else if (device_parsed_name.type == DEVICE_GPU &&
             (!device_parsed_name.has_job || !device_parsed_name.has_id ||
              !device_parsed_name.has_replica ||
              !device_parsed_name.has_task)) {
    return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                              "Device name must be fully specified");
  }

  tfrt::OpHandler* op_handler = nullptr;

  if (device_parsed_name.type == DEVICE_GPU) {
#ifdef GOOGLE_CUDA
    auto fallback_op_handler = tensorflow::tfd::CreateRuntimeFallbackOpHandler(
        corert, /*tf_device_name=*/"");
    corert->RegisterOpHandler("tf", fallback_op_handler.get());
    op_handler = fallback_op_handler.get();
#endif  // GOOGLE_CUDA
  } else {
    auto fallback_op_handler = tensorflow::tfd::CreateKernelFallbackOpHandler(
        corert, corert->GetHostContext()->GetHostDeviceRef());
    corert->RegisterOpHandler("tfkernel", fallback_op_handler.get());
    op_handler = fallback_op_handler.get();
  }

  if (device_parsed_name.type == DEVICE_CPU) {
    auto cpu_device = corert->GetHostContext()->GetHostDeviceRef();
    auto cpu_op_handler =
        tfrt::CreateCpuOpHandler(corert, std::move(cpu_device), op_handler);

    cpu_op_handler.get()->AddImplicitConversion(
        tensorflow::tfd::RuntimeFallbackTensor::kTensorType,
        tfrt::DenseHostTensor::kTensorType);
    cpu_op_handler.get()->AddImplicitConversion(
        tensorflow::tfd::RuntimeFallbackTensor::kTensorType,
        tfrt::AnyScalarHostTensor::kTensorType);
    cpu_op_handler.get()->AddImplicitConversion(
        tensorflow::tfd::RuntimeFallbackTensor::kTensorType,
        tfrt::StringHostTensor::kTensorType);
    cpu_op_handler.get()->AddImplicitConversion(
        tensorflow::KernelFallbackTensor::kTensorType,
        tfrt::DenseHostTensor::kTensorType);
    cpu_op_handler.get()->AddImplicitConversion(
        tensorflow::KernelFallbackTensor::kTensorType,
        tfrt::AnyScalarHostTensor::kTensorType);
    cpu_op_handler.get()->AddImplicitConversion(
        tensorflow::KernelFallbackTensor::kTensorType,
        tfrt::StringHostTensor::kTensorType);

    op_handler = cpu_op_handler.get();
#ifdef GOOGLE_CUDA
  } else if (device_parsed_name.type == DEVICE_GPU) {
    const int gpu_ordinal = 0;
    auto gpu_device = tfrt::gpu::GetOrCreateGpuDevice(
        default_device, gpu_ordinal, corert->GetHostContext());
    auto gpu_op_handler = tfrt::gpu::CreateGpuOpHandler(
        corert, std::move(gpu_device.get()), op_handler);
    op_handler = gpu_op_handler.get();
#endif  // GOOGLE_CUDA
  } else {
    return tensorflow::Status(tensorflow::error::INVALID_ARGUMENT,
                              "Unknown device type");
  }

  corert->RegisterOpHandler(default_device, op_handler);

  return tensorflow::Status::OK();
}

}  // namespace

std::unique_ptr<Runtime> Runtime::Create(
    std::unique_ptr<WorkQueueInterface> work_queue) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStfrtPSruntimePSruntimeDTcc mht_1(mht_1_v, 302, "", "./tensorflow/core/tfrt/runtime/runtime.cc", "Runtime::Create");

  auto* work_queue_ptr = work_queue.get();
  auto expected_core_runtime = tfrt::CoreRuntime::Create(
      [](const tfrt::DecodedDiagnostic& diag) { LOG(ERROR) << diag.message; },
      tfrt::CreateMallocAllocator(), std::move(work_queue),
      kDefaultHostDeviceName);
  DCHECK(expected_core_runtime);
  const auto& status = InitializeOpHandlers(expected_core_runtime.get().get());
  if (!status.ok()) {
    LOG(ERROR) << "Failed to initialize op handlers: " << status;
    return {};
  }

  // We don't use std::make_unique here because the constructor should better be
  // private.
  return std::unique_ptr<Runtime>(
      new Runtime(std::move(expected_core_runtime.get()), work_queue_ptr));
}

// TODO(b/196962112): Remove this overload.
std::unique_ptr<Runtime> Runtime::Create() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStfrtPSruntimePSruntimeDTcc mht_2(mht_2_v, 325, "", "./tensorflow/core/tfrt/runtime/runtime.cc", "Runtime::Create");

  static constexpr int kDefaultNumInterOpThreads = 4;
  // Let system pick the number of intra op threads.
  static constexpr int kDefaultNumIntraOpThreads = 0;
  return Runtime::Create(kDefaultNumInterOpThreads, kDefaultNumIntraOpThreads);
}

std::unique_ptr<Runtime> Runtime::Create(int num_inter_op_threads,
                                         int num_intra_op_threads) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStfrtPSruntimePSruntimeDTcc mht_3(mht_3_v, 336, "", "./tensorflow/core/tfrt/runtime/runtime.cc", "Runtime::Create");

  if (num_intra_op_threads <= 0)
    num_intra_op_threads = tensorflow::port::MaxParallelism();
  return Runtime::Create(
      WrapDefaultWorkQueue(tfrt::CreateMultiThreadedWorkQueue(
          num_intra_op_threads, num_inter_op_threads)));
}

Runtime::Runtime(std::unique_ptr<tfrt::CoreRuntime> core_runtime,
                 WorkQueueInterface* work_queue)
    : core_runtime_(std::move(core_runtime)), work_queue_(work_queue) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStfrtPSruntimePSruntimeDTcc mht_4(mht_4_v, 349, "", "./tensorflow/core/tfrt/runtime/runtime.cc", "Runtime::Runtime");

  DCHECK(work_queue_);
}

Runtime::~Runtime() = default;

}  // namespace tfrt_stub
}  // namespace tensorflow
