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
class MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStf_gpu_runtime_wrappersDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStf_gpu_runtime_wrappersDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStf_gpu_runtime_wrappersDTcc() {
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

#include "tensorflow/compiler/mlir/tools/kernel_gen/tf_gpu_runtime_wrappers.h"

#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

static void ReportInternalError(tensorflow::OpKernelContext *ctx,
                                const std::string msg) {
  if (ctx == nullptr) {
    LOG(WARNING) << msg << "\n";
    return;
  }
  ctx->CtxFailureWithWarning(
      tensorflow::Status{tensorflow::error::INTERNAL, msg});
}

#if GOOGLE_CUDA
using GPUResult = CUresult;
#endif
#if TENSORFLOW_USE_ROCM
using GPUResult = hipError_t;
#endif

void GPUReportIfError(GPUResult result, tensorflow::OpKernelContext *ctx,
                      const char *expr_str) {
  if (!result) return;
  const char *name = nullptr;

#if GOOGLE_CUDA
  cuGetErrorName(result, &name);
#endif
#if TENSORFLOW_USE_ROCM
  name = hipGetErrorName(result);
#endif

  if (!name) name = "<unknown>";
  std::string msg = absl::StrCat("'", expr_str, "' failed with '", name, "'");
  ReportInternalError(ctx, msg);
}

#define GPU_REPORT_IF_ERROR_WITH_CTX(expr, ctx) \
  GPUReportIfError(expr, ctx, #expr)
#define GPU_REPORT_IF_ERROR(expr) GPU_REPORT_IF_ERROR_WITH_CTX(expr, nullptr)

// Implement the GPU module cache and share what can be shared.

namespace mlir {
namespace kernel_gen {
namespace tf_framework {

GPURuntimeCache::~GPURuntimeCache() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStf_gpu_runtime_wrappersDTcc mht_0(mht_0_v, 240, "", "./tensorflow/compiler/mlir/tools/kernel_gen/tf_gpu_runtime_wrappers.cc", "GPURuntimeCache::~GPURuntimeCache");

  tensorflow::mutex_lock lock(mu_);
  for (auto it : gpu_module_by_data_ptr_) {
#if GOOGLE_CUDA
    GPU_REPORT_IF_ERROR(cuModuleUnload(it.second));
#endif
#if TENSORFLOW_USE_ROCM
    GPU_REPORT_IF_ERROR(hipModuleUnload(it.second));
#endif
  }
}

tensorflow::Status GPURuntimeCache::Create(GPURuntimeCache **dst) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStf_gpu_runtime_wrappersDTcc mht_1(mht_1_v, 255, "", "./tensorflow/compiler/mlir/tools/kernel_gen/tf_gpu_runtime_wrappers.cc", "GPURuntimeCache::Create");

  *dst = new GPURuntimeCache;
  return tensorflow::Status::OK();
}

std::string GPURuntimeCache::DebugString() const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStf_gpu_runtime_wrappersDTcc mht_2(mht_2_v, 263, "", "./tensorflow/compiler/mlir/tools/kernel_gen/tf_gpu_runtime_wrappers.cc", "GPURuntimeCache::DebugString");
 return "GPU runtime cache"; }

GPURuntimeCache::GPUModule GPURuntimeCache::LookupOrLoadModule(void *data) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStf_gpu_runtime_wrappersDTcc mht_3(mht_3_v, 268, "", "./tensorflow/compiler/mlir/tools/kernel_gen/tf_gpu_runtime_wrappers.cc", "GPURuntimeCache::LookupOrLoadModule");

  tensorflow::mutex_lock lock(mu_);
  GPUModule &module = gpu_module_by_data_ptr_[data];

#if GOOGLE_CUDA
  if (!module) GPU_REPORT_IF_ERROR(cuModuleLoadData(&module, data));
#endif
#if TENSORFLOW_USE_ROCM
  if (!module) GPU_REPORT_IF_ERROR(hipModuleLoadData(&module, data));
#endif

  return module;
}

// Implements a C wrapper around the TensorFlow runtime and CUDA (or ROCm)
// library that allows launching a kernel on the current device and stream from
// a binary blob for the module and function name.
// The wrapper uses intptr_t instead of CUDA's unsigned int (or ROCm's unsigned
// int) to match the type of MLIR's index type. This avoids the need for casts
// in the generated MLIR code.
extern "C" void _mlir_ciface_tf_launch_kernel(void *ctx, void *module_blob,
                                              char *kernel_name, intptr_t gridX,
                                              intptr_t gridY, intptr_t gridZ,
                                              intptr_t blockX, intptr_t blockY,
                                              intptr_t blockZ, void **params) {
   std::vector<std::string> mht_4_v;
   mht_4_v.push_back("kernel_name: \"" + (kernel_name == nullptr ? std::string("nullptr") : std::string((char*)kernel_name)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSmlirPStoolsPSkernel_genPStf_gpu_runtime_wrappersDTcc mht_4(mht_4_v, 296, "", "./tensorflow/compiler/mlir/tools/kernel_gen/tf_gpu_runtime_wrappers.cc", "_mlir_ciface_tf_launch_kernel");

  // For empty grids, we don't need to do anything.
  if (!gridX || !gridY || !gridZ) return;

  // Get the GPU module cache.
  auto *op_kernel_ctx = static_cast<tensorflow::OpKernelContext *>(ctx);
  auto *rm = op_kernel_ctx->resource_manager();
  if (rm == nullptr) {
    ReportInternalError(op_kernel_ctx, "expected resource_manager");
    return;
  }
  GPURuntimeCache *cache = nullptr;
  OP_REQUIRES_OK(op_kernel_ctx, rm->LookupOrCreate<GPURuntimeCache>(
                                    rm->default_container(),
                                    GPURuntimeCache::kDefaultResourceName,
                                    &cache, GPURuntimeCache::Create));
  assert(cache != nullptr && "cache creation must not fail");
  tensorflow::core::ScopedUnref ref(cache);

  // Get the GPU module.
  stream_executor::Stream *se_stream =
      op_kernel_ctx->op_device_context()->stream();
  void *stream = se_stream->implementation()->GpuStreamHack();
  GPURuntimeCache::GPUModule module = cache->LookupOrLoadModule(module_blob);

#if GOOGLE_CUDA
  CUfunction function;
  GPU_REPORT_IF_ERROR_WITH_CTX(
      cuModuleGetFunction(&function, module, kernel_name), op_kernel_ctx);
  GPU_REPORT_IF_ERROR_WITH_CTX(
      cuLaunchKernel(function, gridX, gridY, gridZ, blockX, blockY, blockZ,
                     /*sharedMemBytes=*/0, reinterpret_cast<CUstream>(stream),
                     params, nullptr),
      op_kernel_ctx);
#endif
#if TENSORFLOW_USE_ROCM
  hipFunction_t function;
  GPU_REPORT_IF_ERROR_WITH_CTX(
      hipModuleGetFunction(&function, module, kernel_name), op_kernel_ctx);
  GPU_REPORT_IF_ERROR_WITH_CTX(
      hipModuleLaunchKernel(
          function, gridX, gridY, gridZ, blockX, blockY, blockZ,
          /*sharedMemBytes=*/0, reinterpret_cast<hipStream_t>(stream), params,
          nullptr),
      op_kernel_ctx);
#endif
}

}  // namespace tf_framework
}  // namespace kernel_gen
}  // namespace mlir
