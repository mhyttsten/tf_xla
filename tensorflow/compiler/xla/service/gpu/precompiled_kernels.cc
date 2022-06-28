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
class MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSprecompiled_kernelsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSprecompiled_kernelsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSprecompiled_kernelsDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/precompiled_kernels.h"

#include <string>

#include "absl/base/call_once.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/stream_executor/gpu/asm_compiler.h"

namespace xla {
namespace gpu {
namespace {

// GPU kernel to populate an array of pointers:
//
//   [base + stride * i for i in range(n)].
//
// Generated from the following CUDA code.
//
// extern "C" {
// __global__ void __xla_MakeBatchPointers(char* base, int stride,
//                                         int n, void** ptrs_out) {
//   int idx = threadIdx.x + blockIdx.x * blockDim.x;
//   if (idx >= n) return;
//   ptrs_out[idx] = base + idx * stride;
// }
// }
constexpr const char* kMakeBatchPointersPtx = R"(
.version 4.2
.target sm_35
.address_size 64

.visible .entry __xla_MakeBatchPointers(
        .param .u64 __xla_MakeBatchPointers_param_0,
        .param .u32 __xla_MakeBatchPointers_param_1,
        .param .u32 __xla_MakeBatchPointers_param_2,
        .param .u64 __xla_MakeBatchPointers_param_3
)
{
        .reg .pred      %p<2>;
        .reg .b32       %r<8>;
        .reg .b64       %rd<8>;

        ld.param.u32    %r2, [__xla_MakeBatchPointers_param_2];
        mov.u32         %r3, %tid.x;
        mov.u32         %r4, %ctaid.x;
        mov.u32         %r5, %ntid.x;
        mad.lo.s32      %r6, %r4, %r5, %r3;
        setp.ge.s32     %p1, %r6, %r2;
        @%p1 bra        LBB0_2;
        ld.param.u64    %rd3, [__xla_MakeBatchPointers_param_0];
        ld.param.u64    %rd4, [__xla_MakeBatchPointers_param_3];
        cvta.to.global.u64      %rd5, %rd4;
        ld.param.u32    %r1, [__xla_MakeBatchPointers_param_1];
        mul.wide.s32    %rd6, %r6, 8;
        add.s64         %rd1, %rd5, %rd6;
        mul.lo.s32      %r7, %r6, %r1;
        cvt.s64.s32     %rd7, %r7;
        add.s64         %rd2, %rd3, %rd7;
        st.global.u64   [%rd1], %rd2;
LBB0_2:
        ret;
}
)";

// Lazily compiles ptx kernel, once per StreamExecutor.
//
// Thread-safe.
template <typename... KernelArgs>
class LazyKernel {
 public:
  LazyKernel(absl::string_view kernel_name, const char* ptx,
             const se::GpuAsmOpts& asm_opts)
      : kernel_name_(kernel_name), ptx_(ptx), asm_opts_(asm_opts) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("kernel_name: \"" + std::string(kernel_name.data(), kernel_name.size()) + "\"");
   mht_0_v.push_back("ptx: \"" + (ptx == nullptr ? std::string("nullptr") : std::string((char*)ptx)) + "\"");
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSprecompiled_kernelsDTcc mht_0(mht_0_v, 262, "", "./tensorflow/compiler/xla/service/gpu/precompiled_kernels.cc", "LazyKernel");
}

  StatusOr<se::TypedKernel<KernelArgs...>*> Get(
      se::StreamExecutor* stream_exec) {
    absl::MutexLock lock(&mu_);

    auto result = kernels_.emplace(stream_exec, nullptr);
    if (result.second) {
      absl::Span<const uint8_t> compiled_ptx;
      StatusOr<absl::Span<const uint8_t>> compiled_ptx_or =
          se::CompileGpuAsmOrGetCached(stream_exec->device_ordinal(), ptx_,
                                       asm_opts_);
      if (compiled_ptx_or.ok()) {
        compiled_ptx = compiled_ptx_or.ConsumeValueOrDie();
      } else {
        static absl::once_flag logged_once;
        absl::call_once(logged_once, [&]() {
          LOG(WARNING)
              << compiled_ptx_or.status().ToString()
              << "\nRelying on driver to perform ptx compilation. "
              << "\nSetting XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda "
              << " or modifying $PATH can be used to set the location of ptxas."
              << "\nThis message will only be logged once.";
        });
      }

      auto kernel = stream_exec->CreateTypedKernel<KernelArgs...>(
          kernel_name_, ptx_, compiled_ptx);
      if (kernel.ok()) {
        result.first->second = *std::move(kernel);
      } else {
        kernels_.erase(result.first);
        return kernel.status();
      }
    }
    return result.first->second.get();
  }

 private:
  std::string kernel_name_;
  const char* ptx_;
  se::GpuAsmOpts asm_opts_;

  absl::Mutex mu_;

  // A mutex keyed on StreamExecutor* is ok because StreamExecutors are never
  // destroyed.
  absl::flat_hash_map<se::StreamExecutor*,
                      std::unique_ptr<se::TypedKernel<KernelArgs...>>>
      kernels_ ABSL_GUARDED_BY(mu_);
};

}  // anonymous namespace

Status MakeBatchPointers(se::Stream* stream, const se::GpuAsmOpts& asm_opts,
                         se::DeviceMemoryBase base_ptr, int stride_bytes, int n,
                         se::DeviceMemoryBase ptrs_out) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSxlaPSservicePSgpuPSprecompiled_kernelsDTcc mht_1(mht_1_v, 321, "", "./tensorflow/compiler/xla/service/gpu/precompiled_kernels.cc", "MakeBatchPointers");

  static auto* lazy_kernel =
      new LazyKernel<se::DeviceMemoryBase /*base_ptr*/, int /*stride_bytes*/,
                     int /*n*/, se::DeviceMemoryBase /*ptrs_out*/>(
          "__xla_MakeBatchPointers", kMakeBatchPointersPtx, asm_opts);

  TF_ASSIGN_OR_RETURN(auto kernel, lazy_kernel->Get(stream->parent()));

  constexpr int kThreads = 128;
  stream->ThenLaunch(se::ThreadDim(kThreads, 1, 1),
                     se::BlockDim(CeilOfRatio(n, kThreads), 1, 1), *kernel,
                     base_ptr, stride_bytes, n, ptrs_out);
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
