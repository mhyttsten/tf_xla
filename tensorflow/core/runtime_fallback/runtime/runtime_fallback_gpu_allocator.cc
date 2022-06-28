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
class MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSruntime_fallback_gpu_allocatorDTcc {
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
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSruntime_fallback_gpu_allocatorDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSruntime_fallback_gpu_allocatorDTcc() {
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
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA) || \
    (defined(TENSORFLOW_USE_ROCM) && TENSORFLOW_USE_ROCM)

#include "tensorflow/core/runtime_fallback/runtime/runtime_fallback_gpu_allocator.h"

#include "llvm/Support/Errc.h"
#include "tensorflow/core/platform/mutex.h"
#include "tfrt/gpu/gpu_types.h"  // from @tf_runtime
#include "tfrt/support/ref_count.h"  // from @tf_runtime
#include "tfrt/support/string_util.h"  // from @tf_runtime

namespace tensorflow {

class RuntimeFallbackGpuAllocator : public tfrt::gpu::GpuAllocator {
 public:
  explicit RuntimeFallbackGpuAllocator(
      tensorflow::Allocator* tf_gpu_allocator,
      const tfrt::gpu::wrapper::Context& context)
      : tf_gpu_allocator_(tf_gpu_allocator), context_(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSruntime_fallback_gpu_allocatorDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/runtime_fallback/runtime/runtime_fallback_gpu_allocator.cc", "RuntimeFallbackGpuAllocator");
}
  ~RuntimeFallbackGpuAllocator() override;

  llvm::Expected<tfrt::gpu::GpuPointer> Allocate(
      size_t size, tfrt::gpu::wrapper::Stream stream) override;

  llvm::Error Deallocate(tfrt::gpu::GpuPointer pointer,
                         tfrt::gpu::wrapper::Stream stream) override;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RuntimeFallbackGpuAllocator);

  // Structures immutable after construction

  // Does not own tf_gpu_allocator.
  tensorflow::Allocator* tf_gpu_allocator_;

  tfrt::gpu::wrapper::Context context_;

  // Structures mutable after construction
  mutable tensorflow::mutex mu_;

  // Because we don't support multiple streams, stream_ is the stream
  // for all allocations. All allocation requests on a different stream will be
  // denied.
  // We can't easily support "stream transitioning" now because:
  //  - we need to synchronize the former stream when we transition to the new
  //  stream.
  //  - the allocator is not notified when the stream is destroyed. So, the
  //  synchronization can happen after the stream is destroyed causing
  //  segfault.
  tfrt::gpu::wrapper::Stream stream_ TF_GUARDED_BY(mu_);
};

RuntimeFallbackGpuAllocator::~RuntimeFallbackGpuAllocator() {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSruntime_fallback_gpu_allocatorDTcc mht_1(mht_1_v, 239, "", "./tensorflow/core/runtime_fallback/runtime/runtime_fallback_gpu_allocator.cc", "RuntimeFallbackGpuAllocator::~RuntimeFallbackGpuAllocator");
}

llvm::Expected<tfrt::gpu::GpuPointer> RuntimeFallbackGpuAllocator::Allocate(
    size_t size, tfrt::gpu::wrapper::Stream stream) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSruntime_fallback_gpu_allocatorDTcc mht_2(mht_2_v, 245, "", "./tensorflow/core/runtime_fallback/runtime/runtime_fallback_gpu_allocator.cc", "RuntimeFallbackGpuAllocator::Allocate");

  {
    tensorflow::mutex_lock lock(mu_);
    if (stream_ == nullptr) {
      stream_ = stream;
    } else if (stream != stream_) {
      return llvm::createStringError(
          llvm::errc::invalid_argument,
          "RuntimeFallbackGpuAllocator does not support multiple streams");
    }
  }
  // tfrt::gpu::GpuAllocator::kAlignment is the minimum alignment. AllocateRaw
  // adjusts alignment internally as needed.
  void* gpu_ptr =
      tf_gpu_allocator_->AllocateRaw(tfrt::gpu::GpuAllocator::kAlignment, size);

  // TODO(zhangqiaorjc): AllocateRaw does LOG(WARNING) for different errors, it
  // should return llvm::Error instead.
  if (gpu_ptr == nullptr)
    return llvm::createStringError(
        llvm::errc::invalid_argument,
        tfrt::StrCat("errors trying to allocate ", size));

  return tfrt::gpu::wrapper::Pointer<void>(gpu_ptr, context_.platform());
}

llvm::Error RuntimeFallbackGpuAllocator::Deallocate(
    tfrt::gpu::GpuPointer pointer, tfrt::gpu::wrapper::Stream stream) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSruntime_fallback_gpu_allocatorDTcc mht_3(mht_3_v, 275, "", "./tensorflow/core/runtime_fallback/runtime/runtime_fallback_gpu_allocator.cc", "RuntimeFallbackGpuAllocator::Deallocate");

  tf_gpu_allocator_->DeallocateRaw(pointer.raw());
  return llvm::Error::success();
}

tfrt::gpu::GpuAllocatorFactory CreateRuntimeFallbackGpuAllocatorFactory(
    tensorflow::Allocator* tf_gpu_allocator) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSruntime_fallback_gpu_allocatorDTcc mht_4(mht_4_v, 284, "", "./tensorflow/core/runtime_fallback/runtime/runtime_fallback_gpu_allocator.cc", "CreateRuntimeFallbackGpuAllocatorFactory");

  return [tf_gpu_allocator](const tfrt::gpu::wrapper::Context& context) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSruntime_fallbackPSruntimePSruntime_fallback_gpu_allocatorDTcc mht_5(mht_5_v, 288, "", "./tensorflow/core/runtime_fallback/runtime/runtime_fallback_gpu_allocator.cc", "lambda");

    return std::make_unique<RuntimeFallbackGpuAllocator>(tf_gpu_allocator,
                                                         context);
  };
}

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
