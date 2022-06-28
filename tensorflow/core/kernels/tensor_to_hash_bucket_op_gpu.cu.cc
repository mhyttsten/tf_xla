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
class MHTracer_DTPStensorflowPScorePSkernelsPStensor_to_hash_bucket_op_gpuDTcuDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_to_hash_bucket_op_gpuDTcuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPStensor_to_hash_bucket_op_gpuDTcuDTcc() {
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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/tensor_to_hash_bucket_op.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "third_party/farmhash_gpu/src/farmhash_gpu.h"

namespace tensorflow {

namespace {

// We set the buffer size to 20 as it is sufficient to cover the number of
// digits in any integer type.
constexpr int kSharedMemBufferSizePerThread = 20;

template <typename T>
__device__ __forceinline__ void FillDigits(T val, int num_digits, int* i,
                                           char* buf) {
   std::vector<std::string> mht_0_v;
   mht_0_v.push_back("buf: \"" + (buf == nullptr ? std::string("nullptr") : std::string((char*)buf)) + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_to_hash_bucket_op_gpuDTcuDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/kernels/tensor_to_hash_bucket_op_gpu.cu.cc", "FillDigits");

  eigen_assert(num_digits <= kSharedMemBufferSizePerThread - (*i));

  int factor = (val < 0 ? -1 : 1);

  int num_digits_a = num_digits;
  do {
    int digit = static_cast<int>((val % 10) * factor);
    buf[(*i) + num_digits - 1] = digit + '0';
    val /= 10;
    num_digits--;
  } while (val != 0);

  (*i) += num_digits_a;
}

template <typename T>
__device__ __forceinline__ int IntegerToString(T val, char* buf) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("buf: \"" + (buf == nullptr ? std::string("nullptr") : std::string((char*)buf)) + "\"");
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_to_hash_bucket_op_gpuDTcuDTcc mht_1(mht_1_v, 223, "", "./tensorflow/core/kernels/tensor_to_hash_bucket_op_gpu.cu.cc", "IntegerToString");

  int num_digits = 0;
  T val_a = val;
  do {
    val_a = val_a / 10;
    num_digits++;
  } while (val_a != 0);

  int i = 0;
  if (val < 0) {
    buf[i++] = '-';
  }

  FillDigits(val, num_digits, &i, buf);

  return i;
}

template <typename T>
__global__ void ComputeHashes(const T* __restrict__ vals, int vals_size,
                              int64 num_buckets, int64* __restrict__ hashes) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStensor_to_hash_bucket_op_gpuDTcuDTcc mht_2(mht_2_v, 246, "", "./tensorflow/core/kernels/tensor_to_hash_bucket_op_gpu.cu.cc", "ComputeHashes");

  extern __shared__ char s[];

  GPU_1D_KERNEL_LOOP(tid, vals_size) {
    int size = IntegerToString(vals[tid],
                               s + threadIdx.x * kSharedMemBufferSizePerThread);
    uint64_t a_hash = ::util_gpu::Fingerprint64(
        s + threadIdx.x * kSharedMemBufferSizePerThread, size);
    int64 a_bucket = static_cast<int64_t>(a_hash % num_buckets);
    hashes[tid] = a_bucket;
  }
}

}  // end namespace

namespace functor {

template <typename T>
void LaunchTensorToHashBucket<Eigen::GpuDevice, T>::operator()(
    OpKernelContext* c, const int64 num_buckets, const T* input,
    const int num_elems, int64* output) {
  auto* stream = c->op_device_context()->stream();
  const Eigen::GpuDevice& d = c->eigen_gpu_device();
  if (num_elems > 0) {
    constexpr size_t kThreadsLimitInBlock = 1024;

    size_t smem_bytes_allowed =
        stream->parent()->GetDeviceDescription().shared_memory_per_block();
    auto smem_bytes_per_thread = kSharedMemBufferSizePerThread * sizeof(char);
    size_t thread_per_block = std::min(
        kThreadsLimitInBlock, smem_bytes_allowed / smem_bytes_per_thread);

    auto smem_bytes_per_block = thread_per_block * smem_bytes_per_thread;
    GpuLaunchConfig config = GetGpuLaunchConfigFixedBlockSize(
        num_elems, d, ComputeHashes<T>, smem_bytes_per_block, thread_per_block);
    OP_REQUIRES_OK(
        c, GpuLaunchKernel(ComputeHashes<T>, config.block_count,
                           config.thread_per_block, smem_bytes_per_block,
                           d.stream(), input, num_elems, num_buckets, output));
  }
}

}  // namespace functor

#define REGISTER_FUNCTORS(type) \
  template struct functor::LaunchTensorToHashBucket<Eigen::GpuDevice, type>;

TF_CALL_INTEGRAL_TYPES(REGISTER_FUNCTORS);

#undef REGISTER_FUNCTORS

}  // namespace tensorflow
#endif  // GOOGLE_CUDA
