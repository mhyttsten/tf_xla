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
class MHTracer_DTPStensorflowPScorePSkernelsPSstateful_random_ops_gpuDTcuDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSstateful_random_ops_gpuDTcuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSstateful_random_ops_gpuDTcuDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#if TENSORFLOW_USE_ROCM
#include "rocm/include/hip/hip_runtime.h"
#endif

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/random_op_gpu.h"
#include "tensorflow/core/kernels/stateful_random_ops_cpu_gpu.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_launch_config.h"

// ROCm hipMemcpyToSymbol can only see this variable if it's in global namespace
__device__ int tensorflow_philox_thread_counter;

namespace tensorflow {

namespace functor {

using random::PhiloxRandom;

template <typename Distribution>
__global__ void FillKernel(
    Distribution dist, int64 state_size, int64 output_size,
    StateElementType* __restrict__ state_data,
    typename Distribution::ResultElementType* __restrict__ output_data) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateful_random_ops_gpuDTcuDTcc mht_0(mht_0_v, 211, "", "./tensorflow/core/kernels/stateful_random_ops_gpu.cu.cc", "FillKernel");

  // Threads in this block share `philox`. Thread 0 is responsible for
  // initializing it.
  __shared__ char philox_raw[sizeof(PhiloxRandom)];
  auto philox = reinterpret_cast<PhiloxRandom*>(philox_raw);
  if (threadIdx.x == 0) {
    *philox = GetPhiloxRandomFromMem(state_data);
  }
  __syncthreads();
  functor::FillPhiloxRandomKernel<Distribution,
                                  Distribution::kVariableSamplesPerOutput>()
      .Run(/*key=*/nullptr, /*counter=*/nullptr, *philox, output_data,
           output_size, dist);
  // The last thread updates the state.
  auto total_thread_count = gridDim.x * blockDim.x;
  auto old_counter_value = atomicAdd(&tensorflow_philox_thread_counter, 1);
  if (old_counter_value == total_thread_count - 1) {
    UpdateMemWithPhiloxRandom(*philox, output_size, state_data);
  }
}

template <typename Distribution>
void UpdateVariableAndFill_Philox<GPUDevice, Distribution>::operator()(
    OpKernelContext* ctx, const GPUDevice& d, Distribution dist,
    UpdateVariableAndFill_Philox_Arg* arg,
    typename Distribution::ResultElementType* output_data) {
  int64 output_size = arg->output_size;
  int64 alg_tag_skip = arg->alg_tag_skip;
  Tensor* state_tensor = arg->state_tensor;
  OP_REQUIRES(ctx, state_tensor != 0,
              errors::InvalidArgument("Null state tensor"));
  OP_REQUIRES(
      ctx, alg_tag_skip == 0,
      errors::InvalidArgument(
          "GPU kernel doesn't support reading algorithm from state variable, "
          "so alg_tag_skip must be 0; got",
          alg_tag_skip));
  auto state_tensor_flat = state_tensor->flat<StateElementType>();
  auto state_size = state_tensor_flat.size();
  auto state_data = state_tensor_flat.data();
  // maximize occupancy
  const int kGroupSize = Distribution::kResultElementCount;
  int work_element_count = (output_size + kGroupSize - 1) / kGroupSize;
  GpuLaunchConfig cfg =
      GetGpuLaunchConfig(work_element_count, d, FillKernel<Distribution>, 0, 0);
  int zero = 0;
#if GOOGLE_CUDA
  cudaMemcpyToSymbol(tensorflow_philox_thread_counter, &zero, sizeof(int));
#else  // TENSORFLOW_USE_ROCM
  int status = hipMemcpyToSymbol(HIP_SYMBOL(tensorflow_philox_thread_counter),
                                 &zero, sizeof(int));
  OP_REQUIRES(ctx, status == hipSuccess,
              errors::InvalidArgument("hipMemcpyToSymbol failed"));
#endif
  TF_CHECK_OK(GpuLaunchKernel(
      FillKernel<Distribution>, cfg.block_count, cfg.thread_per_block, 0,
      d.stream(), dist, state_size, output_size, state_data, output_data));
}

// Precondition: there is only 1 block and 1 thread.
__global__ void SkipKernel(const StateElementType* __restrict__ in_data,
                           uint64 delta,
                           StateElementType* __restrict__ out_data) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstateful_random_ops_gpuDTcuDTcc mht_1(mht_1_v, 276, "", "./tensorflow/core/kernels/stateful_random_ops_gpu.cu.cc", "SkipKernel");

  auto counter = GetCounterFromMem(reinterpret_cast<const uint64*>(in_data));
  UpdateCounterMemWithPhiloxRandom(counter, delta, out_data);
}

void RngSkip_Philox<GPUDevice>::operator()(const GPUDevice& d,
                                           const StateElementType* in_data,
                                           uint64 delta,
                                           StateElementType* out_data) {
  TF_CHECK_OK(GpuLaunchKernel(SkipKernel, 1, 1, 0, d.stream(), in_data, delta,
                              out_data));
}

// Explicit instantiation of the GPU distributions functors.

// clang-format off
// NVCC cannot handle ">>" properly
template struct UpdateVariableAndFill_Philox<
    GPUDevice, random::NormalDistribution<random::PhiloxRandom, Eigen::half> >;
template struct UpdateVariableAndFill_Philox<
    GPUDevice, random::NormalDistribution<random::PhiloxRandom, float> >;
template struct UpdateVariableAndFill_Philox<
    GPUDevice, random::NormalDistribution<random::PhiloxRandom, double> >;
template struct UpdateVariableAndFill_Philox<
    GPUDevice, random::TruncatedNormalDistribution<
                 random::SingleSampleAdapter<random::PhiloxRandom>,
                 Eigen::half> >;
template struct UpdateVariableAndFill_Philox<
    GPUDevice, random::TruncatedNormalDistribution<
                 random::SingleSampleAdapter<random::PhiloxRandom>,
                 float> >;
template struct UpdateVariableAndFill_Philox<
    GPUDevice, random::TruncatedNormalDistribution<
                 random::SingleSampleAdapter<random::PhiloxRandom>,
                 double> >;
template struct UpdateVariableAndFill_Philox<
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, Eigen::half> >;
template struct UpdateVariableAndFill_Philox<
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, float> >;
template struct UpdateVariableAndFill_Philox<
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, double> >;
template struct UpdateVariableAndFill_Philox<
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, int32> >;
template struct UpdateVariableAndFill_Philox<
    GPUDevice, random::UniformDistribution<random::PhiloxRandom, int64> >;
template struct UpdateVariableAndFill_Philox<
    GPUDevice, random::UniformFullIntDistribution<
                 random::PhiloxRandom, int32> >;
template struct UpdateVariableAndFill_Philox<
    GPUDevice, random::UniformFullIntDistribution<
                 random::PhiloxRandom, int64> >;
template struct UpdateVariableAndFill_Philox<
    GPUDevice, random::UniformFullIntDistribution<
                 random::PhiloxRandom, uint32> >;
template struct UpdateVariableAndFill_Philox<
    GPUDevice, random::UniformFullIntDistribution<
                 random::PhiloxRandom, uint64> >;
// clang-format on

}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
