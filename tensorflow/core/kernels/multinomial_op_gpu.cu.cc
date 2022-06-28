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
class MHTracer_DTPStensorflowPScorePSkernelsPSmultinomial_op_gpuDTcuDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSmultinomial_op_gpuDTcuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmultinomial_op_gpuDTcuDTcc() {
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

/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_GPU

#include <assert.h>
#include <stdio.h>

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/gpu_prim.h"
#include "tensorflow/core/kernels/multinomial_op.h"
#include "tensorflow/core/kernels/random_op.h"
#include "tensorflow/core/kernels/reduction_gpu_kernels.cu.h"
#include "tensorflow/core/kernels/reduction_ops_common.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

namespace functor {

using GPUDevice = Eigen::GpuDevice;

// Kernel for Multinomial op.  Data is interpreted to have the following shapes:
//   scores: [B, S, C];  maxima: [B, S];  output: [B, S].
template <typename OutputType>
__global__ void MultinomialKernel(int32 nthreads, const int32 num_classes,
                                  const int32 num_samples,
                                  const float* __restrict__ scores,
                                  const float* __restrict__ maxima,
                                  OutputType* __restrict__ output) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmultinomial_op_gpuDTcuDTcc mht_0(mht_0_v, 215, "", "./tensorflow/core/kernels/multinomial_op_gpu.cu.cc", "MultinomialKernel");

  GPU_1D_KERNEL_LOOP(index, nthreads) {
    const int maxima_idx = index / num_classes;
    if (ldg(maxima + maxima_idx) == ldg(scores + index)) {
      using UnsignedOutputType = typename std::make_unsigned<OutputType>::type;
      GpuAtomicMax(reinterpret_cast<UnsignedOutputType*>(output + maxima_idx),
                   static_cast<UnsignedOutputType>(index % num_classes));
    }
  }
}

template <typename T, typename OutputType>
struct MultinomialFunctor<GPUDevice, T, OutputType> {
  void operator()(OpKernelContext* ctx, const GPUDevice& d,
                  typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<float>::Flat noises,
                  typename TTypes<float>::Flat scores,
                  typename TTypes<float>::Flat maxima, int batch_size,
                  int num_classes, int num_samples,
                  const random::PhiloxRandom& gen,
                  typename TTypes<OutputType>::Matrix output) {
    // Uniform, [0, 1).
    typedef random::UniformDistribution<random::PhiloxRandom, float> Dist;
    functor::FillPhiloxRandom<GPUDevice, Dist>()(
        ctx, d, /*key=*/nullptr, /*counter=*/nullptr, gen, noises.data(),
        noises.size(), Dist());

    Eigen::IndexList<int, int, int> bsc;
    bsc.set(0, batch_size);
    bsc.set(1, num_samples);
    bsc.set(2, num_classes);

    Eigen::IndexList<int, Eigen::type2index<1>, int> boc;
    boc.set(0, batch_size);
    boc.set(2, num_classes);

    Eigen::IndexList<Eigen::type2index<1>, int, Eigen::type2index<1>> oso;
    oso.set(1, num_samples);

    // Calculates "scores = logits - log(-log(noises))"; B*C*S elements.
    // NOTE: we don't store back to "noises" because having it appear on both
    // sides is potentially unsafe (e.g. Eigen may use ldg() to load RHS data).
    // 2e-30 is chosen so as to be small enough to only change 0 -> 2e-30 while
    // not affect any of the other numbers (smallest is ~1e-7), but not so small
    // that log(x) == -inf, which is why it needs to be larger than 0 in the
    // first place.
    To32Bit(scores).device(d) =
        To32Bit(logits).reshape(boc).broadcast(oso).template cast<float>() -
        ((-((To32Bit(noises) + 2e-30f).log())).log());

    // Max-reduce along classes for each (batch, sample).
    typedef const Eigen::array<TTypes<float>::Tensor::Index, 1>& ReductionAxes;
    Constants<GPUDevice> constants;
    gpuprim::Max op;
    functor::ReduceImpl<float, gpuprim::Max, float*, const float*,
                        ReductionAxes>(
        /*ctx=*/ctx, /*out=*/maxima.data(), /*in=*/scores.data(), /*in_rank=*/2,
        /*in_dim0=*/batch_size * num_samples,
        /*in_dim1=*/num_classes, /*in_dim2=*/1, /*out_rank=*/1,
        /*reduction_axes=*/constants.kOne, /*Op=*/op);

    // Necessary for atomicMax() inside the kernel.
    output.device(d) = output.constant(0LL);

    const int32 work_items = batch_size * num_samples * num_classes;
    GpuLaunchConfig config = GetGpuLaunchConfig(work_items, d);
    TF_CHECK_OK(GpuLaunchKernel(
        MultinomialKernel<OutputType>, config.block_count,
        config.thread_per_block, 0, d.stream(), config.virtual_thread_count,
        num_classes, num_samples, scores.data(), maxima.data(), output.data()));
  }
};

// Explicit instantiation of the GPU functors.
template struct MultinomialFunctor<GPUDevice, Eigen::half, int32>;
template struct MultinomialFunctor<GPUDevice, float, int32>;
template struct MultinomialFunctor<GPUDevice, double, int32>;
template struct MultinomialFunctor<GPUDevice, int32, int32>;
template struct MultinomialFunctor<GPUDevice, int64, int32>;

template struct MultinomialFunctor<GPUDevice, Eigen::half, int64>;
template struct MultinomialFunctor<GPUDevice, float, int64>;
template struct MultinomialFunctor<GPUDevice, double, int64>;
template struct MultinomialFunctor<GPUDevice, int32, int64>;
template struct MultinomialFunctor<GPUDevice, int64, int64>;

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
