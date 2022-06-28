/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_SCATTER_FUNCTOR_GPU_CU_H_
#define TENSORFLOW_CORE_KERNELS_SCATTER_FUNCTOR_GPU_CU_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSscatter_functor_gpuDTcuDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_functor_gpuDTcuDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSscatter_functor_gpuDTcuDTh() {
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


#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/scatter_functor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace scatter_op_gpu {

template <typename T, scatter_op::UpdateOp op>
struct ScatterOpKernelBody;

template <typename T>
struct ScatterOpKernelBody<T, scatter_op::UpdateOp::ASSIGN> {
  __device__ void operator()(T* __restrict__ dest, T src) const { *dest = src; }
};

template <typename T>
struct ScatterOpKernelBody<T, scatter_op::UpdateOp::ADD> {
  __device__ void operator()(T* __restrict__ dest, T src) const {
    GpuAtomicAdd(dest, src);
  }
};

template <typename T>
struct ScatterOpKernelBody<T, scatter_op::UpdateOp::SUB> {
  __device__ void operator()(T* __restrict__ dest, T src) const {
    GpuAtomicSub(dest, src);
  }
};

template <typename T>
struct ScatterOpKernelBody<T, scatter_op::UpdateOp::MUL> {
  __device__ void operator()(T* __restrict__ dest, T src) const {
    GpuAtomicMul(dest, src);
  }
};

template <typename T>
struct ScatterOpKernelBody<T, scatter_op::UpdateOp::DIV> {
  __device__ void operator()(T* __restrict__ dest, T src) const {
    GpuAtomicDiv(dest, src);
  }
};

template <typename T>
struct ScatterOpKernelBody<T, scatter_op::UpdateOp::MIN> {
  __device__ void operator()(T* __restrict__ dest, T src) const {
    GpuAtomicMin(dest, src);
  }
};

template <typename T>
struct ScatterOpKernelBody<T, scatter_op::UpdateOp::MAX> {
  __device__ void operator()(T* __restrict__ dest, T src) const {
    GpuAtomicMax(dest, src);
  }
};

template <typename T, typename Index, scatter_op::UpdateOp op>
__global__ void ScatterOpCustomKernel(T* __restrict__ params,
                                      const T* __restrict__ updates,
                                      const Index* __restrict__ indices,
                                      Index first_dim_size, Index updates_size,
                                      Index indices_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_functor_gpuDTcuDTh mht_0(mht_0_v, 258, "", "./tensorflow/core/kernels/scatter_functor_gpu.cu.h", "ScatterOpCustomKernel");

  Index update_block = updates_size / indices_size;
  ScatterOpKernelBody<T, op> body;
  GPU_1D_KERNEL_LOOP(i, updates_size) {
    int indices_i = i / update_block;
    int updates_i = i;
    int param_first_index = indices[indices_i];
    if (!(param_first_index >= 0 && param_first_index < first_dim_size)) {
      // Ignore indices that are out of range.
      continue;
    }
    int64 params_i = param_first_index * update_block + (i % update_block);
    body(&params[params_i], ldg(updates + updates_i));
  }
}

template <typename T, typename Index, scatter_op::UpdateOp op>
__global__ void ScatterScalarOpCustomKernel(T* __restrict__ params,
                                            const T* __restrict__ update,
                                            const Index* __restrict__ indices,
                                            Index first_dim_size,
                                            Index indices_size,
                                            Index synthesized_updates_size) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSscatter_functor_gpuDTcuDTh mht_1(mht_1_v, 283, "", "./tensorflow/core/kernels/scatter_functor_gpu.cu.h", "ScatterScalarOpCustomKernel");

  Index update_block = synthesized_updates_size / indices_size;
  ScatterOpKernelBody<T, op> body;
  GPU_1D_KERNEL_LOOP(i, synthesized_updates_size) {
    int indices_i = i / update_block;
    int param_first_index = indices[indices_i];
    const T update_val = *update;
    if (!(param_first_index >= 0 && param_first_index < first_dim_size)) {
      // Ignore indices that are out of range.
      continue;
    }
    int params_i = param_first_index * update_block + (i % update_block);
    body(&params[params_i], update_val);
  }
}

}  // namespace scatter_op_gpu

namespace functor {
// Specialization for a GPU device.
template <typename T, typename Index, scatter_op::UpdateOp op>
struct ScatterFunctor<GPUDevice, T, Index, op> {
  Index operator()(OpKernelContext* c, const GPUDevice& d,
                   typename TTypes<T>::Matrix params,
                   typename TTypes<T>::ConstMatrix updates,
                   typename TTypes<Index>::ConstFlat indices) {
    // TODO(b/31801742): Implement indices range check. The hardest part is
    // with returning a value after the range check, as we do not want to do
    // device to host memcpy during a stream.
    const Index first_dim_size = params.dimension(0);
    const Index indices_size = indices.size();
    const Index updates_size = updates.size();
    GpuLaunchConfig config = GetGpuLaunchConfig(updates_size, d);
    TF_CHECK_OK(GpuLaunchKernel(
        scatter_op_gpu::ScatterOpCustomKernel<T, Index, op>, config.block_count,
        config.thread_per_block, 0, d.stream(), params.data(), updates.data(),
        indices.data(), first_dim_size, updates_size, indices_size));
    return -1;
  }
};

template <typename T, typename Index, scatter_op::UpdateOp op>
struct ScatterScalarFunctor<GPUDevice, T, Index, op> {
  Index operator()(OpKernelContext* c, const GPUDevice& d,
                   typename TTypes<T>::Matrix params,
                   const typename TTypes<T>::ConstScalar update,
                   typename TTypes<Index>::ConstFlat indices) {
    // TODO(b/31801742): Implement indices range check. The hardest part is
    // with returning a value after the range check, as we do not want to do
    // device to host memcpy during a stream.
    const Index first_dim_size = params.dimension(0);
    const Index indices_size = indices.size();
    const Index synthesized_updates_size = indices_size * params.dimension(1);
    GpuLaunchConfig config = GetGpuLaunchConfig(synthesized_updates_size, d);
    TF_CHECK_OK(GpuLaunchKernel(
        scatter_op_gpu::ScatterScalarOpCustomKernel<T, Index, op>,
        config.block_count, config.thread_per_block, 0, d.stream(),
        params.data(), update.data(), indices.data(), first_dim_size,
        indices_size, synthesized_updates_size));
    return -1;
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_KERNELS_SCATTER_FUNCTOR_GPU_CU_H_
