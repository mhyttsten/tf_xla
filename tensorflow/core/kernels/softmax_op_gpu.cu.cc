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
class MHTracer_DTPStensorflowPScorePSkernelsPSsoftmax_op_gpuDTcuDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsoftmax_op_gpuDTcuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsoftmax_op_gpuDTcuDTcc() {
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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/gpu_prim.h"
#include "tensorflow/core/kernels/reduction_gpu_kernels.cu.h"
#include "tensorflow/core/kernels/reduction_ops_common.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

namespace {

template <typename U, typename T>
__device__ __host__ EIGEN_STRONG_INLINE
    typename std::enable_if<!std::is_same<T, U>::value, U>::type
    strict_cast(T t);

template <typename U, typename T>
__device__ __host__ EIGEN_STRONG_INLINE
    typename std::enable_if<std::is_same<T, U>::value, U>::type
    strict_cast(T t) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsoftmax_op_gpuDTcuDTcc mht_0(mht_0_v, 214, "", "./tensorflow/core/kernels/softmax_op_gpu.cu.cc", "strict_cast");

  return t;
}

template <>
__device__ __host__ EIGEN_STRONG_INLINE float strict_cast<float, Eigen::half>(
    Eigen::half t) {
  return functor::HalfToFloat()(t);
}

template <>
__device__ __host__ EIGEN_STRONG_INLINE Eigen::half
strict_cast<Eigen::half, float>(float t) {
  return functor::FloatToHalf()(t);
}

template <typename T>
struct softmax_traits {
  using accumulator_type = T;
};

template <>
struct softmax_traits<Eigen::half> {
  using accumulator_type = float;
};

template <typename T, typename U, int kUnroll>
__global__ void GenerateNormalizedProb(const T* logits, const U* sum_probs,
                                       const T* max_logits, T* output,
                                       const int num_rows, const int num_cols,
                                       const bool in_log_space) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int row, col;

  // TODO(jamesqin): change to half2 load when inputs are Eigen::half.
  U input[kUnroll];
  U max_val[kUnroll];
  U result[kUnroll];
  for (int i = 0; i < kUnroll; i++) {
    row = tid / num_cols;
    col = tid % num_cols;
    if (row < num_rows && col < num_cols) {
      input[i] = strict_cast<U>(logits[tid]);
      max_val[i] = strict_cast<U>(ldg(max_logits + row));
    }
    tid += gridDim.x * blockDim.x;
  }

  tid = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = 0; i < kUnroll; i++) {
    row = tid / num_cols;
    col = tid % num_cols;
    if (row < num_rows && col < num_cols) {
      if (in_log_space) {
        result[i] = input[i] - max_val[i] - log(ldg(sum_probs + row));
      } else {
        result[i] = exp(input[i] - max_val[i]) / ldg(sum_probs + row);
      }
      output[tid] = strict_cast<T>(result[i]);
    }
    tid += gridDim.x * blockDim.x;
  }
}

template <>
__global__ void GenerateNormalizedProb<Eigen::half, float, 8>(
    const Eigen::half* logits, const float* sum_probs,
    const Eigen::half* max_logits, Eigen::half* output, const int num_rows,
    const int num_cols, const bool in_log_space) {
  const int kUnroll = 8;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int idx[kUnroll];
  int row[kUnroll];

  float input[kUnroll];
  float max_val[kUnroll];
  float result[kUnroll];

  if (tid * kUnroll + kUnroll - 1 < num_rows * num_cols) {
    ulonglong2 logits_d =
        *reinterpret_cast<const ulonglong2*>(logits + tid * kUnroll);
    Eigen::half* logits_h = reinterpret_cast<Eigen::half*>(&logits_d);
    ulonglong2 output_d;
    Eigen::half* output_h = reinterpret_cast<Eigen::half*>(&output_d);

    for (int i = 0; i < kUnroll; i++) {
      idx[i] = tid * kUnroll + i;
      row[i] = idx[i] / num_cols;
      input[i] = strict_cast<float>(logits_h[i]);
      max_val[i] = strict_cast<float>(ldg(max_logits + row[i]));
      if (in_log_space) {
        result[i] = input[i] - max_val[i] - log(ldg(sum_probs + row[i]));
      } else {
        result[i] = exp(input[i] - max_val[i]) / ldg(sum_probs + row[i]);
      }
      output_h[i] = strict_cast<Eigen::half>(result[i]);
    }

    *reinterpret_cast<ulonglong2*>(output + tid * kUnroll) = output_d;
  } else {
    for (int i = 0; i < kUnroll; i++) {
      if (tid * kUnroll + i < num_rows * num_cols) {
        idx[i] = tid * kUnroll + i;
        row[i] = idx[i] / num_cols;
        input[i] = strict_cast<float>(logits[idx[i]]);
        max_val[i] = strict_cast<float>(ldg(max_logits + row[i]));
        if (in_log_space) {
          result[i] = input[i] - max_val[i] - log(ldg(sum_probs + row[i]));
        } else {
          result[i] = exp(input[i] - max_val[i]) / ldg(sum_probs + row[i]);
        }
        output[idx[i]] = strict_cast<Eigen::half>(result[i]);
      }
    }
  }
}

template <typename T, typename U>
struct SubtractAndExpFunctor {
  __host__ __device__ SubtractAndExpFunctor(const T* __restrict__ logits,
                                            const T* __restrict__ max_logits,
                                            const int num_cols)
      : logits_(logits), max_logits_(max_logits), num_cols_(num_cols) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsoftmax_op_gpuDTcuDTcc mht_1(mht_1_v, 339, "", "./tensorflow/core/kernels/softmax_op_gpu.cu.cc", "SubtractAndExpFunctor");
}

  __host__ __device__ U operator()(const int gid) const {
    // TODO(jamesqin): change to half2 load when inputs are Eigen::half.
    const U diff =
        strict_cast<U>(logits_[gid] - ldg(max_logits_ + gid / num_cols_));
    return exp(diff);
  }

  const T* logits_;
  const T* max_logits_;
  const int num_cols_;
};

template <typename T, typename Op, typename InputIter>
void DoRowReduction(OpKernelContext* context, T* output, InputIter input,
                    int rows, int cols) {
  typedef const Eigen::array<TTypes<float>::Tensor::Index, 1>& ReductionAxes;
  Constants<GPUDevice> constants;

  Op op;

  functor::ReduceImpl<T, Op, T*, InputIter, ReductionAxes>(
      context, output, input, 2, rows, cols, 1, 1, constants.kOne, op);
}
}  // namespace

template <typename T>
class SoftmaxOpGPU : public OpKernel {
 public:
  explicit SoftmaxOpGPU(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsoftmax_op_gpuDTcuDTcc mht_2(mht_2_v, 372, "", "./tensorflow/core/kernels/softmax_op_gpu.cu.cc", "SoftmaxOpGPU");

    log_ = absl::StartsWith(type_string(), "Log");
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsoftmax_op_gpuDTcuDTcc mht_3(mht_3_v, 379, "", "./tensorflow/core/kernels/softmax_op_gpu.cu.cc", "Compute");

    const Tensor& logits_in_ = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsVectorOrHigher(logits_in_.shape()),
                errors::InvalidArgument("logits must have >= 1 dimension, got ",
                                        logits_in_.shape().DebugString()));
    auto logits_in = logits_in_.flat_inner_dims<T>();
    const int rows = logits_in.dimension(0);
    const int cols = logits_in.dimension(1);
    Tensor* softmax_out = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 0, logits_in_.shape(), &softmax_out));

    const auto& cu_stream = GetGpuStream(context);
    if (logits_in_.NumElements() > 0) {
      Tensor max_logits;
      Tensor sum_probs;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<T>::value,
                                            softmax_out->shape(), &max_logits));

      typedef typename softmax_traits<T>::accumulator_type acc_type;
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DataTypeToEnum<acc_type>::value,
                                            softmax_out->shape(), &sum_probs));

      DoRowReduction<T, gpuprim::Max, const T*>(
          context, const_cast<T*>(max_logits.flat<T>().data()),
          reinterpret_cast<const T*>(logits_in_.flat<T>().data()), rows, cols);


      gpuprim::CountingInputIterator<int> counting_iterator(0);
      using InputIterType =
          gpuprim::TransformInputIterator<acc_type,
                                          SubtractAndExpFunctor<T, acc_type>,
                                          gpuprim::CountingInputIterator<int>>;

      InputIterType input_itr(
          counting_iterator,
          SubtractAndExpFunctor<T, acc_type>(
              reinterpret_cast<const T*>(logits_in_.flat<T>().data()),
              reinterpret_cast<const T*>(max_logits.flat<T>().data()), cols));

      DoRowReduction<acc_type, gpuprim::Sum, InputIterType>(
          context, const_cast<acc_type*>(sum_probs.flat<acc_type>().data()),
          input_itr, rows, cols);

      auto in_ptr = reinterpret_cast<uintptr_t>(logits_in_.flat<T>().data());
      auto out_ptr = reinterpret_cast<uintptr_t>(softmax_out->flat<T>().data());
      bool aligned = in_ptr % 16 == 0 && out_ptr % 16 == 0;

      const int numThreadsPerBlock = 128;
      if (DataTypeToEnum<T>::value == DT_HALF && aligned) {
        const int kUnroll = 8;
        const int numBlocks =
            Eigen::divup(rows * cols, numThreadsPerBlock * kUnroll);
        TF_CHECK_OK(GpuLaunchKernel(
            GenerateNormalizedProb<T, acc_type, kUnroll>, numBlocks,
            numThreadsPerBlock, 0, cu_stream,
            reinterpret_cast<const T*>(logits_in_.flat<T>().data()),
            reinterpret_cast<const acc_type*>(
                sum_probs.flat<acc_type>().data()),
            reinterpret_cast<const T*>(max_logits.flat<T>().data()),
            const_cast<T*>(softmax_out->flat<T>().data()), rows, cols, log_));
      } else {
        const int kUnroll = 4;
        const int numBlocks =
            Eigen::divup(rows * cols, numThreadsPerBlock * kUnroll);
        TF_CHECK_OK(GpuLaunchKernel(
            GenerateNormalizedProb<T, acc_type, kUnroll>, numBlocks,
            numThreadsPerBlock, 0, cu_stream,
            reinterpret_cast<const T*>(logits_in_.flat<T>().data()),
            reinterpret_cast<const acc_type*>(
                sum_probs.flat<acc_type>().data()),
            reinterpret_cast<const T*>(max_logits.flat<T>().data()),
            const_cast<T*>(softmax_out->flat<T>().data()), rows, cols, log_));
      }
    }
  }

 private:
  bool log_;
};

#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Softmax").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      SoftmaxOpGPU<T>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);

#undef REGISTER_GPU
#define REGISTER_GPU(T)                                             \
  REGISTER_KERNEL_BUILDER(                                          \
      Name("LogSoftmax").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      SoftmaxOpGPU<T>);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU);

#undef REGISTER_GPU

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
