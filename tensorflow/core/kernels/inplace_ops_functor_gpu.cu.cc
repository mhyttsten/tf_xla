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
class MHTracer_DTPStensorflowPScorePSkernelsPSinplace_ops_functor_gpuDTcuDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSinplace_ops_functor_gpuDTcuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSinplace_ops_functor_gpuDTcuDTcc() {
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

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/kernels/inplace_ops_functor.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice Device;

template <typename T>
__global__ void DoParallelConcatOpKernel(int nthreads, const int64 rows,
                                         const int64 cols, int32 loc,
                                         const T* __restrict__ src,
                                         T* __restrict__ dst) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinplace_ops_functor_gpuDTcuDTcc mht_0(mht_0_v, 202, "", "./tensorflow/core/kernels/inplace_ops_functor_gpu.cu.cc", "DoParallelConcatOpKernel");

  GPU_1D_KERNEL_LOOP(idx, nthreads) {
    int64 c = idx % cols;
    int64 r = (loc % rows + rows) % rows;  // Guard index range.
    T* p = dst + r * cols + c;
    const T* q = src + idx;
    *p = ldg(q);
  }
}

template <typename T>
Status DoParallelConcatUpdate(const Device& d, const Tensor& value, int32 loc,
                              Tensor* output) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinplace_ops_functor_gpuDTcuDTcc mht_1(mht_1_v, 217, "", "./tensorflow/core/kernels/inplace_ops_functor_gpu.cu.cc", "DoParallelConcatUpdate");

  const int64 nelem = value.NumElements();
  GpuLaunchConfig cfg = GetGpuLaunchConfig(nelem, d);
  auto Toutput = output->flat_outer_dims<T>();
  const int64 nrows = Toutput.dimension(0);
  const int64 ncols = Toutput.dimension(1);
  const T* src = value.flat<T>().data();
  T* dst = output->flat<T>().data();
  TF_CHECK_OK(GpuLaunchKernel(
      DoParallelConcatOpKernel<T>, cfg.block_count, cfg.thread_per_block, 0,
      d.stream(), cfg.virtual_thread_count, nrows, ncols, loc, src, dst));
  return Status::OK();
}

template <>
Status DoParallelConcat(const Device& d, const Tensor& value, int32 loc,
                        Tensor* output) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinplace_ops_functor_gpuDTcuDTcc mht_2(mht_2_v, 236, "", "./tensorflow/core/kernels/inplace_ops_functor_gpu.cu.cc", "DoParallelConcat");

  CHECK_EQ(value.dtype(), output->dtype());
  switch (value.dtype()) {
#define CASE(type)                                              \
  case DataTypeToEnum<type>::value:                             \
    return DoParallelConcatUpdate<type>(d, value, loc, output); \
    break;

    CASE(float)
    CASE(double)
    CASE(Eigen::half)
// Using TF_CALL_GPU_NUMBER_TYPES(CASE) results in the compiler complaining
// that CASE is not defined...hence the above construction
#undef CASE
    default:
      return errors::InvalidArgument("Unsupported data type: ",
                                     DataTypeString(value.dtype()));
  }
  return Status::OK();
}

template <typename T, InplaceOpType op>
__global__ void DoInplaceOpKernel(int nthreads, const int64 rows,
                                  const int64 cols, const int64 n,
                                  const T* __restrict__ src,
                                  const int32* __restrict__ rowids,
                                  T* __restrict__ dst) {
  GPU_1D_KERNEL_LOOP(idx, nthreads) {
    int64 r = idx / cols;
    int64 c = idx % cols;
    r = (rowids[r] % rows + rows) % rows;  // Guard index range.
    T* p = dst + r * cols + c;
    const T* q = src + idx;
    switch (op) {
      case I_UPDATE:
        *p = ldg(q);
        break;
      case I_ADD:
        *p += ldg(q);
        break;
      case I_SUB:
        *p -= ldg(q);
        break;
    }
  }
}

template <typename T>
void DoInplaceOp(const Device& d, InplaceOpType op, const Tensor& i,
                 const Tensor& v, Tensor* y) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinplace_ops_functor_gpuDTcuDTcc mht_3(mht_3_v, 288, "", "./tensorflow/core/kernels/inplace_ops_functor_gpu.cu.cc", "DoInplaceOp");

  const int64 nelem = v.NumElements();
  GpuLaunchConfig cfg = GetGpuLaunchConfig(nelem, d);
  auto Ty = y->flat_outer_dims<T>();
  const int64 nrows = Ty.dimension(0);
  const int64 ncols = Ty.dimension(1);
  const int64 n = i.NumElements();
  const T* src = v.flat<T>().data();
  // TODO(sjhwang): Check that first dimension fits in int32 range.
  const int32* rowids = i.flat<int32>().data();
  T* dst = y->flat<T>().data();
  switch (op) {
    case I_UPDATE:
      TF_CHECK_OK(GpuLaunchKernel(DoInplaceOpKernel<T, I_UPDATE>,
                                  cfg.block_count, cfg.thread_per_block, 0,
                                  d.stream(), cfg.virtual_thread_count, nrows,
                                  ncols, n, src, rowids, dst));
      break;
    case I_ADD:
      TF_CHECK_OK(GpuLaunchKernel(DoInplaceOpKernel<T, I_ADD>, cfg.block_count,
                                  cfg.thread_per_block, 0, d.stream(),
                                  cfg.virtual_thread_count, nrows, ncols, n,
                                  src, rowids, dst));
      break;
    case I_SUB:
      TF_CHECK_OK(GpuLaunchKernel(DoInplaceOpKernel<T, I_SUB>, cfg.block_count,
                                  cfg.thread_per_block, 0, d.stream(),
                                  cfg.virtual_thread_count, nrows, ncols, n,
                                  src, rowids, dst));
      break;
  }
}

template <bool>
void DoInplaceOp(const Device& d, InplaceOpType op, const Tensor& i,
                 const Tensor& v, Tensor* y) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinplace_ops_functor_gpuDTcuDTcc mht_4(mht_4_v, 326, "", "./tensorflow/core/kernels/inplace_ops_functor_gpu.cu.cc", "DoInplaceOp");

  const int64 nelem = v.NumElements();
  GpuLaunchConfig cfg = GetGpuLaunchConfig(nelem, d);
  auto Ty = y->flat_outer_dims<bool>();
  const int64 nrows = Ty.dimension(0);
  const int64 ncols = Ty.dimension(1);
  const int64 n = i.NumElements();
  const bool* src = v.flat<bool>().data();
  // TODO(sjhwang): Check that first dimension fits in int32 range.
  const int32* rowids = i.flat<int32>().data();
  bool* dst = y->flat<bool>().data();
  if (op == I_UPDATE) {
    TF_CHECK_OK(GpuLaunchKernel(DoInplaceOpKernel<bool, I_UPDATE>,
                                cfg.block_count, cfg.thread_per_block, 0,
                                d.stream(), cfg.virtual_thread_count, nrows,
                                ncols, n, src, rowids, dst));
  }
}

template <>
Status DoInplace(const Device& d, InplaceOpType op, const Tensor& i,
                 const Tensor& v, Tensor* y) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinplace_ops_functor_gpuDTcuDTcc mht_5(mht_5_v, 350, "", "./tensorflow/core/kernels/inplace_ops_functor_gpu.cu.cc", "DoInplace");

  CHECK_EQ(v.dtype(), y->dtype());
  switch (v.dtype()) {
#define CASE(type)                     \
  case DataTypeToEnum<type>::value:    \
    DoInplaceOp<type>(d, op, i, v, y); \
    break;

    CASE(bool)
    CASE(float)
    CASE(double)
    CASE(Eigen::half)
    CASE(int64)
#undef CASE
    default:
      return errors::InvalidArgument("Unsupported data type: ",
                                     DataTypeString(v.dtype()));
  }
  return Status::OK();
}

template <>
Status DoCopy(const Device& d, const Tensor& x, Tensor* y) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSinplace_ops_functor_gpuDTcuDTcc mht_6(mht_6_v, 375, "", "./tensorflow/core/kernels/inplace_ops_functor_gpu.cu.cc", "DoCopy");

  CHECK_EQ(x.dtype(), y->dtype());
  switch (x.dtype()) {
#define CASE(type)                              \
  case DataTypeToEnum<type>::value:             \
    y->flat<type>().device(d) = x.flat<type>(); \
    break;

    CASE(bool)
    CASE(float)
    CASE(double)
    CASE(Eigen::half)
    CASE(complex64)
    CASE(complex128)
    CASE(int64)
#undef CASE
    default:
      return errors::InvalidArgument("Unsupported dtype: ",
                                     DataTypeString(x.dtype()));
  }
  return Status::OK();
}

}  // end namespace functor
}  // namespace tensorflow
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
