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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparse_concat_op_gpuDTcuDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_concat_op_gpuDTcuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparse_concat_op_gpuDTcuDTcc() {
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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/gpu_device_array.h"
#include "tensorflow/core/kernels/gpu_device_array_gpu.h"
#include "tensorflow/core/kernels/gpu_prim_helpers.h"
#include "tensorflow/core/kernels/sparse_concat_op.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

namespace {

template <typename T>
__global__ void SparseConcatKernel(
    int64 output_nnz, int rank, int concat_dim, bool need_to_sort,
    GpuDeviceArrayStruct<const int64*> ind_ptrs_data,
    GpuDeviceArrayStruct<const T*> val_ptrs_data,
    GpuDeviceArrayStruct<int64_t> nnz_scan_data,
    GpuDeviceArrayStruct<int64_t> concat_size_scan_data,
    GpuDeviceArrayStruct<int64_t> output_shape_data,
    int64* __restrict__ output_inds, T* __restrict__ output_vals,
    int64* __restrict__ output_flat_inds) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_concat_op_gpuDTcuDTcc mht_0(mht_0_v, 217, "", "./tensorflow/core/kernels/sparse_concat_op_gpu.cu.cc", "SparseConcatKernel");

  const int64* __restrict__* __restrict__ ind_ptrs =
      GetGpuDeviceArrayOnDevice(&ind_ptrs_data);
  const T* __restrict__* __restrict__ val_ptrs =
      GetGpuDeviceArrayOnDevice(&val_ptrs_data);
  const int64* __restrict__ nnz_scan =
      GetGpuDeviceArrayOnDevice(&nnz_scan_data);
  const int64* __restrict__ concat_size_scan =
      GetGpuDeviceArrayOnDevice(&concat_size_scan_data);
  const int64* __restrict__ output_shape =
      GetGpuDeviceArrayOnDevice(&output_shape_data);
  const int64 num_inputs = ind_ptrs_data.size;

  for (int64 nz : GpuGridRangeX<int64_t>(output_nnz)) {
    const int64 input_num =
        gpu_helper::upper_bound<int64_t>(nnz_scan, num_inputs, nz) - 1;
    const int64 input_nz = nz - nnz_scan[input_num];
    const int64 ind_offset = concat_size_scan[input_num];
    if (!need_to_sort) {
      output_vals[nz] = val_ptrs[input_num][input_nz];
    }
    int64 flat_ind = 0;
    for (int j = 0; j < rank; ++j) {
      const int64 output_ind = ind_ptrs[input_num][input_nz * rank + j] +
                               (j == concat_dim ? ind_offset : 0);
      if (!need_to_sort) {
        output_inds[nz * rank + j] = output_ind;
      } else {
        flat_ind = flat_ind * output_shape[j] + output_ind;
        output_flat_inds[nz] = flat_ind;
      }
    }
  }
}

template <typename T>
__global__ void SparseConcatPermuteKernel(
    int64 output_nnz, int rank, GpuDeviceArrayStruct<const T*> val_ptrs_data,
    GpuDeviceArrayStruct<int64_t> nnz_scan_data,
    GpuDeviceArrayStruct<int64_t> output_shape_data,
    const int64* __restrict__ output_flat_inds,
    const int64* __restrict__ permutation, int64* __restrict__ output_inds,
    T* __restrict__ output_vals) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_concat_op_gpuDTcuDTcc mht_1(mht_1_v, 262, "", "./tensorflow/core/kernels/sparse_concat_op_gpu.cu.cc", "SparseConcatPermuteKernel");

  const T* __restrict__* __restrict__ val_ptrs =
      GetGpuDeviceArrayOnDevice(&val_ptrs_data);
  const int64* __restrict__ nnz_scan =
      GetGpuDeviceArrayOnDevice(&nnz_scan_data);
  const int64* __restrict__ output_shape =
      GetGpuDeviceArrayOnDevice(&output_shape_data);
  const int64 num_inputs = val_ptrs_data.size;

  for (int64 nz : GpuGridRangeX<int64_t>(output_nnz)) {
    const int64 permuted_nz = permutation[nz];
    const int64 input_num =
        gpu_helper::upper_bound<int64_t>(nnz_scan, num_inputs, permuted_nz) - 1;
    const int64 input_nz = permuted_nz - nnz_scan[input_num];
    output_vals[nz] = val_ptrs[input_num][input_nz];
    int64 output_flat_ind = output_flat_inds[permuted_nz];
    for (int j = rank - 1; j >= 0; --j) {
      const int64 output_dim_size = output_shape[j];
      output_inds[nz * rank + j] = output_flat_ind % output_dim_size;
      output_flat_ind /= output_dim_size;
    }
  }
}

}  // namespace

template <typename T>
struct SparseConcatFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* context, const OpInputList& inds,
                  const OpInputList& vals, const OpInputList& shapes,
                  int concat_dim) {
    const int N = inds.size();
    const TensorShape input_shape0(shapes[0].vec<int64_t>());
    const int rank = input_shape0.dims();

    // The input non-zeros are assumed to be sorted by increasing dimension
    // number (i.e., row-major order), so if the concatenation is along the
    // first dimension then they remain in order and we can directly compute the
    // output indices and values. To concatenate along other dimensions, we
    // first compute the flattened (1D) row-major output indices, then sort
    // these to obtain the required permutation, and finally gather the permuted
    // input values.

    GpuDeviceArrayOnHost<const int64*> ind_ptrs(context, N);
    GpuDeviceArrayOnHost<const T*> val_ptrs(context, N);
    GpuDeviceArrayOnHost<int64_t> nnz_scan(context, N + 1);
    GpuDeviceArrayOnHost<int64_t> concat_size_scan(context, N + 1);
    OP_REQUIRES_OK(context, ind_ptrs.Init());
    OP_REQUIRES_OK(context, val_ptrs.Init());
    OP_REQUIRES_OK(context, nnz_scan.Init());
    OP_REQUIRES_OK(context, concat_size_scan.Init());
    int64 nnz_sum = 0;
    int64 concat_size_sum = 0;
    nnz_scan.Set(0, nnz_sum);
    concat_size_scan.Set(0, concat_size_sum);
    for (int i = 0; i < N; ++i) {
      ind_ptrs.Set(i, inds[i].matrix<int64_t>().data());
      val_ptrs.Set(i, vals[i].vec<T>().data());
      nnz_sum += inds[i].dim_size(0);
      nnz_scan.Set(i + 1, nnz_sum);
      const TensorShape current_shape(shapes[i].vec<int64_t>());
      concat_size_sum += current_shape.dim_size(concat_dim);
      concat_size_scan.Set(i + 1, concat_size_sum);
    }
    OP_REQUIRES_OK(context, ind_ptrs.Finalize());
    OP_REQUIRES_OK(context, val_ptrs.Finalize());
    OP_REQUIRES_OK(context, nnz_scan.Finalize());
    OP_REQUIRES_OK(context, concat_size_scan.Finalize());
    const int64 output_nnz = nnz_sum;
    const int64 output_concat_size = concat_size_sum;

    const bool need_to_sort = concat_dim != 0;

    GpuDeviceArrayOnHost<int64_t> output_shape(context, rank);
    int64 output_dense_elements;
    if (need_to_sort) {
      OP_REQUIRES_OK(context, output_shape.Init());
      output_dense_elements = 1;
      for (int j = 0; j < rank; ++j) {
        int64 output_dim_size =
            j == concat_dim ? output_concat_size : input_shape0.dim_size(j);
        output_shape.Set(j, output_dim_size);
        output_dense_elements *= output_dim_size;
      }
      OP_REQUIRES_OK(context, output_shape.Finalize());
    }

    int64* output_inds_ptr = nullptr;
    T* output_vals_ptr = nullptr;
    int64* output_flat_inds_ptr = nullptr;
    Tensor output_flat_inds;
    if (need_to_sort) {
      // SparseConcatKernel will (only) produce output_flat_inds.
      OP_REQUIRES_OK(context,
                     context->allocate_temp(DT_INT64, TensorShape({output_nnz}),
                                            &output_flat_inds));
      output_flat_inds_ptr = output_flat_inds.vec<int64_t>().data();
    } else {
      OP_REQUIRES_OK(
          context, allocate_outputs(context, rank, output_nnz, &output_inds_ptr,
                                    &output_vals_ptr));
    }

    const GPUDevice& device = context->eigen_gpu_device();

    GpuLaunchConfig config = GetGpuLaunchConfig(
        output_nnz, device, &SparseConcatKernel<T>,
        /*dynamic_shared_memory_size=*/0, /*block_size_limit=*/0);
    OP_REQUIRES_OK(
        context, GpuLaunchKernel(
                     SparseConcatKernel<T>, config.block_count,
                     config.thread_per_block, 0, device.stream(), output_nnz,
                     rank, concat_dim, need_to_sort, ind_ptrs.data(),
                     val_ptrs.data(), nnz_scan.data(), concat_size_scan.data(),
                     (need_to_sort ? output_shape.data()
                                   : GpuDeviceArrayStruct<int64_t>()),
                     output_inds_ptr, output_vals_ptr, output_flat_inds_ptr));

    if (!need_to_sort) return;

    OP_REQUIRES_OK(context,
                   allocate_outputs(context, rank, output_nnz, &output_inds_ptr,
                                    &output_vals_ptr));

    Tensor permutation;
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DT_INT64, TensorShape({output_nnz}),
                                          &permutation));
    int64* permutation_ptr = permutation.vec<int64_t>().data();
    OP_REQUIRES_OK(
        context,
        GpuRadixSort(context, /*size=*/output_nnz,
                     /*keys_in=*/output_flat_inds_ptr,
                     /*keys_out=*/static_cast<int64*>(nullptr),
                     /*indices_in=*/static_cast<const int64*>(nullptr),
                     /*indices_out=*/permutation_ptr,
                     /*num_bits=*/Log2Ceiling64(output_dense_elements)));

    config = GetGpuLaunchConfig(
        output_nnz, device, &SparseConcatPermuteKernel<T>,
        /*dynamic_shared_memory_size=*/0, /*block_size_limit=*/0);
    OP_REQUIRES_OK(
        context,
        GpuLaunchKernel(SparseConcatPermuteKernel<T>, config.block_count,
                        config.thread_per_block, 0, device.stream(), output_nnz,
                        rank, val_ptrs.data(), nnz_scan.data(),
                        output_shape.data(), output_flat_inds_ptr,
                        permutation_ptr, output_inds_ptr, output_vals_ptr));
  }

 private:
  Status allocate_outputs(OpKernelContext* context, int rank, int64 output_nnz,
                          int64** output_inds_ptr, T** output_vals_ptr) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_concat_op_gpuDTcuDTcc mht_2(mht_2_v, 417, "", "./tensorflow/core/kernels/sparse_concat_op_gpu.cu.cc", "allocate_outputs");

    Tensor* output_inds = nullptr;
    TF_RETURN_IF_ERROR(context->allocate_output(
        0, TensorShape({output_nnz, rank}), &output_inds));
    *output_inds_ptr = output_inds->matrix<int64_t>().data();
    Tensor* output_vals = nullptr;
    TF_RETURN_IF_ERROR(
        context->allocate_output(1, TensorShape({output_nnz}), &output_vals));
    *output_vals_ptr = output_vals->vec<T>().data();
    return Status::OK();
  }
};

#define DEFINE_SPARSE_CONCAT_FUNCTOR(T) \
  template struct SparseConcatFunctor<GPUDevice, T>;
TF_CALL_POD_TYPES(DEFINE_SPARSE_CONCAT_FUNCTOR);

#undef DEFINE_SPARSE_CONCAT_FUNCTOR

}  // namespace functor

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
