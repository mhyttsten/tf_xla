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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparse_split_op_gpuDTcuDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_split_op_gpuDTcuDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparse_split_op_gpuDTcuDTcc() {
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
#include "tensorflow/core/kernels/sparse_split_op.h"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_solvers.h"  // For ScratchSpace

#if GOOGLE_CUDA
#include "tensorflow/stream_executor/cuda/cuda_activation.h"
using stream_executor::cuda::ScopedActivateExecutorContext;
#elif TENSORFLOW_USE_ROCM
#include "tensorflow/stream_executor/rocm/rocm_activation.h"
using stream_executor::rocm::ScopedActivateExecutorContext;
#endif

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

namespace {

template <typename Index>
inline __device__ Index GetSliceIndex(const Index index, const Index split_size,
                                      const Index residual) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_split_op_gpuDTcuDTcc mht_0(mht_0_v, 219, "", "./tensorflow/core/kernels/sparse_split_op_gpu.cu.cc", "GetSliceIndex");

  if (residual == 0) return index / split_size;
  const Index offset = residual * (split_size + Index(1));
  if (index < offset) {
    return index / (split_size + Index(1));
  } else {
    return residual + ((index - offset) / split_size);
  }
}

template <typename Index>
inline __device__ Index GetDimensionInSlice(const Index index,
                                            const Index split_size,
                                            const Index residual) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_split_op_gpuDTcuDTcc mht_1(mht_1_v, 235, "", "./tensorflow/core/kernels/sparse_split_op_gpu.cu.cc", "GetDimensionInSlice");

  if (residual == 0) return index % split_size;
  const Index offset = residual * (split_size + 1);
  if (index < offset) {
    return index % (split_size + 1);
  } else {
    return (index - offset) % split_size;
  }
}

template <typename Index>
inline Index GetSliceShape(const Index slice_index, const Index split_size,
                           const Index residual) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_split_op_gpuDTcuDTcc mht_2(mht_2_v, 250, "", "./tensorflow/core/kernels/sparse_split_op_gpu.cu.cc", "GetSliceShape");

  if (residual == 0) return split_size;
  if (slice_index < residual) {
    return split_size + 1;
  } else {
    return split_size;
  }
}

template <typename Index>
struct SliceIndexer {
  SliceIndexer(const Index split_dim_size, const Index num_split)
      : split_size_(split_dim_size / num_split),
        residual_(split_dim_size % num_split) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_split_op_gpuDTcuDTcc mht_3(mht_3_v, 266, "", "./tensorflow/core/kernels/sparse_split_op_gpu.cu.cc", "SliceIndexer");
}

  inline __device__ Index GetSliceIndex(const Index index) const {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_split_op_gpuDTcuDTcc mht_4(mht_4_v, 271, "", "./tensorflow/core/kernels/sparse_split_op_gpu.cu.cc", "GetSliceIndex");

    return tensorflow::functor::GetSliceIndex(index, split_size_, residual_);
  }

  inline __device__ Index GetIndexInSlice(const Index index) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_split_op_gpuDTcuDTcc mht_5(mht_5_v, 278, "", "./tensorflow/core/kernels/sparse_split_op_gpu.cu.cc", "GetIndexInSlice");

    return GetDimensionInSlice(index, split_size_, residual_);
  }

  inline __host__ Index GetSliceSize(const Index slice_index) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_split_op_gpuDTcuDTcc mht_6(mht_6_v, 285, "", "./tensorflow/core/kernels/sparse_split_op_gpu.cu.cc", "GetSliceSize");

    return GetSliceShape(slice_index, split_size_, residual_);
  }

 private:
  const Index split_size_;
  const Index residual_;
};

template <typename Index>
__global__ void SparseSplitSliceIndexesKernel(
    Index input_nnz, int rank, int axis, SliceIndexer<Index> slice_indexer,
    const Index* __restrict__ input_indices, int* __restrict__ slice_indexes) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_split_op_gpuDTcuDTcc mht_7(mht_7_v, 300, "", "./tensorflow/core/kernels/sparse_split_op_gpu.cu.cc", "SparseSplitSliceIndexesKernel");

  for (Index input_nz : GpuGridRangeX<Index>(input_nnz)) {
    slice_indexes[input_nz] =
        slice_indexer.GetSliceIndex(input_indices[input_nz * rank + axis]);
  }
}

template <typename Index>
Status LaunchSparseSplitSliceIndexesKernel(const GPUDevice& device,
                                           Index input_nnz, int num_split,
                                           int rank, int axis,
                                           SliceIndexer<Index> slice_indexer,
                                           const Index* input_indices,
                                           int* slice_indexes) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_split_op_gpuDTcuDTcc mht_8(mht_8_v, 316, "", "./tensorflow/core/kernels/sparse_split_op_gpu.cu.cc", "LaunchSparseSplitSliceIndexesKernel");

  if (input_nnz == 0) return Status::OK();
  GpuLaunchConfig config = GetGpuLaunchConfig(
      input_nnz, device, &SparseSplitSliceIndexesKernel<Index>,
      /*dynamic_shared_memory_size=*/0, /*block_size_limit=*/0);
  return GpuLaunchKernel(SparseSplitSliceIndexesKernel<Index>,
                         config.block_count, config.thread_per_block, 0,
                         device.stream(), input_nnz, rank, axis, slice_indexer,
                         input_indices, slice_indexes);
}

template <typename Index>
__global__ void SparseSplitFindSliceEndsKernel(
    Index input_nnz, int num_split,
    const int* __restrict__ sorted_slice_indexes,
    Index* __restrict__ slice_ends) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_split_op_gpuDTcuDTcc mht_9(mht_9_v, 334, "", "./tensorflow/core/kernels/sparse_split_op_gpu.cu.cc", "SparseSplitFindSliceEndsKernel");

  for (int slice_index : GpuGridRangeX<int>(num_split)) {
    slice_ends[slice_index] =
        gpu_helper::upper_bound(sorted_slice_indexes, input_nnz, slice_index);
  }
}

template <typename Index>
Status LaunchSparseSplitFindSliceEndsKernel(const GPUDevice& device,
                                            Index input_nnz, int num_split,
                                            const int* sorted_slice_indexes,
                                            Index* slice_ends) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_split_op_gpuDTcuDTcc mht_10(mht_10_v, 348, "", "./tensorflow/core/kernels/sparse_split_op_gpu.cu.cc", "LaunchSparseSplitFindSliceEndsKernel");

  GpuLaunchConfig config = GetGpuLaunchConfig(
      num_split, device, &SparseSplitFindSliceEndsKernel<Index>,
      /*dynamic_shared_memory_size=*/0, /*block_size_limit=*/0);
  return GpuLaunchKernel(SparseSplitFindSliceEndsKernel<Index>,
                         config.block_count, config.thread_per_block, 0,
                         device.stream(), input_nnz, num_split,
                         sorted_slice_indexes, slice_ends);
}

// Scatters (and offsets) input indices and values to the outputs.
template <typename T, typename Index>
__global__ void SparseSplitScatterKernel(
    Index input_nnz, int rank, int axis, SliceIndexer<Index> slice_indexer,
    const Index* __restrict__ sort_permutation,
    const Index* __restrict__ slice_ends,
    const Index* __restrict__ input_indices, const T* __restrict__ input_values,
    GpuDeviceArrayStruct<Index*> output_indices_data,
    GpuDeviceArrayStruct<T*> output_values_data) {
  Index* __restrict__* __restrict__ output_indices =
      GetGpuDeviceArrayOnDevice(&output_indices_data);
  T* __restrict__* __restrict__ output_values =
      GetGpuDeviceArrayOnDevice(&output_values_data);

  for (Index sorted_input_nz : GpuGridRangeX<Index>(input_nnz)) {
    Index input_nz = sort_permutation[sorted_input_nz];
    int slice_index =
        slice_indexer.GetSliceIndex(input_indices[input_nz * rank + axis]);
    Index slice_nz =
        sorted_input_nz -
        (slice_index == 0 ? Index(0) : slice_ends[slice_index - 1]);
    output_values[slice_index][slice_nz] = input_values[input_nz];
    for (int dim = 0; dim < rank; ++dim) {
      Index input_index = input_indices[input_nz * rank + dim];
      output_indices[slice_index][slice_nz * rank + dim] =
          (dim == axis) ? slice_indexer.GetIndexInSlice(input_index)
                        : input_index;
    }
  }
}

template <typename T, typename Index>
Status LaunchSparseSplitScatterKernel(
    const GPUDevice& device, Index input_nnz, int rank, int axis,
    SliceIndexer<Index> slice_indexer, const Index* sort_permutation,
    const Index* slice_ends, const Index* input_indices, const T* input_values,
    GpuDeviceArrayStruct<Index*> output_indices_data,
    GpuDeviceArrayStruct<T*> output_values_data) {
  if (input_nnz == 0) return Status::OK();
  GpuLaunchConfig config = GetGpuLaunchConfig(
      input_nnz, device, &SparseSplitScatterKernel<T, Index>,
      /*dynamic_shared_memory_size=*/0, /*block_size_limit=*/0);
  return GpuLaunchKernel(SparseSplitScatterKernel<T, Index>, config.block_count,
                         config.thread_per_block, 0, device.stream(), input_nnz,
                         rank, axis, slice_indexer, sort_permutation,
                         slice_ends, input_indices, input_values,
                         output_indices_data, output_values_data);
}

}  // namespace

template <typename T>
struct SparseSplitFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* context, const Tensor& input_indices,
                  const Tensor& input_values, const TensorShape& dense_shape,
                  const int64_t axis, const int num_split,
                  typename AsyncOpKernel::DoneCallback done) {
    using Index = int64_t;

    const Index input_nnz = input_indices.dim_size(0);
    const Index split_dim_size = dense_shape.dim_size(static_cast<int>(axis));
    const int rank = dense_shape.dims();

    const Index* input_indices_ptr = input_indices.matrix<Index>().data();
    const T* input_values_ptr = input_values.vec<T>().data();

    const SliceIndexer<Index> slice_indexer(split_dim_size, num_split);

    const GPUDevice& device = context->eigen_gpu_device();
    se::Stream* stream = context->op_device_context()->stream();
    OP_REQUIRES_ASYNC(context, stream,
                      errors::Internal("No GPU stream available."), done);

    Tensor sort_permutation;
    OP_REQUIRES_OK_ASYNC(
        context,
        context->allocate_temp(DataTypeToEnum<Index>::value,
                               TensorShape({input_nnz}), &sort_permutation),
        done);
    Index* sort_permutation_ptr = sort_permutation.vec<Index>().data();

    Tensor slice_ends;
    OP_REQUIRES_OK_ASYNC(
        context,
        context->allocate_temp(DataTypeToEnum<Index>::value,
                               TensorShape({num_split}), &slice_ends),
        done);
    Index* slice_ends_ptr = slice_ends.vec<Index>().data();

    // First we compute the slice index for each element, sort them, and use a
    // binary search to find the end of each slice.
    {
      Tensor slice_indexes;
      OP_REQUIRES_OK_ASYNC(
          context,
          context->allocate_temp(DT_INT32, TensorShape({input_nnz}),
                                 &slice_indexes),
          done);
      int* slice_indexes_ptr = slice_indexes.vec<int>().data();

      OP_REQUIRES_OK_ASYNC(
          context,
          LaunchSparseSplitSliceIndexesKernel(
              device, input_nnz, num_split, rank, axis, slice_indexer,
              input_indices_ptr, slice_indexes_ptr),
          done);

      Tensor sorted_slice_indexes;
      OP_REQUIRES_OK_ASYNC(
          context,
          context->allocate_temp(DT_INT32, TensorShape({num_split}),
                                 &sorted_slice_indexes),
          done);
      int* sorted_slice_indexes_ptr = sorted_slice_indexes.vec<int>().data();
      OP_REQUIRES_OK_ASYNC(
          context,
          GpuRadixSort(context, /*size=*/input_nnz,
                       /*keys_in=*/slice_indexes_ptr,
                       /*keys_out=*/sorted_slice_indexes_ptr,
                       /*indices_in=*/static_cast<const Index*>(nullptr),
                       /*indices_out=*/sort_permutation_ptr,
                       /*num_bits=*/Log2Ceiling(num_split)),
          done);

      OP_REQUIRES_OK_ASYNC(context,
                           LaunchSparseSplitFindSliceEndsKernel(
                               device, input_nnz, num_split,
                               sorted_slice_indexes_ptr, slice_ends_ptr),
                           done);
    }

    // Copy the slice ends to the host so that we can compute the output shapes.
    ScratchSpace<Index> slice_ends_host(context, num_split, /*on_host=*/true);
    OP_REQUIRES_ASYNC(
        context,
        stream
            ->ThenMemcpy(
                slice_ends_host.mutable_data(),
                se::DeviceMemoryBase(slice_ends_ptr,
                                     num_split * sizeof(*slice_ends_ptr)),
                num_split * sizeof(*slice_ends_ptr))
            .ok(),
        errors::Internal("Failed to copy slice_ends to host"), done);

    auto async_finish_computation =
        [this, context, input_nnz, num_split, rank, axis, dense_shape,
         slice_indexer, slice_ends_host, input_indices, input_indices_ptr,
         input_values, input_values_ptr, sort_permutation, sort_permutation_ptr,
         slice_ends, slice_ends_ptr, done]() -> void {
      // Ensure that within the callback, the proper GPU settings are
      // configured.
      auto stream = context->op_device_context()->stream();
      ScopedActivateExecutorContext scoped_activation{stream->parent()};

      GpuDeviceArrayOnHost<Index*> output_indices(context, num_split);
      GpuDeviceArrayOnHost<T*> output_values(context, num_split);
      OP_REQUIRES_OK_ASYNC(
          context,
          AllocateOutputs(context, num_split, rank, axis, dense_shape,
                          slice_indexer, slice_ends_host.data(),
                          &output_indices, &output_values),
          done);

      const GPUDevice& device = context->eigen_device<GPUDevice>();

      // Finally, scatter (and offset) input indices and values to the outputs.
      OP_REQUIRES_OK_ASYNC(
          context,
          LaunchSparseSplitScatterKernel(
              device, input_nnz, rank, axis, slice_indexer,
              sort_permutation_ptr, slice_ends_ptr, input_indices_ptr,
              input_values_ptr, output_indices.data(), output_values.data()),
          done);

      done();
    };

    context->device()
        ->tensorflow_accelerator_device_info()
        ->event_mgr->ThenExecute(stream, async_finish_computation);
  }

 private:
  template <typename Index>
  Status AllocateOutputs(OpKernelContext* context, int num_split, int rank,
                         int axis, const TensorShape& dense_shape,
                         const SliceIndexer<Index>& slice_indexer,
                         const Index* slice_ends_host,
                         GpuDeviceArrayOnHost<Index*>* output_indices,
                         GpuDeviceArrayOnHost<T*>* output_values) const {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_split_op_gpuDTcuDTcc mht_11(mht_11_v, 550, "", "./tensorflow/core/kernels/sparse_split_op_gpu.cu.cc", "AllocateOutputs");

    TF_RETURN_IF_ERROR(output_indices->Init());
    TF_RETURN_IF_ERROR(output_values->Init());
    for (int slice_index = 0; slice_index < num_split; ++slice_index) {
      Index slice_nnz =
          slice_ends_host[slice_index] -
          (slice_index == 0 ? Index(0) : slice_ends_host[slice_index - 1]);
      Tensor* output_inds = nullptr;
      TF_RETURN_IF_ERROR(context->allocate_output(
          slice_index, {slice_nnz, rank}, &output_inds));
      output_indices->Set(slice_index, output_inds->matrix<Index>().data());
      Tensor* output_vals = nullptr;
      TF_RETURN_IF_ERROR(context->allocate_output(num_split + slice_index,
                                                  {slice_nnz}, &output_vals));
      output_values->Set(slice_index, output_vals->vec<T>().data());
      Tensor* output_shape = nullptr;
      TF_RETURN_IF_ERROR(context->allocate_output(num_split * 2 + slice_index,
                                                  {rank}, &output_shape));
      for (int dim = 0; dim < rank; ++dim) {
        output_shape->vec<int64_t>()(dim) =
            (dim == axis) ? slice_indexer.GetSliceSize(slice_index)
                          : dense_shape.dim_size(dim);
      }
    }
    TF_RETURN_IF_ERROR(output_indices->Finalize());
    TF_RETURN_IF_ERROR(output_values->Finalize());
    return Status::OK();
  }
};

#define DEFINE_SPARSE_SPLIT_FUNCTOR(T) \
  template struct SparseSplitFunctor<GPUDevice, T>;
TF_CALL_POD_TYPES(DEFINE_SPARSE_SPLIT_FUNCTOR);

#undef DEFINE_SPARSE_SPLIT_FUNCTOR

}  // namespace functor

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
