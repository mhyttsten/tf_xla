/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_WHERE_OP_GPU_CU_H_
#define TENSORFLOW_CORE_KERNELS_WHERE_OP_GPU_CU_H_
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
class MHTracer_DTPStensorflowPScorePSkernelsPSwhere_op_gpuDTcuDTh {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSwhere_op_gpuDTcuDTh(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSwhere_op_gpuDTcuDTh() {
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

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/gpu_prim.h"
#include "tensorflow/core/kernels/where_op.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace functor {

template <int NDIM, typename TIndex>
__global__ void PropagateWhereIndicesKernel(
    const TIndex output_rows, const typename Eigen::array<TIndex, NDIM> strides,
    int64* __restrict__ output) {
  // TODO(ebrevdo): Use a multi-dimensional loop, increasing the
  // dimensions of individual indices manually, instead of relying on
  // a scalar loop variable and using integer division.
  GPU_1D_KERNEL_LOOP(i, output_rows) {
    TIndex index_value = ldg(output + NDIM * i);
#pragma unroll
    for (int c = 0; c < NDIM; ++c) {
      *(output + NDIM * i + c) = index_value / strides[c];
      index_value %= strides[c];
    }
  }
}

namespace {

template <typename T>
struct IsNonzero {
  EIGEN_DEVICE_FUNC IsNonzero() : zero(T(0)) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSwhere_op_gpuDTcuDTh mht_0(mht_0_v, 229, "", "./tensorflow/core/kernels/where_op_gpu.cu.h", "IsNonzero");
}
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE bool operator()(const T& x) const {
    return (x != zero);
  }
  const T zero;
};

template <typename T, typename TIndex>
struct CubDeviceReduceCount {
  gpuError_t operator()(void* d_temp_storage, size_t& temp_storage_bytes,
                        const T* d_in, TIndex* d_out, int num_items,
                        gpuStream_t stream = 0,
                        bool debug_synchronous = false) {
    IsNonzero<T> is_nonzero;
    gpuprim::TransformInputIterator<bool, IsNonzero<T>, const T*>
        is_nonzero_iter(d_in, is_nonzero);
    return gpuprim::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes,
                                      is_nonzero_iter, d_out, num_items, stream,
                                      debug_synchronous);
  }
};

template <typename TIndex>
struct CubDeviceReduceCount<bool, TIndex> {
  gpuError_t operator()(void* d_temp_storage, size_t& temp_storage_bytes,
                        const bool* d_in, TIndex* d_out, int num_items,
                        gpuStream_t stream = 0,
                        bool debug_synchronous = false) {
    return gpuprim::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in,
                                      d_out, num_items, stream,
                                      debug_synchronous);
  }
};

template <typename T, typename TIndex, typename OutputIterator,
          bool IsConvertibleToBool>
struct CubDeviceSelectFlaggedCounter;

template <typename T, typename TIndex, typename OutputIterator>
struct CubDeviceSelectFlaggedCounter<T, TIndex, OutputIterator,
                                     false /*IsConvertibleToBool*/> {
  gpuError_t operator()(void* d_temp_storage, size_t& temp_storage_bytes,
                        const T* d_flags, OutputIterator d_out,
                        TIndex* d_num_selected_out, int num_items,
                        gpuStream_t stream = 0,
                        bool debug_synchronous = false) {
    gpuprim::CountingInputIterator<TIndex> select_counter(0);
    IsNonzero<T> is_nonzero;
    gpuprim::TransformInputIterator<bool, IsNonzero<T>, const T*>
        is_nonzero_iter(d_flags, is_nonzero);
    return gpuprim::DeviceSelect::Flagged(
        d_temp_storage, temp_storage_bytes, select_counter /*d_in*/,
        is_nonzero_iter /*d_flags*/, d_out, d_num_selected_out, num_items,
        stream, debug_synchronous);
  }
};

template <typename T, typename TIndex, typename OutputIterator>
struct CubDeviceSelectFlaggedCounter<T, TIndex, OutputIterator,
                                     true /*IsConvertibleToBool*/> {
  gpuError_t operator()(void* d_temp_storage, size_t& temp_storage_bytes,
                        const T* d_flags, OutputIterator d_out,
                        TIndex* d_num_selected_out, int num_items,
                        gpuStream_t stream = 0,
                        bool debug_synchronous = false) {
    gpuprim::CountingInputIterator<TIndex> select_counter(0);
    return gpuprim::DeviceSelect::Flagged(
        d_temp_storage, temp_storage_bytes, select_counter /*d_in*/, d_flags,
        d_out, d_num_selected_out, num_items, stream, debug_synchronous);
  }
};

}  // namespace

template <typename T, typename TIndex>
struct NumTrue<GPUDevice, T, TIndex> {
  EIGEN_ALWAYS_INLINE static Status Compute(
      OpKernelContext* ctx, const GPUDevice& d,
      typename TTypes<T>::ConstFlat input,
      typename TTypes<TIndex>::UnalignedScalar num_true) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSwhere_op_gpuDTcuDTh mht_1(mht_1_v, 311, "", "./tensorflow/core/kernels/where_op_gpu.cu.h", "Compute");

    const auto& cu_stream = GetGpuStream(ctx);

    std::size_t temp_storage_bytes = 0;
    const T* input_data = input.data();
    TIndex* num_true_data = num_true.data();

    // TODO(ebrevdo): sum doesn't work; perhaps need a different
    // iterator?
    auto reducer = CubDeviceReduceCount<T, TIndex>();
    auto first_success = reducer(/*temp_storage*/ nullptr, temp_storage_bytes,
                                 /*d_in*/ input_data,
                                 /*d_out*/ num_true_data,
                                 /*num_items*/ input.size(),
                                 /*stream*/ cu_stream);

    if (first_success != gpuSuccess) {
      return errors::Internal(
          "WhereOp: Could not launch gpuprim::DeviceReduce::Sum to calculate "
          "temp_storage_bytes, status: ",
          GpuGetErrorString(first_success));
    }

    Tensor temp_storage;
    TF_RETURN_IF_ERROR(ctx->allocate_temp(
        DT_INT8, TensorShape({static_cast<int64_t>(temp_storage_bytes)}),
        &temp_storage));

    auto second_success = reducer(
        /*temp_storage*/ temp_storage.flat<int8>().data(), temp_storage_bytes,
        /*d_in*/ input_data,
        /*d_out*/ num_true_data,
        /*num_items*/ input.size(),
        /*stream*/ cu_stream);

    if (second_success != gpuSuccess) {
      return errors::Internal(
          "WhereOp: Could not launch gpuprim::DeviceReduce::Sum to count "
          "number of true / nonzero indices.  temp_storage_bytes: ",
          temp_storage_bytes, ", status: ", GpuGetErrorString(second_success));
    }

    return Status::OK();
  }
};

#define NUMTRUE_GPU_FUNCTOR(T)                  \
  template struct NumTrue<GPUDevice, T, int32>; \
  template struct NumTrue<GPUDevice, T, int64>;

// We only need to declare the NumTrue functor once, but this file is
// included from where_op_gpu_impl_X.cu.cc for X=1,2,...
// Only declare for X = 1.
#if GPU_PROVIDED_DIM == 1

TF_CALL_WHERE_GPU_TYPES(NUMTRUE_GPU_FUNCTOR);

#endif  // GPU_PROVIDED_DIM == 1

#undef NUMTRUE_GPU_FUNCTOR

template <int NDIM>
class WhereOutputIterator {
 public:
  // Required iterator traits
  typedef WhereOutputIterator self_type;
  typedef std::ptrdiff_t difference_type;
  typedef void value_type;
  typedef void pointer;
  typedef int64& reference;

#if (THRUST_VERSION >= 100700)
  // Use Thrust's iterator categories so we can use these iterators in Thrust
  // 1.7 (or newer) methods
  typedef typename thrust::detail::iterator_facade_category<
      thrust::device_system_tag, thrust::random_access_traversal_tag,
      value_type,
      reference>::type iterator_category;  ///< The iterator category
#else
  typedef std::random_access_iterator_tag
      iterator_category;  ///< The iterator category
#endif  // THRUST_VERSION

  WhereOutputIterator(int64* ptr, const Eigen::DenseIndex max_row)
      : ptr_(ptr), max_row_(max_row) {}

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int64& operator[](int n) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSwhere_op_gpuDTcuDTh mht_2(mht_2_v, 400, "", "./tensorflow/core/kernels/where_op_gpu.cu.h", "lambda");

    // If the selection mechanism finds too many true values (because
    // the input tensor changed between allocation of output and now),
    // we may accidentally try to write past the allowable memory.  If
    // valid is false, then we don't do this.  Instead, we'll read off
    // the number of items found in Flagged()'s d_num_selected_out at
    // the end and confirm that it matches the number of rows of output.
    const bool valid = FastBoundsCheck(n, max_row_);
    return *(ptr_ + (valid ? (NDIM * n) : 0));
  }

 private:
  int64* ptr_;
  const Eigen::DenseIndex max_row_;
};

template <typename TIndex, typename T, int NDIM>
Eigen::array<TIndex, NDIM> CalculateStrides(
    typename TTypes<T, NDIM>::ConstTensor input) {
  const Eigen::DSizes<Eigen::DenseIndex, NDIM> dims = input.dimensions();
  Eigen::array<TIndex, NDIM> strides;
  EIGEN_STATIC_ASSERT((static_cast<int>(decltype(input)::Layout) ==
                       static_cast<int>(Eigen::RowMajor)),
                      INTERNAL_ERROR_INPUT_SHOULD_BE_ROWMAJOR);
  strides[NDIM - 1] = 1;
  for (int i = NDIM - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dims[i + 1];
  }
  return strides;
}

template <int NDIM, typename T, typename TIndex>
struct Where<GPUDevice, NDIM, T, TIndex> {
  EIGEN_ALWAYS_INLINE static Status Compute(
      OpKernelContext* ctx, const GPUDevice& d,
      typename TTypes<T, NDIM>::ConstTensor input,
      typename TTypes<int64_t>::Matrix output, TIndex* found_true_host) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSwhere_op_gpuDTcuDTh mht_3(mht_3_v, 439, "", "./tensorflow/core/kernels/where_op_gpu.cu.h", "Compute");

    if (output.dimension(0) == 0) {
      // Nothing to do.
      return Status::OK();
    }

    const auto& cu_stream = GetGpuStream(ctx);

    std::size_t temp_storage_bytes = 0;

    Tensor found_true_t;
    TF_RETURN_IF_ERROR(ctx->allocate_temp(DataTypeToEnum<TIndex>::v(),
                                          TensorShape({}), &found_true_t));
    TIndex* found_true_device = found_true_t.scalar<TIndex>().data();

    WhereOutputIterator<NDIM> output_iterator(
        output.data(),
        /* max_row */ output.dimension(0));

    typedef std::decay<T> DT;
    CubDeviceSelectFlaggedCounter<
        T, TIndex, decltype(output_iterator) /*OutputIterator*/,
        std::is_convertible<DT, bool>::value /*IsConvertibleToBool*/>
        counter;
    auto first_success = counter(/*temp_storage*/ nullptr, temp_storage_bytes,
                                 /*d_flags*/ input.data(),
                                 /*d_out*/ output_iterator,
                                 /*d_num_selected_out*/ found_true_device,
                                 /*num_items*/ input.size(),
                                 /*stream*/ cu_stream);
    if (first_success != gpuSuccess) {
      return errors::Internal(
          "WhereOp: Could not launch gpuprim::DeviceSelect::Flagged to "
          "calculate "
          "temp_storage_bytes, status: ",
          GpuGetErrorString(first_success));
    }

    Tensor temp_storage;
    TF_RETURN_IF_ERROR(ctx->allocate_temp(
        DT_INT8, TensorShape({static_cast<int64_t>(temp_storage_bytes)}),
        &temp_storage));

    auto second_success = counter(
        /*temp_storage*/ temp_storage.flat<int8>().data(), temp_storage_bytes,
        /*d_flags*/ input.data(),
        /*d_out*/ output_iterator,
        /*d_num_selected_out*/ found_true_device,
        /*num_items*/ input.size(),
        /*stream*/ cu_stream);

    if (second_success != gpuSuccess) {
      return errors::Internal(
          "WhereOp: Could not launch gpuprim::DeviceSelect::Flagged to copy "
          "indices out, status: ",
          GpuGetErrorString(second_success));
    }

    // TODO(ebrevdo): Find a way to synchronously copy back data from
    // found_true_device to *found_true_host.

    const Eigen::array<TIndex, NDIM> strides =
        CalculateStrides<TIndex, T, NDIM>(input);
    const TIndex output_rows = output.dimension(0);
    GpuLaunchConfig config = GetGpuLaunchConfig(output_rows, d);
    TF_CHECK_OK(GpuLaunchKernel(PropagateWhereIndicesKernel<NDIM, TIndex>,
                                config.block_count, config.thread_per_block, 0,
                                d.stream(), output_rows, strides,
                                output.data()));

    return Status::OK();
  }
};

#define DECLARE_GPU_SPEC_INDEX(Dims, T, TIndex) \
  template struct Where<GPUDevice, Dims, T, TIndex>

#define DECLARE_GPU_SPEC(T)                           \
  DECLARE_GPU_SPEC_INDEX(GPU_PROVIDED_DIM, T, int32); \
  DECLARE_GPU_SPEC_INDEX(GPU_PROVIDED_DIM, T, int64)

TF_CALL_WHERE_GPU_TYPES(DECLARE_GPU_SPEC);

#undef DECLARE_GPU_SPEC
#undef DECLARE_GPU_SPEC_INDEX

}  // namespace functor

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#endif  // TENSORFLOW_CORE_KERNELS_WHERE_OP_GPU_CU_H_
