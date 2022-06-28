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
class MHTracer_DTPStensorflowPScorePSkernelsPStile_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPStile_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPStile_opsDTcc() {
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

// See docs in ../ops/array_ops.cc.

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include <vector>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/type_index.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// Forward declarations of functors that will be defined in tile_ops_impl.h
namespace functor {
template <typename Device, typename T, typename Tmultiple>
struct Tile {
  void operator()(const Device& d, Tensor* out, const Tensor& in,
                  const gtl::ArraySlice<Tmultiple> broadcast_array) const;
};

template <typename Device, typename T, int NDIM>
struct TileGrad {
  void operator()(const Device& d, typename TTypes<T, NDIM>::Tensor out,
                  typename TTypes<T, NDIM>::ConstTensor in,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIM>& indices,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIM>& sizes,
                  bool first) const;
};

template <typename Device, typename T>
struct TileGrad<Device, T, 0> {
  void operator()(const Device& d, typename TTypes<T, 0>::Tensor out,
                  typename TTypes<T, 0>::ConstTensor in,
                  const Eigen::DSizes<Eigen::DenseIndex, 0>&,
                  const Eigen::DSizes<Eigen::DenseIndex, 0>&, bool first) const;
};

template <typename Device, typename T, int NDIM, int REDUCEDNDIM>
struct ReduceAndReshape {
  void operator()(
      const Device& d, typename TTypes<T, NDIM>::Tensor out,
      typename TTypes<T, NDIM>::ConstTensor in,
      const Eigen::DSizes<Eigen::DenseIndex, REDUCEDNDIM>& reduce_dim,
      const Eigen::DSizes<Eigen::DenseIndex, NDIM>& reshape_dim) const;
};

// Explicit instantiations are defined in tile_ops_{cpu,gpu}_impl.*,
// below are their declarations.

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
extern template struct Tile<GPUDevice, bool, int32>;
extern template struct Tile<GPUDevice, bool, int64_t>;
extern template struct Tile<GPUDevice, float, int32>;
extern template struct Tile<GPUDevice, float, int64_t>;
extern template struct Tile<GPUDevice, double, int32>;
extern template struct Tile<GPUDevice, double, int64_t>;
extern template struct Tile<GPUDevice, complex64, int32>;
extern template struct Tile<GPUDevice, complex64, int64_t>;
extern template struct Tile<GPUDevice, complex128, int32>;
extern template struct Tile<GPUDevice, complex128, int64_t>;
extern template struct Tile<GPUDevice, Eigen::half, int32>;
extern template struct Tile<GPUDevice, Eigen::half, int64_t>;
extern template struct Tile<GPUDevice, int16, int32>;
extern template struct Tile<GPUDevice, int16, int64_t>;
extern template struct Tile<GPUDevice, int32, int32>;
extern template struct Tile<GPUDevice, int32, int64_t>;
extern template struct Tile<GPUDevice, int64_t, int32>;
extern template struct Tile<GPUDevice, int64_t, int64_t>;
#define DECLARE_CUDA_DIM(T, NDIM)                      \
  extern template struct TileGrad<GPUDevice, T, NDIM>; \
  extern template struct ReduceAndReshape<GPUDevice, T, NDIM, 1>
#else  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define DECLARE_CUDA_DIM(T, NDIM)
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define DECLARE_TYPE(T)                             \
  extern template struct Tile<CPUDevice, T, int32>; \
  extern template struct Tile<CPUDevice, T, int64_t>;
TF_CALL_bool(DECLARE_TYPE);
TF_CALL_float(DECLARE_TYPE);
TF_CALL_bfloat16(DECLARE_TYPE);
TF_CALL_double(DECLARE_TYPE);
TF_CALL_uint8(DECLARE_TYPE);
TF_CALL_int32(DECLARE_TYPE);
TF_CALL_int16(DECLARE_TYPE);
TF_CALL_int64(DECLARE_TYPE);
TF_CALL_uint32(DECLARE_TYPE);
TF_CALL_uint64(DECLARE_TYPE);
TF_CALL_half(DECLARE_TYPE);
TF_CALL_complex64(DECLARE_TYPE);
TF_CALL_complex128(DECLARE_TYPE);
TF_CALL_tstring(DECLARE_TYPE);
TF_CALL_variant(DECLARE_TYPE);
#undef DECLARE_TYPE

#define DECLARE_DIM(T, NDIM)                           \
  DECLARE_CUDA_DIM(T, NDIM);                           \
  extern template struct TileGrad<CPUDevice, T, NDIM>; \
  extern template struct ReduceAndReshape<CPUDevice, T, NDIM, 1>;

#define DECLARE_TYPE(T) \
  DECLARE_DIM(T, 1)     \
  DECLARE_DIM(T, 2)     \
  DECLARE_DIM(T, 3)     \
  DECLARE_DIM(T, 4)     \
  DECLARE_DIM(T, 5)     \
  DECLARE_DIM(T, 6)     \
  DECLARE_DIM(T, 7)
TF_CALL_float(DECLARE_TYPE);
TF_CALL_bfloat16(DECLARE_TYPE);
TF_CALL_double(DECLARE_TYPE);
TF_CALL_int16(DECLARE_TYPE);
TF_CALL_int32(DECLARE_TYPE);
TF_CALL_int64(DECLARE_TYPE);
TF_CALL_half(DECLARE_TYPE);
TF_CALL_complex64(DECLARE_TYPE);
TF_CALL_complex128(DECLARE_TYPE);
#undef DECLARE_TYPE

#undef DECLARE_DIM
#undef DECLARE_CUDA_DIM

}  // namespace functor

// --------------------------------------------------------------------------
template <typename Device, typename Tmultiples>
class TileOp : public OpKernel {
 public:
  explicit TileOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStile_opsDTcc mht_0(mht_0_v, 330, "", "./tensorflow/core/kernels/tile_ops.cc", "TileOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStile_opsDTcc mht_1(mht_1_v, 335, "", "./tensorflow/core/kernels/tile_ops.cc", "Compute");

    const Tensor& input = context->input(0);
    const Tensor& multiples = context->input(1);

    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(multiples.shape()),
        errors::InvalidArgument("Expected multiples to be 1-D, but got shape ",
                                multiples.shape().DebugString()));
    OP_REQUIRES(context, input.dims() == multiples.NumElements(),
                errors::InvalidArgument(
                    "Expected multiples argument to be a vector of length ",
                    input.dims(), " but got length ", multiples.dim_size(0)));
    const int input_dims = input.dims();

    // Eigen doesn't support scalars on the GPU, so handle 0-D specially
    if (input_dims == 0) {
      context->set_output(0, input);
      return;
    }

    const gtl::ArraySlice<Tmultiples> multiples_array(
        multiples.flat<Tmultiples>().data(), input_dims);
    TensorShape output_shape;
    for (int i = 0; i < input_dims; ++i) {
      OP_REQUIRES(
          context, multiples_array[i] >= 0,
          errors::InvalidArgument("Expected multiples[", i, "] >= 0, but got ",
                                  multiples_array[i]));
      OP_REQUIRES_OK(context, output_shape.AddDimWithStatus(
                                  input.dim_size(i) * multiples_array[i]));
    }
    if (output_shape == input.shape()) {
      context->set_output(0, input);
      return;
    }
    Tensor* result = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &result));

    // If there's no output, there's nothing to do.
    if (output_shape.num_elements() == 0) return;

#define HANDLE_TYPE(DT)                               \
  if (context->input(0).dtype() == DT) {              \
    HandleCase<DT>(context, multiples_array, result); \
    return;                                           \
  }

#define HANDLE_TYPE_NAME(T) HANDLE_TYPE(DataTypeToEnum<T>::value)

    // Invoke macro using TF_CALL_* so type-filtering for platform applies.
    TF_CALL_bool(HANDLE_TYPE_NAME);
    TF_CALL_bfloat16(HANDLE_TYPE_NAME);
    TF_CALL_float(HANDLE_TYPE_NAME);
    TF_CALL_double(HANDLE_TYPE_NAME);
    TF_CALL_uint8(HANDLE_TYPE_NAME);
    TF_CALL_int8(HANDLE_TYPE_NAME);
    TF_CALL_int32(HANDLE_TYPE_NAME);
    TF_CALL_int16(HANDLE_TYPE_NAME);
    TF_CALL_int64(HANDLE_TYPE_NAME);
    TF_CALL_uint32(HANDLE_TYPE_NAME);
    TF_CALL_uint64(HANDLE_TYPE_NAME);
    TF_CALL_half(HANDLE_TYPE_NAME);
    TF_CALL_tstring(HANDLE_TYPE_NAME);  // when DEVICE=CPUDevice.
    TF_CALL_complex64(HANDLE_TYPE_NAME);
    TF_CALL_complex128(HANDLE_TYPE_NAME);
    TF_CALL_variant(HANDLE_TYPE_NAME);  // when DEVICE=CPUDevice

#undef HANDLE_TYPE_NAME
#undef HANDLE_TYPE

    OP_REQUIRES(
        context, false,
        errors::Unimplemented(
            "TileOp : The input data type is not supported, DataType : ",
            DataTypeString(context->input(0).dtype()),
            ", Dimension : ", input_dims));
  }

 private:
  template <DataType DT>
  void HandleCaseImpl(OpKernelContext* context,
                      const gtl::ArraySlice<Tmultiples> multiples_array,
                      Tensor* result) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStile_opsDTcc mht_2(mht_2_v, 420, "", "./tensorflow/core/kernels/tile_ops.cc", "HandleCaseImpl");

    typedef typename EnumToDataType<DT>::Type T;
    functor::Tile<Device, T, Tmultiples>()(context->eigen_device<Device>(),
                                           result, context->input(0),
                                           multiples_array);
  }

  template <DataType DT>
  void HandleCase(OpKernelContext* context,
                  const gtl::ArraySlice<Tmultiples> multiples_array,
                  Tensor* result);

  TF_DISALLOW_COPY_AND_ASSIGN(TileOp);
};

template <typename Device, typename Tmultiples>
template <DataType DT>
inline void TileOp<Device, Tmultiples>::HandleCase(
    OpKernelContext* context, const gtl::ArraySlice<Tmultiples> multiples_array,
    Tensor* result) {
  // TODO(vrv): print out the device name if useful. Currently disabled to avoid
  // having to use RTTI.
  LOG(FATAL) << "TileOp: Invalid combination of Device, DT: "
             // << typeid(Device).name() << ", "
             << DataTypeString(DT);
}

#define HANDLE_CASE(device, dtype, Tmultiples)                             \
  template <>                                                              \
  template <>                                                              \
  void TileOp<device, Tmultiples>::HandleCase<dtype>(                      \
      OpKernelContext * context,                                           \
      const gtl::ArraySlice<Tmultiples> multiples_array, Tensor* result) { \
    HandleCaseImpl<dtype>(context, multiples_array, result);               \
  }

#define HANDLE_TYPE_NAME_CPU(T)                            \
  HANDLE_CASE(CPUDevice, DataTypeToEnum<T>::value, int32); \
  HANDLE_CASE(CPUDevice, DataTypeToEnum<T>::value, int64_t);

#define HANDLE_TYPE_NAME_GPU(T)                            \
  HANDLE_CASE(GPUDevice, DataTypeToEnum<T>::value, int32); \
  HANDLE_CASE(GPUDevice, DataTypeToEnum<T>::value, int64_t);

TF_CALL_bool(HANDLE_TYPE_NAME_CPU);
TF_CALL_float(HANDLE_TYPE_NAME_CPU);
TF_CALL_bfloat16(HANDLE_TYPE_NAME_CPU);
TF_CALL_double(HANDLE_TYPE_NAME_CPU);
TF_CALL_uint8(HANDLE_TYPE_NAME_CPU);
TF_CALL_int8(HANDLE_TYPE_NAME_CPU);
TF_CALL_int32(HANDLE_TYPE_NAME_CPU);
TF_CALL_int16(HANDLE_TYPE_NAME_CPU);
TF_CALL_int64(HANDLE_TYPE_NAME_CPU);
TF_CALL_uint32(HANDLE_TYPE_NAME_CPU);
TF_CALL_uint64(HANDLE_TYPE_NAME_CPU);
TF_CALL_half(HANDLE_TYPE_NAME_CPU);
TF_CALL_complex64(HANDLE_TYPE_NAME_CPU);
TF_CALL_complex128(HANDLE_TYPE_NAME_CPU);
TF_CALL_tstring(HANDLE_TYPE_NAME_CPU);
TF_CALL_variant(HANDLE_TYPE_NAME_CPU);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
TF_CALL_bool(HANDLE_TYPE_NAME_GPU);
TF_CALL_float(HANDLE_TYPE_NAME_GPU);
TF_CALL_double(HANDLE_TYPE_NAME_GPU);
TF_CALL_int16(HANDLE_TYPE_NAME_GPU);
TF_CALL_int32(HANDLE_TYPE_NAME_GPU);
TF_CALL_int64(HANDLE_TYPE_NAME_GPU);
TF_CALL_half(HANDLE_TYPE_NAME_GPU);
TF_CALL_complex64(HANDLE_TYPE_NAME_GPU);
TF_CALL_complex128(HANDLE_TYPE_NAME_GPU);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM


#undef HANDLE_TYPE_NAME_CPU
#undef HANDLE_TYPE_NAME_GPU
#undef HANDLE_CASE

// --------------------------------------------------------------------------
template <typename Device, typename Tmultiples>
class TileGradientOp : public OpKernel {
 public:
  explicit TileGradientOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStile_opsDTcc mht_3(mht_3_v, 505, "", "./tensorflow/core/kernels/tile_ops.cc", "TileGradientOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStile_opsDTcc mht_4(mht_4_v, 510, "", "./tensorflow/core/kernels/tile_ops.cc", "Compute");

    const Tensor& input = context->input(0);
    const Tensor& multiples = context->input(1);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(multiples.shape()),
        errors::InvalidArgument("Expected multiples to be 1-D, but got shape ",
                                multiples.shape().DebugString()));
    OP_REQUIRES(context, input.dims() == multiples.NumElements(),
                errors::InvalidArgument(
                    "Expected multiples argument to be a vector of length ",
                    input.dims(), " but got length ", multiples.dim_size(0)));

    const int input_dims = input.dims();

    // Eigen doesn't support scalars on the GPU, so handle 0-D specially
    if (input_dims == 0) {
      context->set_output(0, input);
      return;
    }

    const gtl::ArraySlice<Tmultiples> multiples_array(
        multiples.flat<Tmultiples>().data(), input_dims);
    TensorShape output_shape;
    std::vector<Tmultiples> input_dim_size_vec;
    for (int i = 0; i < input_dims; ++i) {
      OP_REQUIRES(
          context, multiples_array[i] > 0,
          errors::InvalidArgument("Expected multiples[", i, "] > 0, but got ",
                                  multiples_array[i]));
      OP_REQUIRES(context, input.dim_size(i) % multiples_array[i] == 0,
                  errors::InvalidArgument("Expected input_dim[", i,
                                          "] to be divisible by multiples[", i,
                                          "], but ", input.dim_size(i), " % ",
                                          multiples_array[i], " != 0"));
      output_shape.AddDim(input.dim_size(i) / multiples_array[i]);
      input_dim_size_vec.push_back(input.dim_size(i));
    }
    if (output_shape == input.shape()) {
      context->set_output(0, input);
      return;
    }
    Tensor* result = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &result));

#define HANDLE_DIM(DT, NDIM)                                           \
  if (context->input(0).dtype() == DT && input_dims == NDIM) {         \
    HandleCase<DT, NDIM>(context, input_dim_size_vec, multiples_array, \
                         result);                                      \
    return;                                                            \
  }

#define HANDLE_TYPE(T) \
  HANDLE_DIM(T, 1)     \
  HANDLE_DIM(T, 2)     \
  HANDLE_DIM(T, 3)     \
  HANDLE_DIM(T, 4)     \
  HANDLE_DIM(T, 5)     \
  HANDLE_DIM(T, 6)     \
  HANDLE_DIM(T, 7)

#define HANDLE_TYPE_NAME(T) HANDLE_TYPE(DataTypeToEnum<T>::value)

    TF_CALL_float(HANDLE_TYPE_NAME);
    TF_CALL_double(HANDLE_TYPE_NAME);
    TF_CALL_int32(HANDLE_TYPE_NAME);
    TF_CALL_int16(HANDLE_TYPE_NAME);
    TF_CALL_int64(HANDLE_TYPE_NAME);
    TF_CALL_half(HANDLE_TYPE_NAME);
    TF_CALL_bfloat16(HANDLE_TYPE_NAME);
    TF_CALL_complex64(HANDLE_TYPE_NAME);
    TF_CALL_complex128(HANDLE_TYPE_NAME);

#undef HANDLE_TYPE_NAME
#undef HANDLE_TYPE
#undef HANDLE_DIM

    OP_REQUIRES(context, false,
                errors::Unimplemented("TileGradientOp : The input data type or "
                                      "dimension is not supported, DataType : ",
                                      DataTypeString(context->input(0).dtype()),
                                      ", Dimension : ", input_dims));
  }

 private:
  template <DataType DT, int NDIM>
  void HandleCase(OpKernelContext* context,
                  const std::vector<Tmultiples>& input_dims,
                  const gtl::ArraySlice<Tmultiples> multiples_array,
                  Tensor* result);

  template <DataType DT, int NDIM>
  void HandleCaseImpl(OpKernelContext* context,
                      const std::vector<Tmultiples>& input_dims,
                      const gtl::ArraySlice<Tmultiples> multiples_array,
                      Tensor* result) {
    typedef typename EnumToDataType<DT>::Type T;

    bool reduction_only = true;
    std::vector<Tmultiples> reduction_dims;

    for (int i = 0; i < NDIM; ++i) {
      if (input_dims[i] > multiples_array[i] && multiples_array[i] > 1) {
        reduction_only = false;
        break;
      } else {
        if (multiples_array[i] == input_dims[i]) {
          reduction_dims.push_back(i);
        }
      }
    }

    if (reduction_only) {
#define HANDLE_DIM(D)                                            \
  if (reduction_dims.size() == (D)) {                            \
    HandleReduce<T, NDIM, (D)>(context, reduction_dims, result); \
    return;                                                      \
  }
      // NOTE(keveman): Handling the most common case here.
      // Adding more cases here would require more templating and code
      // explosion. For instance, HANDLE_DIM(2) wouldn't make sense for NDIM=1.
      HANDLE_DIM(1);

// Fall through to the unoptimized version.
#undef HANDLE_DIM
    }

    Eigen::DSizes<Eigen::DenseIndex, NDIM> indices;
    Eigen::DSizes<Eigen::DenseIndex, NDIM> sizes;

    // Accumulate slices along the dimensions into the output. The number of
    // slices along dimension 'i' is simply the multiple along dimension 'i'
    // passed to the original Tile op.
    for (int i = 0; i < NDIM; ++i) {
      sizes[i] = input_dims[i] / multiples_array[i];
      indices[i] = 0;
    }

    bool first = true;
    while (true) {
      functor::TileGrad<Device, T, NDIM>()(
          context->eigen_device<Device>(), result->tensor<T, NDIM>(),
          context->input(0).tensor<T, NDIM>(), indices, sizes, first);
      first = false;
      // Increment the begin indices.
      int i = 0;
      while (i < NDIM && indices[i] / sizes[i] == multiples_array[i] - 1) {
        indices[i] = 0;
        ++i;
      }
      // We are finished if we have iterated to the maximum along all
      // dimensions.
      if (i == NDIM) {
        break;
      }
      indices[i] += sizes[i];
    }
  }

  template <typename T, int NDIM, int REDUCENDIM>
  void HandleReduce(OpKernelContext* context,
                    const std::vector<Tmultiples>& reduce_dim_in,
                    Tensor* result) {
    static_assert(NDIM >= REDUCENDIM, "Too many reduced dimensions");
    Eigen::DSizes<Eigen::DenseIndex, REDUCENDIM> reduce_dim;
    Eigen::DSizes<Eigen::DenseIndex, NDIM> reshape_dim;

    for (int i = 0; i < REDUCENDIM; ++i) {
      reduce_dim[i] = reduce_dim_in[i];
    }

    for (int i = 0; i < NDIM; ++i) {
      reshape_dim[i] = result->dim_size(i);
    }

    functor::ReduceAndReshape<Device, T, NDIM, REDUCENDIM>()(
        context->eigen_device<Device>(), result->tensor<T, NDIM>(),
        context->input(0).tensor<T, NDIM>(), reduce_dim, reshape_dim);
  }

  TF_DISALLOW_COPY_AND_ASSIGN(TileGradientOp);
};

template <typename Device, typename Tmultiples>
template <DataType DT, int NDIM>
inline void TileGradientOp<Device, Tmultiples>::HandleCase(
    OpKernelContext* context, const std::vector<Tmultiples>& input_dims,
    const gtl::ArraySlice<Tmultiples> multiples_array, Tensor* result) {
  LOG(FATAL) << "TileGradientOp: Invalid combination of Device, DT and NDIM: "
             << TypeIndex::Make<Device>().name() << ", " << DataTypeString(DT)
             << ", " << NDIM;
}

#define HANDLE_CASE(device, T, dtype, Tmultiples, ndim)                        \
  template <>                                                                  \
  template <>                                                                  \
  void TileGradientOp<device, Tmultiples>::HandleCase<dtype, ndim>(            \
      OpKernelContext * context, const std::vector<Tmultiples>& input_dims,    \
      const gtl::ArraySlice<Tmultiples> multiples_array, Tensor* result) {     \
    HandleCaseImpl<dtype, ndim>(context, input_dims, multiples_array, result); \
  }

// 0-D handled specially above
#define HANDLE_CASE_DIM(device, T, dtype)    \
  HANDLE_CASE(device, T, dtype, int32, 1);   \
  HANDLE_CASE(device, T, dtype, int32, 2);   \
  HANDLE_CASE(device, T, dtype, int32, 3);   \
  HANDLE_CASE(device, T, dtype, int32, 4);   \
  HANDLE_CASE(device, T, dtype, int32, 5);   \
  HANDLE_CASE(device, T, dtype, int32, 6);   \
  HANDLE_CASE(device, T, dtype, int32, 7);   \
  HANDLE_CASE(device, T, dtype, int64_t, 1); \
  HANDLE_CASE(device, T, dtype, int64_t, 2); \
  HANDLE_CASE(device, T, dtype, int64_t, 3); \
  HANDLE_CASE(device, T, dtype, int64_t, 4); \
  HANDLE_CASE(device, T, dtype, int64_t, 5); \
  HANDLE_CASE(device, T, dtype, int64_t, 6); \
  HANDLE_CASE(device, T, dtype, int64_t, 7);

#define HANDLE_TYPE_NAME_CPU(T) \
  HANDLE_CASE_DIM(CPUDevice, T, DataTypeToEnum<T>::value);

#define HANDLE_TYPE_NAME_GPU(T) \
  HANDLE_CASE_DIM(GPUDevice, T, DataTypeToEnum<T>::value);

TF_CALL_float(HANDLE_TYPE_NAME_CPU);
TF_CALL_double(HANDLE_TYPE_NAME_CPU);
TF_CALL_int16(HANDLE_TYPE_NAME_CPU);
TF_CALL_int32(HANDLE_TYPE_NAME_CPU);
TF_CALL_int64(HANDLE_TYPE_NAME_CPU);
TF_CALL_half(HANDLE_TYPE_NAME_CPU);
TF_CALL_complex64(HANDLE_TYPE_NAME_CPU);
TF_CALL_complex128(HANDLE_TYPE_NAME_CPU);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
TF_CALL_float(HANDLE_TYPE_NAME_GPU);
TF_CALL_double(HANDLE_TYPE_NAME_GPU);
TF_CALL_int16(HANDLE_TYPE_NAME_GPU);
TF_CALL_int32(HANDLE_TYPE_NAME_GPU);
TF_CALL_int64(HANDLE_TYPE_NAME_GPU);
TF_CALL_half(HANDLE_TYPE_NAME_GPU);
TF_CALL_complex64(HANDLE_TYPE_NAME_GPU);
TF_CALL_complex128(HANDLE_TYPE_NAME_GPU);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM


#undef HANDLE_TYPE_NAME_CPU
#undef HANDLE_TYPE_NAME_GPU
#undef HANDLE_CASE_DIM
#undef HANDLE_CASE

REGISTER_KERNEL_BUILDER(Name("Tile")
                            .Device(DEVICE_CPU)
                            .HostMemory("multiples")
                            .TypeConstraint<int32>("Tmultiples"),
                        TileOp<CPUDevice, int32>);
REGISTER_KERNEL_BUILDER(Name("Tile")
                            .Device(DEVICE_CPU)
                            .HostMemory("multiples")
                            .TypeConstraint<int64_t>("Tmultiples"),
                        TileOp<CPUDevice, int64>);
REGISTER_KERNEL_BUILDER(Name("TileGrad")
                            .Device(DEVICE_CPU)
                            .HostMemory("multiples")
                            .TypeConstraint<int32>("Tmultiples"),
                        TileGradientOp<CPUDevice, int32>);
REGISTER_KERNEL_BUILDER(Name("TileGrad")
                            .Device(DEVICE_CPU)
                            .HostMemory("multiples")
                            .TypeConstraint<int64_t>("Tmultiples"),
                        TileGradientOp<CPUDevice, int64>);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define REGISTER_GPU_TILE(type)                                      \
  REGISTER_KERNEL_BUILDER(Name("Tile")                               \
                              .Device(DEVICE_GPU)                    \
                              .TypeConstraint<type>("T")             \
                              .TypeConstraint<int32>("Tmultiples")   \
                              .HostMemory("multiples"),              \
                          TileOp<GPUDevice, int32>);                 \
  REGISTER_KERNEL_BUILDER(Name("Tile")                               \
                              .Device(DEVICE_GPU)                    \
                              .TypeConstraint<type>("T")             \
                              .TypeConstraint<int64_t>("Tmultiples") \
                              .HostMemory("multiples"),              \
                          TileOp<GPUDevice, int64>);

#define REGISTER_GPU_TILE_GRAD(type)                                 \
  REGISTER_KERNEL_BUILDER(Name("TileGrad")                           \
                              .Device(DEVICE_GPU)                    \
                              .TypeConstraint<type>("T")             \
                              .TypeConstraint<int32>("Tmultiples")   \
                              .HostMemory("multiples"),              \
                          TileGradientOp<GPUDevice, int32>);         \
  REGISTER_KERNEL_BUILDER(Name("TileGrad")                           \
                              .Device(DEVICE_GPU)                    \
                              .TypeConstraint<type>("T")             \
                              .TypeConstraint<int64_t>("Tmultiples") \
                              .HostMemory("multiples"),              \
                          TileGradientOp<GPUDevice, int64>);

#define REGISTER_GPU(type) \
  REGISTER_GPU_TILE(type); \
  REGISTER_GPU_TILE_GRAD(type);

TF_CALL_bool(REGISTER_GPU_TILE);
TF_CALL_float(REGISTER_GPU);
TF_CALL_double(REGISTER_GPU);
TF_CALL_half(REGISTER_GPU);
TF_CALL_int16(REGISTER_GPU);
TF_CALL_int32(REGISTER_GPU);
TF_CALL_int64(REGISTER_GPU);
TF_CALL_complex64(REGISTER_GPU);
TF_CALL_complex128(REGISTER_GPU)

#undef REGISTER_GPU_TILE
#undef REGISTER_GPU_TILE_GRAD
#undef REGISTER_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM


}  // namespace tensorflow
