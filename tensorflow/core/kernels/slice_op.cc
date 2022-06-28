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
class MHTracer_DTPStensorflowPScorePSkernelsPSslice_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSslice_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSslice_opDTcc() {
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

#include "tensorflow/core/kernels/slice_op.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/prefetch.h"

namespace tensorflow {

namespace {

void IntTensorToInt64Vec(const Tensor& tensor,
                         gtl::InlinedVector<int64_t, 4>* out) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSslice_opDTcc mht_0(mht_0_v, 209, "", "./tensorflow/core/kernels/slice_op.cc", "IntTensorToInt64Vec");

  out->resize(tensor.NumElements());
  int64_t* out_ptr = out->data();
  if (tensor.dtype() == DT_INT32) {
    const int32* tensor_ptr = tensor.flat<int32>().data();
    for (int64_t i = 0; i < tensor.NumElements(); ++i) {
      out_ptr[i] = tensor_ptr[i];
    }
  } else if (tensor.dtype() == DT_INT64) {
    const int64_t* tensor_ptr = tensor.flat<int64_t>().data();
    for (int64_t i = 0; i < tensor.NumElements(); ++i) {
      out_ptr[i] = tensor_ptr[i];
    }
  } else {
    LOG(FATAL) << "begin must be either int32 or int64";
  }
}

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

// Shared code that is not dependent on the type of T.  We do this to reduce
// code size by not duplicating all this for all T (float, double, int32, etc.)
void SharedSliceValidation(OpKernelContext* context, const Tensor& input,
                           TensorShape* output_shape, bool* is_identity,
                           bool* slice_dim0,
                           gtl::InlinedVector<int64_t, 4>* begin,
                           gtl::InlinedVector<int64_t, 4>* size) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSslice_opDTcc mht_1(mht_1_v, 239, "", "./tensorflow/core/kernels/slice_op.cc", "SharedSliceValidation");

  const Tensor& begin_tensor = context->input(1);
  const Tensor& size_tensor = context->input(2);

  OP_REQUIRES(
      context,
      TensorShapeUtils::IsVector(begin_tensor.shape()) &&
          TensorShapeUtils::IsVector(size_tensor.shape()) &&
          begin_tensor.NumElements() == input.dims() &&
          size_tensor.NumElements() == input.dims(),
      errors::InvalidArgument(
          "Expected begin and size arguments to be 1-D tensors of size ",
          input.dims(), ", but got shapes ", begin_tensor.shape().DebugString(),
          " and ", size_tensor.shape().DebugString(), " instead."));

  const int input_dims = input.dims();
  IntTensorToInt64Vec(begin_tensor, begin);
  IntTensorToInt64Vec(size_tensor, size);
  for (int i = 0; i < input_dims; ++i) {
    if ((*size)[i] == -1) {
      // A size[i] of -1 means "all elements from begin[i] to dim_size(i)".
      (*size)[i] = input.dim_size(i) - (*begin)[i];
    }
  }

  *is_identity = true;
  *slice_dim0 = true;
  for (int i = 0; i < input_dims; ++i) {
    int64_t b = (*begin)[i];
    int64_t s = (*size)[i];
    if (input.dim_size(i) == 0) {
      OP_REQUIRES(
          context, b == 0 && s == 0,
          errors::InvalidArgument("Expected begin[", i, "] == 0 (got ", b,
                                  ") and size[", i, "] == 0 ", "(got ", s,
                                  ") when ", "input.dim_size(", i, ") == 0"));
    } else {
      OP_REQUIRES(context, 0 <= b && b <= input.dim_size(i),
                  errors::InvalidArgument("Expected begin[", i, "] in [0, ",
                                          input.dim_size(i), "], but got ", b));
      OP_REQUIRES(
          context, 0 <= s && b + s <= input.dim_size(i),
          errors::InvalidArgument("Expected size[", i, "] in [0, ",
                                  input.dim_size(i) - b, "], but ", "got ", s));
    }
    output_shape->AddDim(s);
    const bool take_all = (b == 0) && (s == input.dim_size(i));
    (*is_identity) &= take_all;
    (*slice_dim0) &= (i == 0) || take_all;
  }
}

// Extracted out code in SliceOp::Compute so that MklSliceOp can reuse this
// generic code
template <typename T>
static void SharedSliceCommonCases(OpKernelContext* context,
                                   const Tensor& input,
                                   gtl::InlinedVector<int64, 4>* begin,
                                   gtl::InlinedVector<int64, 4>* size,
                                   Tensor** result, bool* done) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSslice_opDTcc mht_2(mht_2_v, 301, "", "./tensorflow/core/kernels/slice_op.cc", "SharedSliceCommonCases");

  bool is_identity = true;
  bool slice_dim0 = true;
  TensorShape output_shape;
  *done = false;

  SharedSliceValidation(context, input, &output_shape, &is_identity,
                        &slice_dim0, begin, size);
  if (!context->status().ok()) return;
  if (is_identity) {
    VLOG(1) << "Slice identity";
    context->set_output(0, input);
    *done = true;
    return;
  }

  if (slice_dim0 &&
      IsDim0SliceAligned<T>(input.shape(), (*begin)[0], (*size)[0])) {
    VLOG(1) << "Slice dim 0: " << input.shape().DebugString();
    CHECK_GE(input.dims(), 1);  // Otherwise, is_identity should be true.
    context->set_output(0, input.Slice((*begin)[0], (*begin)[0] + (*size)[0]));
    *done = true;
    return;
  }

  OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, result));
}

template <typename Device, typename T>
class SliceOp : public OpKernel {
 public:
  explicit SliceOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSslice_opDTcc mht_3(mht_3_v, 335, "", "./tensorflow/core/kernels/slice_op.cc", "SliceOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSslice_opDTcc mht_4(mht_4_v, 340, "", "./tensorflow/core/kernels/slice_op.cc", "Compute");

    gtl::InlinedVector<int64_t, 4> begin;
    gtl::InlinedVector<int64_t, 4> size;
    const Tensor& input = context->input(0);
    Tensor* result = nullptr;
    bool done = false;
    SharedSliceCommonCases<T>(context, input, &begin, &size, &result, &done);
    if (!context->status().ok() || done == true) return;

    const int input_dims = input.dims();

    if (result->NumElements() > 0) {
      if (std::is_same<Device, CPUDevice>::value && input_dims == 2 &&
          DataTypeCanUseMemcpy(DataTypeToEnum<T>::v())) {
        auto input_t = input.tensor<T, 2>();
        auto output_t = result->tensor<T, 2>();

        const int64_t row_begin = begin[0];
        const int64_t col_begin = begin[1];
        const int64_t row_size = size[0];
        const int64_t col_size = size[1];

        // TODO(agarwal): Consider multi-threading this loop for cases where
        // row_size is very large.
        for (int i = 0; i < row_size; ++i) {
          const int64_t row = row_begin + i;
          if (i + 1 < size[0]) {
            port::prefetch<port::PREFETCH_HINT_T0>(&output_t(i + 1, 0));
            port::prefetch<port::PREFETCH_HINT_T0>(
                &input_t(row + 1, col_begin));
          }
          memcpy(&output_t(i, 0), &input_t(row, col_begin),
                 col_size * sizeof(T));
        }
        return;
      }
#define HANDLE_DIM(NDIM)                                   \
  if (input_dims == NDIM) {                                \
    HandleCase<NDIM>(context, begin, size, input, result); \
    return;                                                \
  }

      HANDLE_DIM(1);
      HANDLE_DIM(2);
      HANDLE_DIM(3);
      HANDLE_DIM(4);
      HANDLE_DIM(5);
      HANDLE_DIM(6);
      HANDLE_DIM(7);
      HANDLE_DIM(8);

#undef HANDLE_DIM

      OP_REQUIRES(
          context, false,
          errors::Unimplemented("SliceOp : Unhandled input dimensions"));
    }
  }

 private:
  template <int NDIM>
  void HandleCase(OpKernelContext* context, gtl::ArraySlice<int64_t> begin,
                  gtl::ArraySlice<int64_t> size, const Tensor& input,
                  Tensor* result) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSslice_opDTcc mht_5(mht_5_v, 406, "", "./tensorflow/core/kernels/slice_op.cc", "HandleCase");

    Eigen::DSizes<Eigen::DenseIndex, NDIM> indices;
    Eigen::DSizes<Eigen::DenseIndex, NDIM> sizes;
    for (int i = 0; i < NDIM; ++i) {
      indices[i] = begin[i];
      sizes[i] = size[i];
    }

    functor::Slice<Device, T, NDIM>()(context->eigen_device<Device>(),
                                      result->tensor<T, NDIM>(),
                                      input.tensor<T, NDIM>(), indices, sizes);
  }
};

}  // namespace

// Forward declarations of the functor specializations for declared in the
// sharded source files.
namespace functor {
#define DECLARE_CPU_SPEC(T, NDIM)                                  \
  template <>                                                      \
  void Slice<CPUDevice, T, NDIM>::operator()(                      \
      const CPUDevice& d, typename TTypes<T, NDIM>::Tensor output, \
      typename TTypes<T, NDIM>::ConstTensor input,                 \
      const Eigen::DSizes<Eigen::DenseIndex, NDIM>& indices,       \
      const Eigen::DSizes<Eigen::DenseIndex, NDIM>& sizes);        \
  extern template struct Slice<CPUDevice, T, NDIM>;

#define DECLARE_FOR_N(T)  \
  DECLARE_CPU_SPEC(T, 1); \
  DECLARE_CPU_SPEC(T, 2); \
  DECLARE_CPU_SPEC(T, 3); \
  DECLARE_CPU_SPEC(T, 4); \
  DECLARE_CPU_SPEC(T, 5); \
  DECLARE_CPU_SPEC(T, 6); \
  DECLARE_CPU_SPEC(T, 7); \
  DECLARE_CPU_SPEC(T, 8);

TF_CALL_ALL_TYPES(DECLARE_FOR_N);

#undef DECLARE_FOR_N
#undef DECLARE_CPU_SPEC
}  // namespace functor

#define REGISTER_SLICE(type)                             \
  REGISTER_KERNEL_BUILDER(Name("Slice")                  \
                              .Device(DEVICE_CPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("begin")       \
                              .HostMemory("size"),       \
                          SliceOp<CPUDevice, type>)

TF_CALL_POD_STRING_TYPES(REGISTER_SLICE);
TF_CALL_QUANTIZED_TYPES(REGISTER_SLICE);
#undef REGISTER_SLICE

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T, NDIM)                                  \
  template <>                                                      \
  void Slice<GPUDevice, T, NDIM>::operator()(                      \
      const GPUDevice& d, typename TTypes<T, NDIM>::Tensor output, \
      typename TTypes<T, NDIM>::ConstTensor input,                 \
      const Eigen::DSizes<Eigen::DenseIndex, NDIM>& indices,       \
      const Eigen::DSizes<Eigen::DenseIndex, NDIM>& sizes);        \
  extern template struct Slice<GPUDevice, T, NDIM>;

#define DECLARE_FOR_N(T)  \
  DECLARE_GPU_SPEC(T, 1); \
  DECLARE_GPU_SPEC(T, 2); \
  DECLARE_GPU_SPEC(T, 3); \
  DECLARE_GPU_SPEC(T, 4); \
  DECLARE_GPU_SPEC(T, 5); \
  DECLARE_GPU_SPEC(T, 6); \
  DECLARE_GPU_SPEC(T, 7); \
  DECLARE_GPU_SPEC(T, 8);

TF_CALL_bfloat16(DECLARE_FOR_N);
TF_CALL_int8(DECLARE_FOR_N);
TF_CALL_int32(DECLARE_FOR_N);
TF_CALL_int64(DECLARE_FOR_N);
TF_CALL_GPU_ALL_TYPES(DECLARE_FOR_N);

#undef DECLARE_FOR_N
#undef DECLARE_GPU_SPEC
}  // namespace functor

#define REGISTER_GPU(type)                               \
  REGISTER_KERNEL_BUILDER(Name("Slice")                  \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("begin")       \
                              .HostMemory("size"),       \
                          SliceOp<GPUDevice, type>)

TF_CALL_bfloat16(REGISTER_GPU);
TF_CALL_int8(REGISTER_GPU);
TF_CALL_int64(REGISTER_GPU);
TF_CALL_GPU_ALL_TYPES(REGISTER_GPU);

#undef REGISTER_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// A special DEVICE_DEFAULT kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("Slice")
                            .Device(DEVICE_DEFAULT)
                            .TypeConstraint<int32>("T")
                            .HostMemory("input")
                            .HostMemory("begin")
                            .HostMemory("size")
                            .HostMemory("output"),
                        SliceOp<CPUDevice, int32>);

}  // namespace tensorflow
