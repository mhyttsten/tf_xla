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
class MHTracer_DTPStensorflowPScorePSkernelsPSstrided_slice_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSstrided_slice_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSstrided_slice_opDTcc() {
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

#include "tensorflow/core/lib/core/refcount.h"
#define EIGEN_USE_THREADS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/kernels/strided_slice_op.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/kernels/inplace_ops_functor.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/kernels/slice_op.h"
#include "tensorflow/core/kernels/strided_slice_op_impl.h"
#include "tensorflow/core/kernels/training_op_helpers.h"
#include "tensorflow/core/kernels/variable_ops.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/prefetch.h"
#include "tensorflow/core/util/strided_slice_op.h"

namespace tensorflow {
namespace {

template <typename T>
struct MemCpyFunctor {
  // Returns true if the copy was made with memcpy, false otherwise.
  bool Copy(const Tensor& input, const gtl::InlinedVector<int64, 4>& begin,
            const gtl::InlinedVector<int64, 4>& end, Tensor* result) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstrided_slice_opDTcc mht_0(mht_0_v, 220, "", "./tensorflow/core/kernels/strided_slice_op.cc", "Copy");

    if (DataTypeCanUseMemcpy(DataTypeToEnum<T>::v())) {
      auto in = input.tensor<T, 2>();
      auto output = result->tensor<T, 2>();
      // TODO(agarwal): Consider multi-threading if size[0] is large
      for (int row_in = begin[0], row_out = 0; row_in < end[0];
           ++row_in, ++row_out) {
        if (row_in + 1 < end[0]) {
          port::prefetch<port::PREFETCH_HINT_T0>(&output(row_in + 1, 0));
          port::prefetch<port::PREFETCH_HINT_T0>(&in(row_in + 1, begin[1]));
        }
        memcpy(&output(row_out, 0), &in(row_in, begin[1]),
               (end[1] - begin[1]) * sizeof(T));
      }
      return true;
    }
    return false;
  }
};

template <>
struct MemCpyFunctor<ResourceHandle> {
  bool Copy(const Tensor& input, const gtl::InlinedVector<int64_t, 4>& begin,
            const gtl::InlinedVector<int64_t, 4>& end, Tensor* result) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstrided_slice_opDTcc mht_1(mht_1_v, 246, "", "./tensorflow/core/kernels/strided_slice_op.cc", "Copy");

    return false;
  }
};

}  // namespace

template <typename Device, typename T>
class StridedSliceOp : public OpKernel {
 public:
  explicit StridedSliceOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstrided_slice_opDTcc mht_2(mht_2_v, 259, "", "./tensorflow/core/kernels/strided_slice_op.cc", "StridedSliceOp");

    OP_REQUIRES_OK(context, context->GetAttr("begin_mask", &begin_mask));
    OP_REQUIRES_OK(context, context->GetAttr("end_mask", &end_mask));
    OP_REQUIRES_OK(context, context->GetAttr("ellipsis_mask", &ellipsis_mask));
    OP_REQUIRES_OK(context, context->GetAttr("new_axis_mask", &new_axis_mask));
    OP_REQUIRES_OK(context,
                   context->GetAttr("shrink_axis_mask", &shrink_axis_mask));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstrided_slice_opDTcc mht_3(mht_3_v, 271, "", "./tensorflow/core/kernels/strided_slice_op.cc", "Compute");

    TensorShape processing_shape, final_shape;
    bool is_identity = true;
    bool slice_dim0 = true;
    bool is_simple_slice = true;
    gtl::InlinedVector<int64_t, 4> begin;
    gtl::InlinedVector<int64_t, 4> end;
    gtl::InlinedVector<int64_t, 4> strides;

    OP_REQUIRES_OK(
        context, ValidateStridedSliceOp(
                     &context->input(1), &context->input(2), context->input(3),
                     context->input(0).shape(), begin_mask, end_mask,
                     ellipsis_mask, new_axis_mask, shrink_axis_mask,
                     &processing_shape, &final_shape, &is_identity,
                     &is_simple_slice, &slice_dim0, &begin, &end, &strides));
    const Tensor& input = context->input(0);

    // Optimization #1, slice is a no-op plus reshape
    if (is_identity) {
      VLOG(1) << "Strided slice identity ";
      Tensor tmp;
      OP_REQUIRES(context, tmp.CopyFrom(input, final_shape),
                  errors::Internal("Copy failed"));
      context->set_output(0, tmp);
      return;
    }

    // Optimization #2, slice is memory contiguous (only occurs in dim 0)
    if (slice_dim0 && IsDim0SliceAligned<T>(input.shape(), begin[0], end[0])) {
      OP_REQUIRES(context, input.dims() >= 1,
                  errors::InvalidArgument(
                      "Input must have rank at least 1, got: ", input.dims()));
      // Otherwise, is_identity should be true.
      VLOG(1) << "Strided slice dim 0: " << input.shape().DebugString();
      // To tolerate begin[0] > end[0] (a 0-output slice), we min(begin, end).
      Tensor slice = input.Slice(std::min(begin[0], end[0]), end[0]);
      Tensor tmp;
      OP_REQUIRES(context, tmp.CopyFrom(slice, final_shape),
                  errors::Internal("Copy failed"));
      context->set_output(0, tmp);
      return;
    }

    Tensor* result = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, final_shape, &result));
    const int input_dims = input.dims();
    const int processing_dims = processing_shape.dims();

    if (processing_shape.num_elements() > 0) {
      // Optimization #3, slice has stride 1 in all dimensions
      // Optimization #3A, slice has only two dimensions
      // TODO(aselle): Here we are restricting to processing_shape and
      // final_shape being 2D. This isn't strictly necessary, but I don't
      // want to blow up code gen size, because to shape<> you need static
      // NDIM and T
      if (is_simple_slice && std::is_same<Device, CPUDevice>::value &&
          input_dims == 2 && processing_shape.dims() == 2 &&
          final_shape.dims() == 2 && new_axis_mask == 0) {
        MemCpyFunctor<T> functor;
        if (functor.Copy(input, begin, end, result)) {
          return;
        }
      }

#define HANDLE_DIM(NDIM)                                                       \
  if (processing_dims == NDIM) {                                               \
    HandleStridedSliceCase<Device, T, NDIM>(context, begin, end, strides,      \
                                            processing_shape, is_simple_slice, \
                                            result);                           \
    return;                                                                    \
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
          errors::Unimplemented("Unhandled input dimensions ", input_dims));
    }
  }

 private:
  int32 begin_mask, end_mask;
  int32 ellipsis_mask, new_axis_mask, shrink_axis_mask;
};

template <typename Device, typename T>
class StridedSliceGradOp : public OpKernel {
 public:
  explicit StridedSliceGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstrided_slice_opDTcc mht_4(mht_4_v, 373, "", "./tensorflow/core/kernels/strided_slice_op.cc", "StridedSliceGradOp");

    OP_REQUIRES_OK(context, context->GetAttr("begin_mask", &begin_mask));
    OP_REQUIRES_OK(context, context->GetAttr("end_mask", &end_mask));
    OP_REQUIRES_OK(context, context->GetAttr("ellipsis_mask", &ellipsis_mask));
    OP_REQUIRES_OK(context, context->GetAttr("new_axis_mask", &new_axis_mask));
    OP_REQUIRES_OK(context,
                   context->GetAttr("shrink_axis_mask", &shrink_axis_mask));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstrided_slice_opDTcc mht_5(mht_5_v, 385, "", "./tensorflow/core/kernels/strided_slice_op.cc", "Compute");

    TensorShape processing_shape, final_shape;
    bool is_identity = true;
    bool slice_dim0 = true;
    bool is_simple_slice = true;
    gtl::InlinedVector<int64_t, 4> begin;
    gtl::InlinedVector<int64_t, 4> end;
    gtl::InlinedVector<int64_t, 4> strides;

    TensorShape input_shape;
    const Tensor& input_shape_tensor = context->input(0);
    OP_REQUIRES(
        context, input_shape_tensor.dims() == 1,
        errors::InvalidArgument("shape must be 1-D, got shape.shape = ",
                                input_shape_tensor.shape().DebugString()));
    if (input_shape_tensor.dtype() == DT_INT32) {
      OP_REQUIRES_OK(
          context, TensorShapeUtils::MakeShape(input_shape_tensor.vec<int32>(),
                                               &input_shape));
    } else if (input_shape_tensor.dtype() == DT_INT64) {
      OP_REQUIRES_OK(context,
                     TensorShapeUtils::MakeShape(
                         input_shape_tensor.vec<int64_t>(), &input_shape));
    } else {
      LOG(FATAL) << "shape must have type int32 or int64.";
    }

    OP_REQUIRES_OK(
        context,
        ValidateStridedSliceOp(
            &context->input(1), &context->input(2), context->input(3),
            input_shape, begin_mask, end_mask, ellipsis_mask, new_axis_mask,
            shrink_axis_mask, &processing_shape, &final_shape, &is_identity,
            &is_simple_slice, &slice_dim0, &begin, &end, &strides));

    // Check to make sure dy is consistent with the original slice
    TensorShape dy_shape = context->input(4).shape();
    OP_REQUIRES(
        context, final_shape == dy_shape,
        errors::InvalidArgument("shape of dy was ", dy_shape.DebugString(),
                                " instead of ", final_shape.DebugString()));

    if (!context->status().ok()) return;

    // const int input_dims = input.dims();
    const int processing_dims = processing_shape.dims();
    Tensor* result = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_shape, &result));

    if (processing_shape.dims() == 0) {
      auto in = context->input(4);
      OP_REQUIRES(context, result->CopyFrom(in, processing_shape),
                  errors::Internal("Copy failed"));
      return;
    }

#define HANDLE_DIM(NDIM)                                                      \
  if (processing_dims == NDIM) {                                              \
    HandleStridedSliceGradCase<Device, T, NDIM>(context, begin, end, strides, \
                                                processing_shape,             \
                                                is_simple_slice, result);     \
    return;                                                                   \
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
  }

 private:
  int32 begin_mask, end_mask;
  int32 ellipsis_mask, new_axis_mask, shrink_axis_mask;
};

template <typename Device, typename T, bool isTensor>
class StridedSliceAssignOp : public OpKernel {
 public:
  explicit StridedSliceAssignOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstrided_slice_opDTcc mht_6(mht_6_v, 473, "", "./tensorflow/core/kernels/strided_slice_op.cc", "StridedSliceAssignOp");

    OP_REQUIRES_OK(context, context->GetAttr("begin_mask", &begin_mask));
    OP_REQUIRES_OK(context, context->GetAttr("end_mask", &end_mask));
    OP_REQUIRES_OK(context, context->GetAttr("ellipsis_mask", &ellipsis_mask));
    OP_REQUIRES_OK(context, context->GetAttr("new_axis_mask", &new_axis_mask));
    OP_REQUIRES_OK(context,
                   context->GetAttr("shrink_axis_mask", &shrink_axis_mask));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSstrided_slice_opDTcc mht_7(mht_7_v, 485, "", "./tensorflow/core/kernels/strided_slice_op.cc", "Compute");

    TensorShape processing_shape, final_shape;
    bool is_identity = true;
    bool slice_dim0 = true;
    bool is_simple_slice = true;
    gtl::InlinedVector<int64_t, 4> begin;
    gtl::InlinedVector<int64_t, 4> end;
    gtl::InlinedVector<int64_t, 4> strides;

    Tensor* old_lhs = nullptr;
    Tensor tmp;
    if (isTensor) {
      const Tensor& input = context->input(0);

      int forwarded_input;
      OP_REQUIRES_OK(context,
                     context->forward_input_or_allocate_output(
                         {0}, 0, input.shape(), &old_lhs, &forwarded_input));
      if (forwarded_input < 0) {
        OP_REQUIRES_OK(context,
                       tensorflow::functor::DoCopy(
                           context->eigen_device<Device>(), input, old_lhs));
      }
    } else {
      if (context->input_dtype(0) == DT_RESOURCE) {
        core::RefCountPtr<Var> v;
        OP_REQUIRES_OK(
            context, LookupResource(context, HandleFromInput(context, 0), &v));
        OP_REQUIRES_OK(context,
                       EnsureSparseVariableAccess<Device, T>(context, v.get()));
        mutex_lock ml(*v->mu());
        old_lhs = v->tensor();
        OP_REQUIRES(context, old_lhs->dtype() == DataTypeToEnum<T>::value,
                    errors::InvalidArgument(
                        "l-value dtype ", DataTypeString(old_lhs->dtype()),
                        " does not match r-value dtype ",
                        DataTypeString(DataTypeToEnum<T>::value)));
      } else {
        context->forward_ref_input_to_ref_output(0, 0);
        tmp = context->mutable_input(0, true);
        old_lhs = &tmp;
      }
    }

    OP_REQUIRES_OK(
        context, ValidateStridedSliceOp(
                     &context->input(1), &context->input(2), context->input(3),
                     old_lhs->shape(), begin_mask, end_mask, ellipsis_mask,
                     new_axis_mask, shrink_axis_mask, &processing_shape,
                     &final_shape, &is_identity, &is_simple_slice, &slice_dim0,
                     &begin, &end, &strides));

    if (processing_shape.num_elements()) {
      const Tensor& input = context->input(4);
      TensorShape input_shape = input.shape();
      TensorShape original_shape = old_lhs->shape();
      // TODO(aselle): This check is too strong, we only should need
      // input_shape to be broadcastable to final_shape
      OP_REQUIRES(
          context, final_shape == input_shape,
          errors::Unimplemented(
              "sliced l-value shape ", final_shape.DebugString(),
              " does not match r-value shape ", input_shape.DebugString(),
              ". Automatic broadcasting not ", "yet implemented."));
      const int processing_dims = processing_shape.dims();

      // 0-dimensional case implies the left and right are exactly the same
      // scalar shape

// Handle general dimensions
#define HANDLE_DIM(NDIM)                                                       \
  if (processing_dims == NDIM) {                                               \
    HandleStridedSliceAssignCase<Device, T, NDIM>()(context, begin, end,       \
                                                    strides, processing_shape, \
                                                    is_simple_slice, old_lhs); \
    return;                                                                    \
  }
      HANDLE_DIM(0);
      HANDLE_DIM(1);
      HANDLE_DIM(2);
      HANDLE_DIM(3);
      HANDLE_DIM(4);
      HANDLE_DIM(5);
      HANDLE_DIM(6);
      HANDLE_DIM(7);
      HANDLE_DIM(8);
#undef HANDLE_DIM

      OP_REQUIRES(context, false,
                  errors::Unimplemented("Unhandled input dimensions ",
                                        processing_dims));
    }
  }

 private:
  int32 begin_mask, end_mask;
  int32 ellipsis_mask, new_axis_mask, shrink_axis_mask;
};

#define REGISTER_STRIDED_SLICE(type)                                    \
  REGISTER_KERNEL_BUILDER(Name("StridedSlice")                          \
                              .Device(DEVICE_CPU)                       \
                              .TypeConstraint<type>("T")                \
                              .HostMemory("begin")                      \
                              .HostMemory("end")                        \
                              .HostMemory("strides"),                   \
                          StridedSliceOp<CPUDevice, type>)              \
  REGISTER_KERNEL_BUILDER(Name("StridedSliceGrad")                      \
                              .Device(DEVICE_CPU)                       \
                              .TypeConstraint<type>("T")                \
                              .HostMemory("shape")                      \
                              .HostMemory("begin")                      \
                              .HostMemory("end")                        \
                              .HostMemory("strides"),                   \
                          StridedSliceGradOp<CPUDevice, type>)          \
  REGISTER_KERNEL_BUILDER(Name("StridedSliceAssign")                    \
                              .Device(DEVICE_CPU)                       \
                              .TypeConstraint<type>("T")                \
                              .HostMemory("begin")                      \
                              .HostMemory("end")                        \
                              .HostMemory("strides"),                   \
                          StridedSliceAssignOp<CPUDevice, type, false>) \
  REGISTER_KERNEL_BUILDER(Name("ResourceStridedSliceAssign")            \
                              .Device(DEVICE_CPU)                       \
                              .TypeConstraint<type>("T")                \
                              .HostMemory("ref")                        \
                              .HostMemory("begin")                      \
                              .HostMemory("end")                        \
                              .HostMemory("strides"),                   \
                          StridedSliceAssignOp<CPUDevice, type, false>) \
  REGISTER_KERNEL_BUILDER(Name("TensorStridedSliceUpdate")              \
                              .Device(DEVICE_CPU)                       \
                              .TypeConstraint<type>("T")                \
                              .HostMemory("begin")                      \
                              .HostMemory("end")                        \
                              .HostMemory("strides"),                   \
                          StridedSliceAssignOp<CPUDevice, type, true>)

TF_CALL_ALL_TYPES(REGISTER_STRIDED_SLICE);
TF_CALL_QUANTIZED_TYPES(REGISTER_STRIDED_SLICE);

#undef REGISTER_STRIDED_SLICE

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_GPU(type)                                              \
  REGISTER_KERNEL_BUILDER(Name("StridedSlice")                          \
                              .Device(DEVICE_GPU)                       \
                              .TypeConstraint<type>("T")                \
                              .HostMemory("begin")                      \
                              .HostMemory("end")                        \
                              .HostMemory("strides"),                   \
                          StridedSliceOp<GPUDevice, type>)              \
  REGISTER_KERNEL_BUILDER(Name("StridedSliceGrad")                      \
                              .Device(DEVICE_GPU)                       \
                              .TypeConstraint<type>("T")                \
                              .HostMemory("shape")                      \
                              .HostMemory("begin")                      \
                              .HostMemory("end")                        \
                              .HostMemory("strides"),                   \
                          StridedSliceGradOp<GPUDevice, type>)          \
  REGISTER_KERNEL_BUILDER(Name("StridedSliceAssign")                    \
                              .Device(DEVICE_GPU)                       \
                              .TypeConstraint<type>("T")                \
                              .HostMemory("begin")                      \
                              .HostMemory("end")                        \
                              .HostMemory("strides"),                   \
                          StridedSliceAssignOp<GPUDevice, type, false>) \
  REGISTER_KERNEL_BUILDER(Name("ResourceStridedSliceAssign")            \
                              .Device(DEVICE_GPU)                       \
                              .TypeConstraint<type>("T")                \
                              .HostMemory("ref")                        \
                              .HostMemory("begin")                      \
                              .HostMemory("end")                        \
                              .HostMemory("strides"),                   \
                          StridedSliceAssignOp<GPUDevice, type, false>) \
  REGISTER_KERNEL_BUILDER(Name("TensorStridedSliceUpdate")              \
                              .Device(DEVICE_GPU)                       \
                              .TypeConstraint<type>("T")                \
                              .HostMemory("begin")                      \
                              .HostMemory("end")                        \
                              .HostMemory("strides"),                   \
                          StridedSliceAssignOp<GPUDevice, type, true>)

TF_CALL_uint8(REGISTER_GPU);
TF_CALL_int8(REGISTER_GPU);
TF_CALL_int64(REGISTER_GPU);
TF_CALL_uint32(REGISTER_GPU);
TF_CALL_GPU_ALL_TYPES(REGISTER_GPU);

// A special GPU kernel for int32.
// TODO(b/25387198): Also enable int32 in device memory. This kernel
// registration requires all int32 inputs and outputs to be in host memory.
REGISTER_KERNEL_BUILDER(Name("StridedSlice")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("input")
                            .HostMemory("begin")
                            .HostMemory("end")
                            .HostMemory("strides")
                            .HostMemory("output"),
                        StridedSliceOp<CPUDevice, int32>);
REGISTER_KERNEL_BUILDER(Name("StridedSliceGrad")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("shape")
                            .HostMemory("begin")
                            .HostMemory("end")
                            .HostMemory("strides")
                            .HostMemory("dy")
                            .HostMemory("output"),
                        StridedSliceGradOp<CPUDevice, int32>);
REGISTER_KERNEL_BUILDER(Name("StridedSliceAssign")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("ref")
                            .HostMemory("begin")
                            .HostMemory("end")
                            .HostMemory("strides"),
                        StridedSliceAssignOp<CPUDevice, int32, false>);
REGISTER_KERNEL_BUILDER(Name("ResourceStridedSliceAssign")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("ref")
                            .HostMemory("begin")
                            .HostMemory("end")
                            .HostMemory("strides"),
                        StridedSliceAssignOp<CPUDevice, int32, false>);
REGISTER_KERNEL_BUILDER(Name("TensorStridedSliceUpdate")
                            .Device(DEVICE_GPU)
                            .TypeConstraint<int32>("T")
                            .HostMemory("input")
                            .HostMemory("begin")
                            .HostMemory("end")
                            .HostMemory("strides"),
                        StridedSliceAssignOp<CPUDevice, int32, true>);
#undef REGISTER_GPU

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
