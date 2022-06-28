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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparse_fill_empty_rows_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_fill_empty_rows_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparse_fill_empty_rows_opDTcc() {
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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/sparse_fill_empty_rows_op.h"

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace functor {

template <typename T, typename Tindex>
struct SparseFillEmptyRows<CPUDevice, T, Tindex> {
  Status operator()(OpKernelContext* context, const Tensor& default_value_t,
                    const Tensor& indices_t, const Tensor& values_t,
                    const Tensor& dense_shape_t,
                    typename AsyncOpKernel::DoneCallback done) {
    (void)done;  // Unused (only used in GPU implementation)
    const int kOutputIndicesOutput = 0;
    const int kOutputValuesOutput = 1;
    const int kEmptyRowIndicatorOutput = 2;
    const int kReverseIndexMapOutput = 3;

    const T& default_value = default_value_t.scalar<T>()();
    const auto indices = indices_t.matrix<Tindex>();
    const auto values = values_t.vec<T>();
    const auto dense_shape = dense_shape_t.vec<Tindex>();

    const Tindex N = indices_t.shape().dim_size(0);
    const Tindex dense_rows = dense_shape(0);

    bool* empty_row_indicator = nullptr;
    if (context->output_required(kEmptyRowIndicatorOutput)) {
      Tensor* empty_row_indicator_t = nullptr;
      TensorShape output_shape;
      TF_RETURN_IF_ERROR(
          TensorShape::BuildTensorShape({dense_rows}, &output_shape));
      TF_RETURN_IF_ERROR(context->allocate_output(
          kEmptyRowIndicatorOutput, output_shape, &empty_row_indicator_t));
      empty_row_indicator = empty_row_indicator_t->vec<bool>().data();
    }
    Tindex* reverse_index_map = nullptr;
    if (context->output_required(kReverseIndexMapOutput)) {
      Tensor* reverse_index_map_t = nullptr;
      TensorShape output_shape;
      TF_RETURN_IF_ERROR(TensorShape::BuildTensorShape({N}, &output_shape));
      TF_RETURN_IF_ERROR(context->allocate_output(
          kReverseIndexMapOutput, output_shape, &reverse_index_map_t));
      reverse_index_map = reverse_index_map_t->vec<Tindex>().data();
    }

    int rank = indices_t.shape().dim_size(1);

    if (dense_rows == 0) {
      if (N != 0) {
        return errors::InvalidArgument(
            "Received SparseTensor with dense_shape[0] = 0 but "
            "indices.shape[0] = ",
            N);
      }
      Tensor* output_indices_t;
      TensorShape output_indices_shape;
      TF_RETURN_IF_ERROR(
          TensorShape::BuildTensorShape({0, rank}, &output_indices_shape));
      TF_RETURN_IF_ERROR(context->allocate_output(
          kOutputIndicesOutput, output_indices_shape, &output_indices_t));
      Tensor* output_values_t;
      TF_RETURN_IF_ERROR(context->allocate_output(
          kOutputValuesOutput, TensorShape({0}), &output_values_t));

      // Exit early, nothing more to do.
      return Status::OK();
    }

    bool rows_are_ordered = true;
    Tindex last_indices_row = 0;
    std::vector<Tindex> csr_offset(dense_rows, 0);
    for (int i = 0; i < N; ++i) {
      const Tindex row = indices(i, 0);
      if (row < 0 || row >= dense_rows) {
        return errors::InvalidArgument("indices(", i, ", 0) is invalid: ", row,
                                       " >= ", dense_rows);
      }
      ++csr_offset[row];
      rows_are_ordered = rows_are_ordered & (row >= last_indices_row);
      last_indices_row = row;
    }
    bool all_rows_full = true;
    for (int row = 0; row < dense_rows; ++row) {
      // csr_offset here describes the number of elements in this dense row
      bool row_empty = (csr_offset[row] == 0);
      if (empty_row_indicator) {
        empty_row_indicator[row] = row_empty;
      }
      all_rows_full = all_rows_full & !row_empty;
      // In filled version, each row has at least one element.
      csr_offset[row] = std::max(csr_offset[row], Tindex{1});
      // Update csr_offset to represent the number of elements up to and
      // including dense_row + 1:
      //  csr_offset(0) == #{elements of row 0}
      //  csr_offset(1) == #{elements of row 1} + #{elements of row 0}
      //  ..
      //  csr_offset(i) == starting index for elements in row i + 1.
      if (row > 0) {
        csr_offset[row] += csr_offset[row - 1];
      }
    }

    if (all_rows_full && rows_are_ordered) {
      context->set_output(kOutputIndicesOutput, indices_t);
      context->set_output(kOutputValuesOutput, values_t);
      if (reverse_index_map) {
        for (Tindex i = 0; i < N; ++i) {
          reverse_index_map[i] = i;
        }
      }
    } else {
      Tensor* output_indices_t;
      const Tindex N_full = csr_offset[dense_rows - 1];
      TensorShape output_indices_shape;
      TF_RETURN_IF_ERROR(
          TensorShape::BuildTensorShape({N_full, rank}, &output_indices_shape));
      TF_RETURN_IF_ERROR(context->allocate_output(
          kOutputIndicesOutput, output_indices_shape, &output_indices_t));
      auto output_indices = output_indices_t->matrix<Tindex>();

      Tensor* output_values_t;
      TF_RETURN_IF_ERROR(context->allocate_output(
          kOutputValuesOutput, TensorShape({N_full}), &output_values_t));
      auto output_values = output_values_t->vec<T>();

      std::vector<Tindex> filled_count(dense_rows, 0);

      // Fill in values for rows that are not missing
      for (Tindex i = 0; i < N; ++i) {
        const Tindex row = indices(i, 0);
        Tindex& offset = filled_count[row];
        const Tindex output_i = ((row == 0) ? 0 : csr_offset[row - 1]) + offset;
        offset++;  // Increment the filled count for this row.
        std::copy_n(&indices(i, 0), rank, &output_indices(output_i, 0));
        output_values(output_i) = values(i);
        // We'll need this reverse index map to backprop correctly.
        if (reverse_index_map) {
          reverse_index_map[i] = output_i;
        }
      }

      // Fill in values for rows that are missing
      for (Tindex row = 0; row < dense_rows; ++row) {
        const Tindex row_count = filled_count[row];
        if (row_count == 0) {  // We haven't filled this row
          const Tindex starting_index = (row == 0) ? 0 : csr_offset[row - 1];
          // Remaining index values were set to zero already.
          // Just need to set the row index in the right location.
          output_indices(starting_index, 0) = row;
          for (Tindex col = 1; col < rank; ++col) {
            output_indices(starting_index, col) = 0;
          }
          output_values(starting_index) = default_value;
        }
      }
    }

    return Status::OK();
  }
};

}  // namespace functor

namespace {

template <typename Device, typename T, typename Tindex>
void SparseFillEmptyRowsOpImpl(OpKernelContext* context,
                               AsyncOpKernel::DoneCallback done = nullptr) {
  // Note that setting this empty lambda as the default parameter value directly
  // can cause strange compiler/linker errors, so we do it like this instead.
  if (!done) {
    done = [] {};
  }

  const int kIndicesInput = 0;
  const int kValuesInput = 1;
  const int kDenseShapeInput = 2;
  const int kDefaultValueInput = 3;

  const Tensor& indices_t = context->input(kIndicesInput);
  const Tensor& values_t = context->input(kValuesInput);
  const Tensor& dense_shape_t = context->input(kDenseShapeInput);
  const Tensor& default_value_t = context->input(kDefaultValueInput);

  OP_REQUIRES_ASYNC(
      context, TensorShapeUtils::IsVector(dense_shape_t.shape()),
      errors::InvalidArgument("dense_shape must be a vector, saw: ",
                              dense_shape_t.shape().DebugString()),
      done);
  OP_REQUIRES_ASYNC(context, TensorShapeUtils::IsMatrix(indices_t.shape()),
                    errors::InvalidArgument("indices must be a matrix, saw: ",
                                            indices_t.shape().DebugString()),
                    done);
  OP_REQUIRES_ASYNC(context, TensorShapeUtils::IsVector(values_t.shape()),
                    errors::InvalidArgument("values must be a vector, saw: ",
                                            values_t.shape().DebugString()),
                    done);
  OP_REQUIRES_ASYNC(
      context, indices_t.dim_size(0) == values_t.dim_size(0),
      errors::InvalidArgument("The length of `values` (", values_t.dim_size(0),
                              ") must match the first dimension of `indices` (",
                              indices_t.dim_size(0), ")."),
      done);
  OP_REQUIRES_ASYNC(
      context, TensorShapeUtils::IsScalar(default_value_t.shape()),
      errors::InvalidArgument("default_value must be a scalar, saw: ",
                              default_value_t.shape().DebugString()),
      done);
  // TODO(ebrevdo): add shape checks between values, indices,
  // Also add check that dense rank > 0.
  OP_REQUIRES_ASYNC(context, dense_shape_t.NumElements() != 0,
                    errors::InvalidArgument("Dense shape cannot be empty."),
                    done);

  using FunctorType = functor::SparseFillEmptyRows<Device, T, Tindex>;
  OP_REQUIRES_OK_ASYNC(context,
                       FunctorType()(context, default_value_t, indices_t,
                                     values_t, dense_shape_t, done),
                       done);
}

}  // namespace

template <typename Device, typename T, typename Tindex>
class SparseFillEmptyRowsOp : public OpKernel {
 public:
  explicit SparseFillEmptyRowsOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_fill_empty_rows_opDTcc mht_0(mht_0_v, 434, "", "./tensorflow/core/kernels/sparse_fill_empty_rows_op.cc", "SparseFillEmptyRowsOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_fill_empty_rows_opDTcc mht_1(mht_1_v, 439, "", "./tensorflow/core/kernels/sparse_fill_empty_rows_op.cc", "Compute");

    SparseFillEmptyRowsOpImpl<Device, T, Tindex>(context);
  }
};

#define REGISTER_KERNELS(D, T, Tindex)                   \
  REGISTER_KERNEL_BUILDER(Name("SparseFillEmptyRows")    \
                              .Device(DEVICE_##D)        \
                              .HostMemory("dense_shape") \
                              .TypeConstraint<T>("T"),   \
                          SparseFillEmptyRowsOp<D##Device, T, Tindex>)

#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T, int64)
TF_CALL_ALL_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

#undef REGISTER_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// The GPU implementation is async because it requires waiting for a
// host->device memcpy before the output is allocated (similar to
// SegmentSumGPUOp).
template <typename T, typename Tindex>
class SparseFillEmptyRowsGPUOp : public AsyncOpKernel {
 public:
  explicit SparseFillEmptyRowsGPUOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_fill_empty_rows_opDTcc mht_2(mht_2_v, 469, "", "./tensorflow/core/kernels/sparse_fill_empty_rows_op.cc", "SparseFillEmptyRowsGPUOp");
}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_fill_empty_rows_opDTcc mht_3(mht_3_v, 474, "", "./tensorflow/core/kernels/sparse_fill_empty_rows_op.cc", "ComputeAsync");

    SparseFillEmptyRowsOpImpl<GPUDevice, T, Tindex>(context, done);
  }
};

#define REGISTER_KERNELS(T, Tindex)                      \
  REGISTER_KERNEL_BUILDER(Name("SparseFillEmptyRows")    \
                              .Device(DEVICE_GPU)        \
                              .HostMemory("dense_shape") \
                              .TypeConstraint<T>("T"),   \
                          SparseFillEmptyRowsGPUOp<T, Tindex>)

// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T, Tindex)                                            \
  template <>                                                                  \
  Status SparseFillEmptyRows<GPUDevice, T, Tindex>::operator()(                \
      OpKernelContext* context, const Tensor& default_value_t,                 \
      const Tensor& indices_t, const Tensor& values_t,                         \
      const Tensor& dense_shape_t, typename AsyncOpKernel::DoneCallback done); \
  extern template struct SparseFillEmptyRows<GPUDevice, T, Tindex>;
#define DECLARE_GPU_SPEC_INT64(T) DECLARE_GPU_SPEC(T, int64_t)
TF_CALL_POD_TYPES(DECLARE_GPU_SPEC_INT64)
#undef DECLARE_GPU_SPEC_INT64
#undef DECLARE_GPU_SPEC
}  // namespace functor

#define REGISTER_KERNELS_TINDEX(T) REGISTER_KERNELS(T, int64)
TF_CALL_POD_TYPES(REGISTER_KERNELS_TINDEX)
#undef REGISTER_KERNELS_TINDEX

#undef REGISTER_KERNELS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace functor {

template <typename T, typename Tindex>
struct SparseFillEmptyRowsGrad<CPUDevice, T, Tindex> {
  Status operator()(OpKernelContext* context,
                    typename TTypes<Tindex>::ConstVec reverse_index_map,
                    typename TTypes<T>::ConstVec grad_values,
                    typename TTypes<T>::Vec d_values,
                    typename TTypes<T>::Scalar d_default_value) {
    const CPUDevice& device = context->eigen_device<CPUDevice>();
    const Tindex N = reverse_index_map.dimension(0);
    const Tindex N_full = grad_values.dimension(0);

    T& d_default_value_scalar = d_default_value();
    d_default_value_scalar = T();

    Tensor visited_t;
    TF_RETURN_IF_ERROR(
        context->allocate_temp(DT_BOOL, TensorShape({N_full}), &visited_t));
    auto visited = visited_t.vec<bool>();
    visited.device(device) = visited.constant(false);

    for (int i = 0; i < N; ++i) {
      // Locate the index of the output of the forward prop associated
      // with this location in the input of the forward prop.  Copy
      // the gradient into it.  Mark it as visited.
      int64_t reverse_index = reverse_index_map(i);
      if (reverse_index < 0 || reverse_index >= N_full) {
        return errors::InvalidArgument(
            "Elements in reverse index must be in [0, ", N_full, ") but got ",
            reverse_index);
      }
      d_values(i) = grad_values(reverse_index);
      visited(reverse_index) = true;
    }
    for (int j = 0; j < N_full; ++j) {
      // The default value gradient gets the accumulated remainder of
      // the backprop values (since the default value was used to fill
      // in these slots in the forward calculation).
      if (!visited(j)) {
        d_default_value_scalar += grad_values(j);
      }
    }
    return Status::OK();
  }
};

}  // namespace functor

template <typename Device, typename T, typename Tindex>
class SparseFillEmptyRowsGradOp : public OpKernel {
 public:
  explicit SparseFillEmptyRowsGradOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_fill_empty_rows_opDTcc mht_4(mht_4_v, 565, "", "./tensorflow/core/kernels/sparse_fill_empty_rows_op.cc", "SparseFillEmptyRowsGradOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_fill_empty_rows_opDTcc mht_5(mht_5_v, 570, "", "./tensorflow/core/kernels/sparse_fill_empty_rows_op.cc", "Compute");

    const Tensor* reverse_index_map_t;
    const Tensor* grad_values_t;
    OP_REQUIRES_OK(context,
                   context->input("reverse_index_map", &reverse_index_map_t));
    OP_REQUIRES_OK(context, context->input("grad_values", &grad_values_t));

    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(reverse_index_map_t->shape()),
        errors::InvalidArgument("reverse_index_map must be a vector, saw: ",
                                reverse_index_map_t->shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(grad_values_t->shape()),
                errors::InvalidArgument("grad_values must be a vector, saw: ",
                                        grad_values_t->shape().DebugString()));

    const auto reverse_index_map = reverse_index_map_t->vec<Tindex>();
    const auto grad_values = grad_values_t->vec<T>();

    const Tindex N = reverse_index_map_t->shape().dim_size(0);

    Tensor* d_values_t;
    OP_REQUIRES_OK(context, context->allocate_output(
                                "d_values", TensorShape({N}), &d_values_t));
    auto d_values = d_values_t->vec<T>();
    Tensor* d_default_value_t;
    OP_REQUIRES_OK(context,
                   context->allocate_output("d_default_value", TensorShape({}),
                                            &d_default_value_t));
    auto d_default_value = d_default_value_t->scalar<T>();

    OP_REQUIRES_OK(context,
                   functor::SparseFillEmptyRowsGrad<Device, T, Tindex>()(
                       context, reverse_index_map, grad_values, d_values,
                       d_default_value));
  }
};

#define REGISTER_KERNELS(D, T, Tindex)                    \
  REGISTER_KERNEL_BUILDER(Name("SparseFillEmptyRowsGrad") \
                              .Device(DEVICE_##D)         \
                              .TypeConstraint<T>("T"),    \
                          SparseFillEmptyRowsGradOp<D##Device, T, Tindex>)

#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T, int64)
TF_CALL_NUMBER_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Forward declarations of the functor specializations for GPU.
namespace functor {
#define DECLARE_GPU_SPEC(T, Tindex)                                 \
  template <>                                                       \
  Status SparseFillEmptyRowsGrad<GPUDevice, T, Tindex>::operator()( \
      OpKernelContext* context,                                     \
      typename TTypes<Tindex>::ConstVec reverse_index_map,          \
      typename TTypes<T>::ConstVec grad_values,                     \
      typename TTypes<T>::Vec d_values,                             \
      typename TTypes<T>::Scalar d_default_value);                  \
  extern template struct SparseFillEmptyRowsGrad<GPUDevice, T, Tindex>;
#define DECLARE_GPU_SPEC_INT64(T) DECLARE_GPU_SPEC(T, int64_t)
TF_CALL_REAL_NUMBER_TYPES(DECLARE_GPU_SPEC_INT64);
#undef DECLARE_GPU_SPEC_INT64
#undef DECLARE_GPU_SPEC
}  // namespace functor

#define REGISTER_GPU_KERNELS(T) REGISTER_KERNELS(GPU, T, int64)
TF_CALL_REAL_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef REGISTER_KERNELS
}  // namespace tensorflow
