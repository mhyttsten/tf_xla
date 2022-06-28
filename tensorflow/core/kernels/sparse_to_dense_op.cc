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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparse_to_dense_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_to_dense_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparse_to_dense_opDTcc() {
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

// See core/ops/sparse_ops.cc for documentation.
//
// NOTE: the operations in this file only are suitable for execution
// on CPUs.

#define EIGEN_USE_THREADS

#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/util/ptr_util.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/kernels/gpu_utils.h"
#include "tensorflow/core/kernels/sparse_to_dense_op_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace tensorflow {

namespace {

Status CheckSparseToDenseShapes(const Tensor& indices,
                                const Tensor& output_shape,
                                const Tensor& sparse_values,
                                const Tensor& default_value) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_to_dense_opDTcc mht_0(mht_0_v, 222, "", "./tensorflow/core/kernels/sparse_to_dense_op.cc", "CheckSparseToDenseShapes");

  // sparse_indices
  if (indices.dims() > 2) {
    return errors::InvalidArgument(
        "sparse_indices should be a scalar, vector, or matrix, "
        "got shape ",
        indices.shape().DebugString());
  }
  const int64_t num_elems = indices.dims() > 0 ? indices.dim_size(0) : 1;
  const int64_t num_dims = indices.dims() > 1 ? indices.dim_size(1) : 1;

  // output_shape
  if (!TensorShapeUtils::IsVector(output_shape.shape())) {
    return errors::InvalidArgument("output_shape must be rank 1, got shape ",
                                   output_shape.shape().DebugString());
  }

  if (output_shape.NumElements() != num_dims) {
    return errors::InvalidArgument(
        "output_shape has incorrect number of elements: ",
        output_shape.NumElements(), " should be: ", num_dims);
  }

  // sparse_values
  const int64_t num_values = sparse_values.NumElements();
  if (sparse_values.dims() != 0 &&
      (sparse_values.dims() != 1 || num_values != num_elems)) {
    return errors::InvalidArgument("sparse_values has incorrect shape ",
                                   sparse_values.shape().DebugString(),
                                   ", should be [] or [", num_elems, "]");
  }

  // default_value
  if (!TensorShapeUtils::IsScalar(default_value.shape())) {
    return errors::InvalidArgument("default_value should be a scalar.");
  }
  return Status::OK();
}

}  // end namespace

// Operator to convert sparse representations to dense.
template <typename T, typename Index>
class SparseToDense : public OpKernel {
 public:
  explicit SparseToDense(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_to_dense_opDTcc mht_1(mht_1_v, 270, "", "./tensorflow/core/kernels/sparse_to_dense_op.cc", "SparseToDense");

    OP_REQUIRES_OK(context,
                   context->GetAttr("validate_indices", &validate_indices_));
  }

  void Compute(OpKernelContext* c) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_to_dense_opDTcc mht_2(mht_2_v, 278, "", "./tensorflow/core/kernels/sparse_to_dense_op.cc", "Compute");

    const Tensor& indices = c->input(0);
    const Tensor& output_shape = c->input(1);
    const Tensor& sparse_values = c->input(2);
    const Tensor& default_value = c->input(3);
    OP_REQUIRES_OK(c, CheckSparseToDenseShapes(indices, output_shape,
                                               sparse_values, default_value));

    const int64_t num_elems = indices.dims() > 0 ? indices.dim_size(0) : 1;
    const int64_t num_dims = indices.dims() > 1 ? indices.dim_size(1) : 1;

    auto output_shape_vec = output_shape.flat<Index>();
    TensorShape output_tensor_shape;
    OP_REQUIRES_OK(c, TensorShapeUtils::MakeShape(output_shape_vec.data(),
                                                  output_shape_vec.size(),
                                                  &output_tensor_shape));
    Tensor* output = nullptr;
    OP_REQUIRES_OK(c, c->allocate_output(0, output_tensor_shape, &output));

    const Tensor* indices_shaped;
    std::unique_ptr<Tensor> indices_shaped_holder;
    if (indices.dtype() == DT_INT64 && indices.dims() == 2) {
      indices_shaped = &indices;
    } else {
      TensorShape ix_shape({num_elems, num_dims});
      indices_shaped_holder = MakeUnique<Tensor>(DT_INT64, ix_shape);
      indices_shaped = indices_shaped_holder.get();
      if (indices.dtype() == DT_INT64) {
        CHECK(indices_shaped_holder->CopyFrom(indices, ix_shape));
      } else {
        indices_shaped_holder->matrix<int64_t>() =
            indices.shaped<Index, 2>(ix_shape.dim_sizes())
                .template cast<int64_t>();
      }
    }

    // If we received a scalar, we'll need to create a new
    // tensor with copies of the values as a vec.
    const Tensor* sparse_values_b;
    std::unique_ptr<Tensor> sparse_values_b_holder;

    if (TensorShapeUtils::IsScalar(sparse_values.shape())) {
      sparse_values_b_holder = MakeUnique<Tensor>(DataTypeToEnum<T>::value,
                                                  TensorShape({num_elems}));
      sparse_values_b = sparse_values_b_holder.get();
      sparse_values_b_holder->vec<T>().setConstant(sparse_values.scalar<T>()());
    } else {
      sparse_values_b = &sparse_values;
    }

    // Assume SparseTensor is lexicographically sorted.
    gtl::InlinedVector<int64_t, 8> order(output->shape().dims());
    std::iota(order.begin(), order.end(), 0);
    sparse::SparseTensor st;
    OP_REQUIRES_OK(
        c, sparse::SparseTensor::Create(*indices_shaped, *sparse_values_b,
                                        output->shape(), order, &st));

    if (validate_indices_) {
      OP_REQUIRES_OK(c, st.IndicesValid());
    }

    output->flat<T>().setConstant(default_value.scalar<T>()());
    OP_REQUIRES(c, st.template ToDense<T>(output, false /* initialize */),
                errors::InvalidArgument(
                    "Indices are not valid (out of bounds).  Shape: ",
                    output->shape().DebugString()));
  }

 private:
  bool validate_indices_;
};

#define REGISTER_KERNELS(type, index_type)                             \
  REGISTER_KERNEL_BUILDER(Name("SparseToDense")                        \
                              .Device(DEVICE_CPU)                      \
                              .TypeConstraint<type>("T")               \
                              .TypeConstraint<index_type>("Tindices"), \
                          SparseToDense<type, index_type>);

#define REGISTER_KERNELS_ALL(type) \
  REGISTER_KERNELS(type, int32);   \
  REGISTER_KERNELS(type, int64_t);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_KERNELS_ALL);
REGISTER_KERNELS_ALL(bool);
REGISTER_KERNELS_ALL(tstring);
REGISTER_KERNELS_ALL(complex64);
REGISTER_KERNELS_ALL(complex128);

#undef REGISTER_KERNELS_ALL
#undef REGISTER_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
template <typename T, typename Index>
class SparseToDenseGPU : public AsyncOpKernel {
 public:
  explicit SparseToDenseGPU(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_to_dense_opDTcc mht_3(mht_3_v, 379, "", "./tensorflow/core/kernels/sparse_to_dense_op.cc", "SparseToDenseGPU");

    OP_REQUIRES_OK(context,
                   context->GetAttr("validate_indices", &validate_indices_));
  }

  void ComputeAsync(OpKernelContext* c, DoneCallback done) final {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_to_dense_opDTcc mht_4(mht_4_v, 387, "", "./tensorflow/core/kernels/sparse_to_dense_op.cc", "ComputeAsync");

    auto* stream = c->op_device_context()->stream();
    OP_REQUIRES_ASYNC(c, stream, errors::Internal("No GPU stream available."),
                      done);

    const Tensor& indices = c->input(0);
    const Tensor& output_shape = c->input(1);
    const Tensor& sparse_values = c->input(2);
    const Tensor& default_value = c->input(3);
    OP_REQUIRES_OK_ASYNC(c,
                         CheckSparseToDenseShapes(indices, output_shape,
                                                  sparse_values, default_value),
                         done);

    auto output_shape_vec = output_shape.flat<Index>();
    TensorShape output_tensor_shape;
    OP_REQUIRES_OK_ASYNC(c,
                         TensorShapeUtils::MakeShape(output_shape_vec.data(),
                                                     output_shape_vec.size(),
                                                     &output_tensor_shape),
                         done);
    Tensor* output = nullptr;
    OP_REQUIRES_OK_ASYNC(c, c->allocate_output(0, output_tensor_shape, &output),
                         done);

    Tensor output_shape_tensor;
    OP_REQUIRES_OK_ASYNC(
        c,
        c->allocate_temp(DataTypeToEnum<Index>::value,
                         {output_shape_vec.size()}, &output_shape_tensor),
        done);
    auto output_shape_data =
        AsDeviceMemory(output_shape_tensor.template flat<Index>().data(),
                       output_shape_tensor.template flat<Index>().size());
    OP_REQUIRES_ASYNC(
        c,
        stream
            ->ThenMemcpy(&output_shape_data, output_shape_vec.data(),
                         output_shape_tensor.NumElements() * sizeof(Index))
            .ok(),
        errors::InvalidArgument(
            "failed to copy output_shape vector from host to "
            "device in SparseToDenseOp"),
        done);

    functor::LaunchSparseToDense<T, Index>()(
        c, done, this, validate_indices_, indices, sparse_values,
        output_shape_tensor, default_value.scalar<T>()(), output);
  }

 private:
  bool validate_indices_;
};

// TODO(b/184077412): SparseToDense causes an illegal access error.

#define REGISTER_GPU_KERNELS(type, index_type)                         \
  REGISTER_KERNEL_BUILDER(Name("SparseToDense")                        \
                              .Device(DEVICE_GPU)                      \
                              .HostMemory("default_value")             \
                              .HostMemory("output_shape")              \
                              .TypeConstraint<type>("T")               \
                              .TypeConstraint<index_type>("Tindices"), \
                          SparseToDenseGPU<type, index_type>);

#define REGISTER_GPU_KERNELS_ALL(type) \
  REGISTER_GPU_KERNELS(type, int32);   \
  REGISTER_GPU_KERNELS(type, int64_t);

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS_ALL);
TF_CALL_INTEGRAL_TYPES(REGISTER_GPU_KERNELS_ALL)
REGISTER_GPU_KERNELS_ALL(bool)

#undef REGISTER_GPU_KERNELS_ALL
#undef REGISTER_GPU_KERNELS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
