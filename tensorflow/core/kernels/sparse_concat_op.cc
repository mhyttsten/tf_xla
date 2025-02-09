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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparse_concat_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_concat_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparse_concat_opDTcc() {
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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/sparse_concat_op.h"

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/util/overflow.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace functor {

template <typename T>
struct SparseConcatFunctor<CPUDevice, T> {
  void operator()(OpKernelContext* context, const OpInputList& inds,
                  const OpInputList& vals, const OpInputList& shapes,
                  int concat_dim) {
    const int N = inds.size();
    const TensorShape input_shape(shapes[0].vec<int64_t>());
    const int input_rank = input_shape.dims();

    // The input and output sparse tensors are assumed to be ordered along
    // increasing dimension number. But in order for concat to work properly,
    // order[0] must be concat_dim. So we will reorder the inputs to the
    // concat ordering, concatenate, then reorder back to the standard order.
    // We make a deep copy of the input tensors to ensure that the in-place
    // reorder doesn't create race conditions for other ops that may be
    // concurrently reading the indices and values tensors.

    gtl::InlinedVector<int64, 8> std_order(input_rank);
    std::iota(std_order.begin(), std_order.end(), 0);

    std::vector<int64_t> concat_order;
    concat_order.reserve(input_rank);
    concat_order.push_back(concat_dim);
    for (int j = 0; j < input_rank; ++j) {
      if (j != concat_dim) {
        concat_order.push_back(j);
      }
    }

    std::vector<sparse::SparseTensor> sp_inputs;
    for (int i = 0; i < N; ++i) {
      const TensorShape current_shape(shapes[i].vec<int64_t>());
      sparse::SparseTensor tensor;
      OP_REQUIRES_OK(context,
                     sparse::SparseTensor::Create(
                         tensor::DeepCopy(inds[i]), tensor::DeepCopy(vals[i]),
                         current_shape, std_order, &tensor));
      sp_inputs.push_back(std::move(tensor));
      sp_inputs[i].Reorder<T>(concat_order);
    }

    sparse::SparseTensor concat = sparse::SparseTensor::Concat<T>(sp_inputs);
    concat.Reorder<T>(std_order);

    context->set_output(0, concat.indices());
    context->set_output(1, concat.values());
  }
};

}  // namespace functor

template <typename Device, typename T>
class SparseConcatOp : public OpKernel {
 public:
  explicit SparseConcatOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_concat_opDTcc mht_0(mht_0_v, 264, "", "./tensorflow/core/kernels/sparse_concat_op.cc", "SparseConcatOp");

    OP_REQUIRES_OK(context, context->GetAttr("concat_dim", &concat_dim_attr_));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_concat_opDTcc mht_1(mht_1_v, 271, "", "./tensorflow/core/kernels/sparse_concat_op.cc", "Compute");

    OpInputList inds;
    OP_REQUIRES_OK(context, context->input_list("indices", &inds));
    const int N = inds.size();
    for (int i = 0; i < N; i++) {
      OP_REQUIRES(context, TensorShapeUtils::IsMatrix(inds[i].shape()),
                  errors::InvalidArgument(
                      "Input indices should be a matrix but received shape ",
                      inds[i].shape().DebugString(), " at position ", i));
    }

    OpInputList vals;
    OP_REQUIRES_OK(context, context->input_list("values", &vals));
    OP_REQUIRES(context, vals.size() == N,
                errors::InvalidArgument("Expected ", N, " input values, got ",
                                        vals.size()));
    for (int i = 0; i < N; i++) {
      OP_REQUIRES(context, TensorShapeUtils::IsVector(vals[i].shape()),
                  errors::InvalidArgument(
                      "Input values should be a vector but received shape ",
                      vals[i].shape().DebugString(), " at position ", i));
    }

    OpInputList shapes;
    OP_REQUIRES_OK(context, context->input_list("shapes", &shapes));
    OP_REQUIRES(context, shapes.size() == N,
                errors::InvalidArgument("Expected ", N, " input shapes, got ",
                                        shapes.size()));
    bool overflow_ocurred = false;
    for (int i = 0; i < N; i++) {
      int64_t new_num_elements = 1;
      OP_REQUIRES(context, TensorShapeUtils::IsVector(shapes[i].shape()),
                  errors::InvalidArgument(
                      "Input shapes should be a vector but received shape ",
                      shapes[i].shape().DebugString(), " at position ", i));
      auto input_shape_vector = shapes[i].vec<int64_t>();
      for (int j = 0; j < input_shape_vector.size(); j++) {
        new_num_elements =
            MultiplyWithoutOverflow(new_num_elements, input_shape_vector(j));
        if (new_num_elements < 0) {
          overflow_ocurred = true;
          break;
        }
      }

      if (overflow_ocurred) {
        break;
      }
    }

    OP_REQUIRES(
        context, !overflow_ocurred,
        errors::Internal("Encountered overflow from large input shape."));

    const TensorShape input_shape(shapes[0].vec<int64_t>());
    const int input_rank = input_shape.dims();
    const int concat_dim = (concat_dim_attr_ < 0)
                               ? input_rank + concat_dim_attr_
                               : concat_dim_attr_;
    OP_REQUIRES(context, concat_dim >= 0 && concat_dim < input_rank,
                errors::InvalidArgument("Concat dimension must be in range [",
                                        -input_rank, ", ", input_rank,
                                        "), got ", concat_dim_attr_));
    TensorShape output_shape = input_shape;
    for (int i = 1; i < N; ++i) {
      const TensorShape current_shape(shapes[i].vec<int64_t>());
      OP_REQUIRES(
          context, current_shape.dims() == input_rank,
          errors::InvalidArgument(
              "Ranks of all input tensors must match: expected ", input_rank,
              " but got ", current_shape.dims(), " at position ", i));
      for (int j = 0; j < input_rank; ++j) {
        if (j != concat_dim) {
          OP_REQUIRES(
              context, input_shape.dim_size(j) == current_shape.dim_size(j),
              errors::InvalidArgument(
                  "Input shapes must match: expected ", input_shape.dim_size(j),
                  " for dimension ", j, " but got ", current_shape.dim_size(j),
                  " at position ", i));
        } else {
          output_shape.set_dim(
              j, output_shape.dim_size(j) + current_shape.dim_size(j));
        }
      }
    }

    Tensor* output_shape_out = nullptr;
    OP_REQUIRES_OK(
        context, context->allocate_output(2, TensorShape({output_shape.dims()}),
                                          &output_shape_out));
    auto output_shape_t = output_shape_out->vec<int64_t>();
    for (int j = 0; j < output_shape.dims(); ++j) {
      output_shape_t(j) = output_shape.dim_size(j);
    }

    int64_t output_nnz = 0;
    for (int i = 0; i < N; ++i) {
      output_nnz += inds[i].dim_size(0);
    }
    if (output_nnz == 0) {
      Tensor* output_inds = nullptr;
      OP_REQUIRES_OK(context,
                     context->allocate_output(0, TensorShape({0, input_rank}),
                                              &output_inds));
      Tensor* output_vals = nullptr;
      OP_REQUIRES_OK(
          context, context->allocate_output(1, TensorShape({0}), &output_vals));
      return;  // No work to do
    }

    functor::SparseConcatFunctor<Device, T>()(context, inds, vals, shapes,
                                              concat_dim);
  }

 private:
  int concat_dim_attr_;
};

#define REGISTER_KERNELS(type)                                           \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("SparseConcat").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SparseConcatOp<CPUDevice, type>)

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

typedef Eigen::GpuDevice GPUDevice;

#define REGISTER_KERNELS(type)                            \
  REGISTER_KERNEL_BUILDER(Name("SparseConcat")            \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("shapes")       \
                              .HostMemory("output_shape") \
                              .TypeConstraint<type>("T"), \
                          SparseConcatOp<GPUDevice, type>)
TF_CALL_POD_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
