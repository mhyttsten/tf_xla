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
class MHTracer_DTPStensorflowPScorePSkernelsPSreshape_utilDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSreshape_utilDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSreshape_utilDTcc() {
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

#include "tensorflow/core/kernels/reshape_util.h"

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op_requires.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

namespace functor {

template <>
struct ReshapeSparseTensorFunctor<CPUDevice> {
  Status operator()(OpKernelContext *context, const TensorShape &input_shape,
                    const TensorShape &output_shape,
                    typename TTypes<int64_t>::ConstMatrix input_indices,
                    typename TTypes<int64_t>::Matrix output_indices) const {
    (void)context;  // Unused (only used in GPU implementation)
    const int64_t input_rank = input_shape.dims();
    const int64_t output_rank = output_shape.dims();
    const int64_t nnz = input_indices.dimension(0);
    gtl::InlinedVector<int64_t, 8> input_strides(input_rank);
    if (input_rank > 0) {
      input_strides[input_rank - 1] = 1;
      for (int d = input_rank - 2; d >= 0; --d) {
        input_strides[d] = input_strides[d + 1] * input_shape.dim_size(d + 1);
      }
    }

    gtl::InlinedVector<int64_t, 8> output_strides(output_rank);
    if (output_rank > 0) {
      output_strides[output_rank - 1] = 1;
      for (int d = output_rank - 2; d >= 0; --d) {
        output_strides[d] =
            output_strides[d + 1] * output_shape.dim_size(d + 1);
      }
    }

    for (int i = 0; i < nnz; ++i) {
      int64_t id = 0;
      for (int j = 0; j < input_rank; ++j) {
        id += input_indices(i, j) * input_strides[j];
      }
      for (int j = 0; j < output_rank; ++j) {
        output_indices(i, j) = id / output_strides[j];
        id %= output_strides[j];
      }
    }
    return Status::OK();
  }
};

}  // namespace functor

template <typename Device>
void ReshapeSparseTensor(OpKernelContext *context,
                         const Tensor &input_indices_in,
                         const Tensor &input_shape_in,
                         const Tensor &target_shape_in, int output_indices_idx,
                         int output_shape_idx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSreshape_utilDTcc mht_0(mht_0_v, 258, "", "./tensorflow/core/kernels/reshape_util.cc", "ReshapeSparseTensor");

  OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_indices_in.shape()),
              errors::InvalidArgument(
                  "Input indices should be a matrix but received shape ",
                  input_indices_in.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsVector(input_shape_in.shape()),
              errors::InvalidArgument(
                  "Input shape should be a vector but received shape ",
                  input_shape_in.shape().DebugString()));
  OP_REQUIRES(context, TensorShapeUtils::IsVector(target_shape_in.shape()),
              errors::InvalidArgument(
                  "Target shape should be a vector but received shape ",
                  target_shape_in.shape().DebugString()));

  const int64_t output_rank = target_shape_in.NumElements();
  TensorShape input_shape;
  OP_REQUIRES_OK(context, TensorShape::BuildTensorShape(
                              input_shape_in.vec<int64_t>(), &input_shape));
  const int64_t dense_size = input_shape.num_elements();
  const int64_t nnz = input_indices_in.shape().dim_size(0);

  // Compute the output shape. Determine product of specified dimensions, and
  // find the index of the unspecified one.
  TensorShape output_shape;
  int64_t product = 1;
  int unknown_index = -1;
  auto target_shape = target_shape_in.vec<int64_t>();
  for (int d = 0; d < output_rank; ++d) {
    const int64_t size = target_shape(d);
    if (size == -1) {
      OP_REQUIRES(
          context, unknown_index == -1,
          errors::InvalidArgument("only one output dimension may be -1, "
                                  "not both ",
                                  unknown_index, " and ", d));
      unknown_index = d;
      output_shape.AddDim(1);
    } else {
      OP_REQUIRES(context, size >= 0,
                  errors::InvalidArgument("size ", d,
                                          " must be non-negative, not ", size));
      product *= size;
      output_shape.AddDim(size);
    }
  }
  if (unknown_index != -1) {
    OP_REQUIRES(
        context, product > 0,
        errors::InvalidArgument("reshape cannot infer the missing "
                                "input size for an empty tensor unless all "
                                "specified input sizes are non-zero"));
    const int64_t missing = dense_size / product;
    OP_REQUIRES(
        context, product * missing == dense_size,
        errors::InvalidArgument(
            "Input to reshape is a SparseTensor with ", dense_size,
            " dense values, but the requested shape requires a multiple of ",
            product, ". input_shape=", input_shape.DebugString(),
            " output_shape=", output_shape.DebugString()));
    output_shape.set_dim(unknown_index, missing);
  }

  OP_REQUIRES(
      context, output_shape.num_elements() == dense_size,
      errors::InvalidArgument("Input to reshape is a tensor with ", dense_size,
                              " dense values, but the requested shape has ",
                              output_shape.num_elements(),
                              ". input_shape=", input_shape.DebugString(),
                              " output_shape=", output_shape.DebugString()));

  // Optimize for reshaping to the same shape.
  if (input_shape == output_shape) {
    context->set_output(output_indices_idx, input_indices_in);
    context->set_output(output_shape_idx, input_shape_in);
    return;
  }

  Tensor *result_shape = nullptr;
  OP_REQUIRES_OK(context, context->allocate_output(output_shape_idx,
                                                   TensorShape({output_rank}),
                                                   &result_shape));
  auto output_shape_vec = result_shape->vec<int64_t>();
  for (int j = 0; j < output_shape.dims(); ++j) {
    output_shape_vec(j) = output_shape.dim_size(j);
  }

  Tensor *result_indices = nullptr;
  OP_REQUIRES_OK(context,
                 context->allocate_output(output_indices_idx,
                                          TensorShape({nnz, output_rank}),
                                          &result_indices));
  if (nnz > 0) {
    OP_REQUIRES(
        context, dense_size > 0 && product > 0,
        errors::InvalidArgument(
            "Input tensor has ", nnz, " non zero elements but input shape (",
            input_shape.DebugString(), ") or output shape (",
            output_shape.DebugString(), ") is empty"));
    OP_REQUIRES_OK(context, functor::ReshapeSparseTensorFunctor<Device>()(
                                context, input_shape, output_shape,
                                input_indices_in.matrix<int64_t>(),
                                result_indices->matrix<int64_t>()));
  }
}

#define EXPLICITLY_INSTANTIATE_FUNCTION(Device)                    \
  template void ReshapeSparseTensor<Device>(                       \
      OpKernelContext * context, const Tensor &input_indices_in,   \
      const Tensor &input_shape_in, const Tensor &target_shape_in, \
      int output_indices_idx, int output_shape_idx)
EXPLICITLY_INSTANTIATE_FUNCTION(CPUDevice);

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
EXPLICITLY_INSTANTIATE_FUNCTION(GPUDevice);
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#undef EXPLICITLY_INSTANTIATE_FUNCTION

}  // namespace tensorflow
