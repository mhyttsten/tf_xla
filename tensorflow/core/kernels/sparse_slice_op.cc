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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparse_slice_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_slice_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparse_slice_opDTcc() {
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

#include "tensorflow/core/kernels/sparse_slice_op.h"

#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace functor {

template <typename T>
struct SparseSliceFunctor<CPUDevice, T> {
  void operator()(OpKernelContext* context, const Tensor& input_indices,
                  const Tensor& input_values, const Tensor& input_shape,
                  const Tensor& input_start, const Tensor& input_size,
                  typename AsyncOpKernel::DoneCallback done) const {
    (void)done;  // Unused (only used in GPU implementation)
    const int input_dims = input_shape.NumElements();

    sparse::SparseTensor sparse_tensor;
    TensorShape sparse_tensor_shape;
    OP_REQUIRES_OK(context,
                   TensorShapeBase<TensorShape>::BuildTensorShapeBase(
                       input_shape.vec<int64_t>(), &sparse_tensor_shape));
    OP_REQUIRES_OK(context, sparse::SparseTensor::Create(
                                input_indices, input_values,
                                sparse_tensor_shape, &sparse_tensor));

    const gtl::ArraySlice<int64_t> start(input_start.flat<int64_t>().data(),
                                         input_dims);
    const gtl::ArraySlice<int64_t> size(input_size.flat<int64_t>().data(),
                                        input_dims);

    const StatusOr<sparse::SparseTensor> output_or =
        sparse::SparseTensor::Slice<T>(sparse_tensor, start, size);
    OP_REQUIRES_OK(context, output_or.status());
    auto output = output_or.ValueOrDie();

    context->set_output(0, output.indices());
    context->set_output(1, output.values());

    TensorShape output_shape;
    OP_REQUIRES_OK(context, TensorShapeBase<TensorShape>::BuildTensorShapeBase(
                                output.shape(), &output_shape));

    TensorShape allocated_shape;
    OP_REQUIRES_OK(context, TensorShapeBase<TensorShape>::BuildTensorShapeBase(
                                {output_shape.dims()}, &allocated_shape));

    Tensor* shape = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(2, allocated_shape, &shape));
    for (int dim = 0; dim < output_shape.dims(); ++dim) {
      shape->vec<int64_t>()(dim) = output_shape.dim_size(dim);
    }
  }
};

}  // namespace functor

namespace {

template <typename Device, typename T>
void SparseSliceOpImpl(OpKernelContext* context,
                       typename AsyncOpKernel::DoneCallback done = nullptr) {
  // Note that setting this empty lambda as the default parameter value directly
  // can cause strange compiler/linker errors, so we do it like this instead.
  if (!done) {
    done = [] {};
  }

  const Tensor& input_indices = context->input(0);
  const Tensor& input_values = context->input(1);
  const Tensor& input_shape = context->input(2);
  const Tensor& input_start = context->input(3);
  const Tensor& input_size = context->input(4);

  OP_REQUIRES_ASYNC(context, TensorShapeUtils::IsMatrix(input_indices.shape()),
                    errors::InvalidArgument(
                        "Input indices should be a matrix but received shape ",
                        input_indices.shape().DebugString()),
                    done);
  OP_REQUIRES_ASYNC(context, TensorShapeUtils::IsVector(input_values.shape()),
                    errors::InvalidArgument(
                        "Input values should be a vector but received shape ",
                        input_values.shape().DebugString()),
                    done);
  OP_REQUIRES_ASYNC(context, TensorShapeUtils::IsVector(input_shape.shape()),
                    errors::InvalidArgument(
                        "Input shape should be a vector but received shape ",
                        input_shape.shape().DebugString()),
                    done);
  OP_REQUIRES_ASYNC(context, TensorShapeUtils::IsVector(input_start.shape()),
                    errors::InvalidArgument(
                        "Input start should be a vector but received shape ",
                        input_start.shape().DebugString()),
                    done);
  OP_REQUIRES_ASYNC(context, TensorShapeUtils::IsVector(input_size.shape()),
                    errors::InvalidArgument(
                        "Input size should be a vector but received shape ",
                        input_size.shape().DebugString()),
                    done);

  const int input_dims = input_shape.NumElements();
  OP_REQUIRES_ASYNC(context, input_dims == input_start.NumElements(),
                    errors::InvalidArgument(
                        "Expected start to be a vector of length ", input_dims,
                        " but got length ", input_start.NumElements()),
                    done);

  OP_REQUIRES_ASYNC(context, input_dims == input_size.NumElements(),
                    errors::InvalidArgument(
                        "Expected size to be a vector of length ", input_dims,
                        " but got length ", input_size.NumElements()),
                    done);

  functor::SparseSliceFunctor<Device, T>()(context, input_indices, input_values,
                                           input_shape, input_start, input_size,
                                           done);
}

}  // namespace

template <typename Device, typename T>
class SparseSliceOp : public OpKernel {
 public:
  explicit SparseSliceOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_slice_opDTcc mht_0(mht_0_v, 318, "", "./tensorflow/core/kernels/sparse_slice_op.cc", "SparseSliceOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_slice_opDTcc mht_1(mht_1_v, 323, "", "./tensorflow/core/kernels/sparse_slice_op.cc", "Compute");

    SparseSliceOpImpl<Device, T>(context);
  }
};

#define REGISTER_KERNELS(type)                                          \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSlice").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SparseSliceOp<CPUDevice, type>)

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
class SparseSliceGPUOp : public AsyncOpKernel {
 public:
  explicit SparseSliceGPUOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_slice_opDTcc mht_2(mht_2_v, 347, "", "./tensorflow/core/kernels/sparse_slice_op.cc", "SparseSliceGPUOp");
}

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_slice_opDTcc mht_3(mht_3_v, 352, "", "./tensorflow/core/kernels/sparse_slice_op.cc", "ComputeAsync");

    SparseSliceOpImpl<GPUDevice, T>(context, done);
  }
};

#define REGISTER_KERNELS(type)                            \
  REGISTER_KERNEL_BUILDER(Name("SparseSlice")             \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("shape")        \
                              .HostMemory("start")        \
                              .HostMemory("size")         \
                              .HostMemory("output_shape") \
                              .TypeConstraint<type>("T"), \
                          SparseSliceGPUOp<type>)

TF_CALL_POD_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
