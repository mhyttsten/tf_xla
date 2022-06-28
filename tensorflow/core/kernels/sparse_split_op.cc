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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparse_split_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_split_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparse_split_opDTcc() {
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

#include "tensorflow/core/kernels/sparse_split_op.h"

#include <vector>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

namespace functor {

template <typename T>
struct SparseSplitFunctor<CPUDevice, T> {
  void operator()(OpKernelContext* context, const Tensor& input_indices,
                  const Tensor& input_values, const TensorShape& dense_shape,
                  const int64_t axis, const int num_split,
                  typename AsyncOpKernel::DoneCallback done) {
    (void)done;  // Unused (only used in GPU implementation)
    sparse::SparseTensor sparse_tensor;
    OP_REQUIRES_OK(context,
                   sparse::SparseTensor::Create(input_indices, input_values,
                                                dense_shape, &sparse_tensor));

    std::vector<sparse::SparseTensor> outputs;
    OP_REQUIRES_OK(context, sparse::SparseTensor::Split<T>(
                                sparse_tensor, axis, num_split, &outputs));

    for (int slice_index = 0; slice_index < num_split; ++slice_index) {
      context->set_output(slice_index, outputs[slice_index].indices());
      context->set_output(slice_index + num_split,
                          outputs[slice_index].values());
      Tensor* shape = nullptr;
      OP_REQUIRES_OK(context, context->allocate_output(
                                  slice_index + 2 * num_split,
                                  {outputs[slice_index].dims()}, &shape));
      auto output_shape = outputs[slice_index].shape();
      for (int dim = 0; dim < outputs[slice_index].dims(); ++dim) {
        shape->vec<int64_t>()(dim) = output_shape[dim];
      }
    }
  }
};

}  // namespace functor

namespace {

template <typename Device, typename T>
void SparseSplitOpImpl(OpKernelContext* context, int num_split,
                       AsyncOpKernel::DoneCallback done = nullptr) {
  // Note that setting this empty lambda as the default parameter value directly
  // can cause strange compiler/linker errors, so we do it like this instead.
  if (!done) {
    done = [] {};
  }

  const Tensor& input_axis = context->input(0);
  const Tensor& input_indices = context->input(1);
  const Tensor& input_values = context->input(2);
  const Tensor& input_shape = context->input(3);

  OP_REQUIRES_ASYNC(context, TensorShapeUtils::IsScalar(input_axis.shape()),
                    errors::InvalidArgument(
                        "Input axis should be a scalar but received shape ",
                        input_axis.shape().DebugString()),
                    done);
  OP_REQUIRES_ASYNC(context, TensorShapeUtils::IsMatrix(input_indices.shape()),
                    errors::InvalidArgument(
                        "Input indices should be a matrix but received shape ",
                        input_indices.shape().DebugString()),
                    done);
  OP_REQUIRES_ASYNC(context, TensorShapeUtils::IsVector(input_values.shape()),
                    errors::InvalidArgument(
                        "Input values should be a vector but received shape ",
                        input_indices.shape().DebugString()),
                    done);
  OP_REQUIRES_ASYNC(context, TensorShapeUtils::IsVector(input_shape.shape()),
                    errors::InvalidArgument(
                        "Input shape should be a vector but received shape ",
                        input_shape.shape().DebugString()),
                    done);

  const int64_t axis_input = input_axis.scalar<int64_t>()();
  const int64_t input_rank = input_shape.vec<int64_t>().size();
  const int64_t axis = (axis_input < 0) ? input_rank + axis_input : axis_input;

  OP_REQUIRES_ASYNC(
      context, axis >= 0 && axis < input_rank,
      errors::InvalidArgument("Input axis should be in range [", -input_rank,
                              ", ", input_rank, "), got ", axis_input),
      done);

  OP_REQUIRES_ASYNC(
      context, num_split >= 1 && num_split <= input_shape.vec<int64_t>()(axis),
      errors::InvalidArgument("Input num_split should be between 1 "
                              "and the splitting dimension size (",
                              input_shape.vec<int64_t>()(axis), "), got ",
                              num_split),
      done);

  // Prevent overflow by constructing the dense shape separately
  TensorShape dense_shape;
  const auto input_shape_flat = input_shape.flat<int64_t>();
  for (int i = 0; i < input_shape.NumElements(); i++) {
    OP_REQUIRES_OK_ASYNC(
        context, dense_shape.AddDimWithStatus(input_shape_flat(i)), done);
  }

  functor::SparseSplitFunctor<Device, T>()(context, input_indices, input_values,
                                           dense_shape, axis, num_split, done);
}

}  // namespace

template <typename T>
class SparseSplitOp : public OpKernel {
 public:
  explicit SparseSplitOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_split_opDTcc mht_0(mht_0_v, 307, "", "./tensorflow/core/kernels/sparse_split_op.cc", "SparseSplitOp");

    OP_REQUIRES_OK(context, context->GetAttr("num_split", &num_split_));
  }

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_split_opDTcc mht_1(mht_1_v, 314, "", "./tensorflow/core/kernels/sparse_split_op.cc", "Compute");

    SparseSplitOpImpl<CPUDevice, T>(context, num_split_);
  }

 private:
  int num_split_;
};

#define REGISTER_KERNELS(type)                                          \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("SparseSplit").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SparseSplitOp<type>)

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

typedef Eigen::GpuDevice GPUDevice;

// The GPU implementation is async because it requires waiting for a
// host->device memcpy before the output is allocated (similar to
// SegmentSumGPUOp).
template <typename T>
class SparseSplitGPUOp : public AsyncOpKernel {
 public:
  explicit SparseSplitGPUOp(OpKernelConstruction* context)
      : AsyncOpKernel(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_split_opDTcc mht_2(mht_2_v, 344, "", "./tensorflow/core/kernels/sparse_split_op.cc", "SparseSplitGPUOp");

    OP_REQUIRES_OK(context, context->GetAttr("num_split", &num_split_));
  }

  void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_split_opDTcc mht_3(mht_3_v, 351, "", "./tensorflow/core/kernels/sparse_split_op.cc", "ComputeAsync");

    SparseSplitOpImpl<GPUDevice, T>(context, num_split_, done);
  }

 private:
  int num_split_;
};

#define REGISTER_KERNELS(type)                            \
  REGISTER_KERNEL_BUILDER(Name("SparseSplit")             \
                              .Device(DEVICE_GPU)         \
                              .HostMemory("split_dim")    \
                              .HostMemory("shape")        \
                              .HostMemory("output_shape") \
                              .TypeConstraint<type>("T"), \
                          SparseSplitGPUOp<type>)
TF_CALL_POD_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
