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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparse_reorder_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_reorder_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparse_reorder_opDTcc() {
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

#include "tensorflow/core/kernels/sparse_reorder_op.h"

#include <algorithm>
#include <numeric>
#include <unordered_map>
#include <utility>

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/util/sparse/sparse_tensor.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
using GPUDevice = Eigen::GpuDevice;
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

namespace functor {

template <typename T>
struct SparseReorderFunctor<CPUDevice, T> {
  void operator()(OpKernelContext* context, const Tensor& input_ind,
                  const Tensor& input_val, const Tensor& input_shape_in) {
    gtl::ArraySlice<int64_t> input_shape(input_shape_in.vec<int64_t>().data(),
                                         input_shape_in.NumElements());

    gtl::InlinedVector<int64_t, 8> std_order(input_shape.size());
    std::iota(std_order.begin(), std_order.end(), 0);

    // Check if the sparse tensor is already ordered correctly
    sparse::SparseTensor input_sp;
    OP_REQUIRES_OK(
        context, sparse::SparseTensor::Create(input_ind, input_val, input_shape,
                                              std_order, &input_sp));

    if (input_sp.IndicesValid().ok()) {
      context->set_output(0, input_sp.indices());
      context->set_output(1, input_sp.values());
    } else {
      // Deep-copy the input Tensors, then reorder in-place
      sparse::SparseTensor reordered_sp;
      OP_REQUIRES_OK(context,
                     sparse::SparseTensor::Create(tensor::DeepCopy(input_ind),
                                                  tensor::DeepCopy(input_val),
                                                  input_shape, &reordered_sp));
      reordered_sp.Reorder<T>(std_order);
      context->set_output(0, reordered_sp.indices());
      context->set_output(1, reordered_sp.values());
    }
  }
};

}  // namespace functor

template <typename Device, typename T>
class SparseReorderOp : public OpKernel {
 public:
  explicit SparseReorderOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_reorder_opDTcc mht_0(mht_0_v, 250, "", "./tensorflow/core/kernels/sparse_reorder_op.cc", "SparseReorderOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_reorder_opDTcc mht_1(mht_1_v, 255, "", "./tensorflow/core/kernels/sparse_reorder_op.cc", "Compute");

    const Tensor& input_ind = context->input(0);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(input_ind.shape()),
                errors::InvalidArgument(
                    "Input indices should be a matrix but received shape ",
                    input_ind.shape().DebugString()));

    const Tensor& input_val = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_val.shape()),
                errors::InvalidArgument(
                    "Input values should be a vector but received shape ",
                    input_val.shape().DebugString()));

    const Tensor& input_shape_in = context->input(2);
    OP_REQUIRES(context, TensorShapeUtils::IsVector(input_shape_in.shape()),
                errors::InvalidArgument(
                    "Input shape should be a vector but received shape ",
                    input_shape_in.shape().DebugString()));

    functor::SparseReorderFunctor<Device, T>()(context, input_ind, input_val,
                                               input_shape_in);
  }
};

#define REGISTER_KERNELS(type)                                            \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("SparseReorder").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      SparseReorderOp<CPUDevice, type>)

TF_CALL_ALL_TYPES(REGISTER_KERNELS);
#undef REGISTER_KERNELS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define REGISTER_GPU_KERNELS(type)                                        \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("SparseReorder").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      SparseReorderOp<GPUDevice, type>)

TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
TF_CALL_INTEGRAL_TYPES(REGISTER_GPU_KERNELS);
REGISTER_GPU_KERNELS(bool);
#undef REGISTER_GPU_KERNELS

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
