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
class MHTracer_DTPStensorflowPScorePSkernelsPSsparse_xent_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_xent_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSsparse_xent_opDTcc() {
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

// See docs in ../ops/nn_ops.cc.

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/sparse_xent_op.h"

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/util/determinism.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Index>
Status CheckInvalidLabelIndex(const Tensor& labels, int64_t max_index) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_xent_opDTcc mht_0(mht_0_v, 205, "", "./tensorflow/core/kernels/sparse_xent_op.cc", "CheckInvalidLabelIndex");

  if (labels.NumElements() == 0) return Status::OK();
  const auto label_values = labels.vec<Index>();
  int64_t bad_index;
  auto min_max_dim_value = std::minmax_element(
      label_values.data(), label_values.data() + label_values.size());
  if (*min_max_dim_value.first < 0 || *min_max_dim_value.second >= max_index) {
    bad_index = (*min_max_dim_value.first < 0) ? *min_max_dim_value.first
                                               : *min_max_dim_value.second;
    return errors::InvalidArgument(
        "Received a label value of ", bad_index,
        " which is outside the valid range of [0, ", max_index,
        ").  Label values: ", labels.SummarizeValue(labels.NumElements()));
  }
  return Status::OK();
}

template <typename Device, typename T, typename Index>
class SparseSoftmaxXentWithLogitsOp : public OpKernel {
 public:
  explicit SparseSoftmaxXentWithLogitsOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_xent_opDTcc mht_1(mht_1_v, 229, "", "./tensorflow/core/kernels/sparse_xent_op.cc", "SparseSoftmaxXentWithLogitsOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSsparse_xent_opDTcc mht_2(mht_2_v, 234, "", "./tensorflow/core/kernels/sparse_xent_op.cc", "Compute");

    const Tensor& logits = context->input(0);
    const Tensor& labels = context->input(1);
    OP_REQUIRES(context, TensorShapeUtils::IsMatrix(logits.shape()),
                errors::InvalidArgument("logits must be 2-D, but got shape ",
                                        logits.shape().DebugString()));
    OP_REQUIRES(context, TensorShapeUtils::IsVector(labels.shape()),
                errors::InvalidArgument("labels must be 1-D, but got shape ",
                                        labels.shape().DebugString()));
    OP_REQUIRES(context, logits.dim_size(0) == labels.dim_size(0),
                errors::InvalidArgument(
                    "logits and labels must have the same first dimension, "
                    "got logits shape ",
                    logits.shape().DebugString(), " and labels shape ",
                    labels.shape().DebugString()));
    OP_REQUIRES(context, logits.dim_size(1) > 0,
                errors::InvalidArgument(
                    "Must have at least one class, but got logits shape ",
                    logits.shape().DebugString()));

    if (std::is_same<Device, GPUDevice>::value) {
      OP_REQUIRES(
          context, !OpDeterminismRequired(),
          errors::Unimplemented(
              "The GPU implementation of SparseSoftmaxCrossEntropyWithLogits"
              " that would have been executed is not deterministic. Note that"
              " the Python API uses an alternative, deterministic,"
              " GPU-accelerated path when determinsim is enabled."));
    }

    Tensor scratch;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                                                   labels.shape(), &scratch));

    Tensor* loss_out = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {1}, 0, labels.shape(), &loss_out));
    Tensor* back_out = nullptr;
    OP_REQUIRES_OK(context, context->forward_input_or_allocate_output(
                                {0}, 1, logits.shape(), &back_out));

    if (logits.dim_size(0) > 0) {
      if (std::is_same<Device, CPUDevice>::value) {
        OP_REQUIRES_OK(
            context, CheckInvalidLabelIndex<Index>(labels, logits.dim_size(1)));
      }
      functor::SparseXentFunctor<Device, T, Index> functor;
      functor(context, logits.matrix<T>(), labels.vec<Index>(),
              scratch.vec<T>(), loss_out->vec<T>(), back_out->matrix<T>());
    }
  }
};

// Partial specialization for a CPUDevice, that uses the Eigen implementation
// from XentEigenImpl.
namespace functor {
template <typename T, typename Index>
struct SparseXentFunctor<CPUDevice, T, Index> {
  void operator()(OpKernelContext* ctx, typename TTypes<T>::ConstMatrix logits,
                  typename TTypes<Index>::ConstVec labels,
                  typename TTypes<T>::Vec scratch, typename TTypes<T>::Vec loss,
                  typename TTypes<T>::Matrix backprop) {
    SparseXentEigenImpl<CPUDevice, T, Index>::Compute(ctx, logits, labels,
                                                      scratch, loss, backprop);
  }
};
}  // namespace functor

#define REGISTER(Dev, T, Index)                   \
  REGISTER_KERNEL_BUILDER(                        \
      Name("SparseSoftmaxCrossEntropyWithLogits") \
          .Device(DEVICE_##Dev)                   \
          .TypeConstraint<T>("T")                 \
          .TypeConstraint<Index>("Tlabels"),      \
      SparseSoftmaxXentWithLogitsOp<Dev##Device, T, Index>);
REGISTER(CPU, float, int32)
REGISTER(CPU, float, int64_t)
REGISTER(CPU, double, int32)
REGISTER(CPU, double, int64_t)
REGISTER(CPU, Eigen::half, int32)
REGISTER(CPU, Eigen::half, int64_t)
REGISTER(CPU, bfloat16, int32)
REGISTER(CPU, bfloat16, int64_t)

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
REGISTER(GPU, float, int32)
REGISTER(GPU, float, int64_t)
REGISTER(GPU, Eigen::half, int32)
REGISTER(GPU, Eigen::half, int64_t)
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#undef REGISTER

}  // namespace tensorflow
