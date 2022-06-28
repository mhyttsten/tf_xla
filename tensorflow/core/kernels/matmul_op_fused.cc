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
class MHTracer_DTPStensorflowPScorePSkernelsPSmatmul_op_fusedDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSmatmul_op_fusedDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmatmul_op_fusedDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

// Implements matmul operations with other kernels baked into the
// processing, to optimize latency and memory usage:
//  - MatMul + BiasAdd + <Activation>
//  - MatMul + FusedBatchNorm + <Activation>
//
// Activation: Relu, Relu6, Elu, etc...
//
// Currently supported only on CPU device.

#ifndef TENSORFLOW_CORE_KERNELS_MATMUL_OP_FUSED_H_
#define TENSORFLOW_CORE_KERNELS_MATMUL_OP_FUSED_H_

#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

#include <string>
#include <vector>

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/fused_eigen_output_kernels.h"
#include "tensorflow/core/util/tensor_format.h"

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#include "tensorflow/core/kernels/eigen_contraction_kernel.h"
#endif

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T>
struct LaunchFusedMatMulOp {
  void operator()(
      OpKernelContext* context, const Tensor& a, const Tensor& b,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair,
      FusedComputationType fusion, const FusedComputationArgs& fusion_args,
      Tensor* output);
};

template <typename T>
struct LaunchFusedMatMulOp<CPUDevice, T> {
  void operator()(
      OpKernelContext* context, const Tensor& a, const Tensor& b,
      const Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1>& dim_pair,
      FusedComputationType fusion, const FusedComputationArgs& fusion_args,
      Tensor* output) {
    auto lhs = a.matrix<T>();
    auto rhs = b.matrix<T>();
    auto out = output->matrix<T>();

    auto& d = context->eigen_device<CPUDevice>();

    // Executes Eigen contraction with output kernel wrapped into type erased
    // wrapper to reduce the number of unique template instantiations.
    auto executeWithOutputKernel = [&](auto output_kernel) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmatmul_op_fusedDTcc mht_0(mht_0_v, 244, "", "./tensorflow/core/kernels/matmul_op_fused.cc", "lambda");

      OutputKernelWrapper output_kernel_wrapper(
          [&output_kernel](
              const ContractionOutputMapper<T, Eigen::Index>& output_mapper,
              const Eigen::TensorContractionParams& params, Eigen::Index i,
              Eigen::Index j, Eigen::Index num_rows, Eigen::Index num_cols) {
            output_kernel(output_mapper, params, i, j, num_rows, num_cols);
          });

      out.device(d) = lhs.contract(rhs, dim_pair, output_kernel_wrapper);
    };

    BiasAddArgs<T> bias_add_args;
    if (BiasAddArgs<T>::IsSupported(fusion)) {
      if (fusion == FusedComputationType::kBiasAddWithLeakyRelu) {
        OP_REQUIRES_OK(context, InitBiasAddArgs(context, &bias_add_args,
                                                &fusion_args.leakyrelu_alpha));
      } else {
        OP_REQUIRES_OK(context, InitBiasAddArgs(context, &bias_add_args));
      }
    }

    switch (fusion) {
      case FusedComputationType::kBiasAdd:
        executeWithOutputKernel(WithBiasAdd<T>(bias_add_args));
        break;
      case FusedComputationType::kBiasAddWithRelu:
        executeWithOutputKernel(WithBiasAddAndRelu<T>(bias_add_args));
        break;
      case FusedComputationType::kBiasAddWithRelu6:
        executeWithOutputKernel(WithBiasAddAndRelu6<T>(bias_add_args));
        break;
      case FusedComputationType::kBiasAddWithElu:
        executeWithOutputKernel(WithBiasAddAndElu<T>(bias_add_args));
        break;
      case FusedComputationType::kBiasAddWithLeakyRelu:
        executeWithOutputKernel(WithBiasAddAndLeakyRelu<T>(bias_add_args));
        break;
      case FusedComputationType::kUndefined:
        OP_REQUIRES_OK(context, errors::Internal("Fusion type is undefined"));
        break;
      default:
        OP_REQUIRES_OK(context,
                       errors::Internal("Fusion type is not supported"));
    }
  }

 private:
  // Wrap output_kernel into type erased struct to reduce the number of unique
  // template instantiations for Eigen Tensor contraction expressions.
  //
  // We do not pass std::function directly as an output kernel because it blows
  // up the binary size in debug mode with super long symbol names.
  struct OutputKernelWrapper {
    using OutputKernelFn =
        std::function<void(const ContractionOutputMapper<T, Eigen::Index>&,
                           const Eigen::TensorContractionParams&, Eigen::Index,
                           Eigen::Index, Eigen::Index, Eigen::Index)>;

    explicit OutputKernelWrapper(OutputKernelFn fn)
        : output_kernel_fn(std::move(fn)) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmatmul_op_fusedDTcc mht_1(mht_1_v, 307, "", "./tensorflow/core/kernels/matmul_op_fused.cc", "OutputKernelWrapper");
}

    void operator()(
        const ContractionOutputMapper<T, Eigen::Index>& output_mapper,
        const Eigen::TensorContractionParams& params, Eigen::Index i,
        Eigen::Index j, Eigen::Index num_rows, Eigen::Index num_cols) const {
      output_kernel_fn(output_mapper, params, i, j, num_rows, num_cols);
    }

    OutputKernelFn output_kernel_fn;
  };
};

template <typename Device, typename T>
class FusedMatMulOp : public OpKernel {
 public:
  explicit FusedMatMulOp(OpKernelConstruction* context) : OpKernel(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmatmul_op_fusedDTcc mht_2(mht_2_v, 326, "", "./tensorflow/core/kernels/matmul_op_fused.cc", "FusedMatMulOp");

    OP_REQUIRES_OK(context, context->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(context, context->GetAttr("transpose_b", &transpose_b_));

    std::vector<FusedComputationPattern> patterns;

    using FCT = FusedComputationType;
    if (std::is_same<Device, CPUDevice>::value) {
      patterns = {
          {FCT::kBiasAdd, {"BiasAdd"}},
          {FCT::kBiasAddWithRelu, {"BiasAdd", "Relu"}},
          {FCT::kBiasAddWithRelu6, {"BiasAdd", "Relu6"}},
          {FCT::kBiasAddWithElu, {"BiasAdd", "Elu"}},
          {FCT::kBiasAddWithLeakyRelu, {"BiasAdd", "LeakyRelu"}},
      };
    }

    OP_REQUIRES_OK(context, InitializeFusedComputation(
                                context, "MatMul", patterns,
                                &fused_computation_, &fused_computation_args_));
  }

  void Compute(OpKernelContext* ctx) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmatmul_op_fusedDTcc mht_3(mht_3_v, 351, "", "./tensorflow/core/kernels/matmul_op_fused.cc", "Compute");

    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);

    // Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(ctx, a.dims() == b.dims(),
                errors::InvalidArgument("In[0] and In[1] has different ndims: ",
                                        a.shape().DebugString(), " vs. ",
                                        b.shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsMatrix(a.shape()),
        errors::InvalidArgument("In[0] is not a matrix. Instead it has shape ",
                                a.shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsMatrix(b.shape()),
        errors::InvalidArgument("In[1] is not a matrix. Instead it has shape ",
                                b.shape().DebugString()));
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
    dim_pair[0].first = transpose_a_ ? 0 : 1;
    dim_pair[0].second = transpose_b_ ? 1 : 0;

    OP_REQUIRES(
        ctx, a.dim_size(dim_pair[0].first) == b.dim_size(dim_pair[0].second),
        errors::InvalidArgument(
            "Matrix size-incompatible: In[0]: ", a.shape().DebugString(),
            ", In[1]: ", b.shape().DebugString()));
    int a_dim_remaining = 1 - dim_pair[0].first;
    int b_dim_remaining = 1 - dim_pair[0].second;
    TensorShape out_shape(
        {a.dim_size(a_dim_remaining), b.dim_size(b_dim_remaining)});
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));

    if (out->NumElements() == 0) {
      // If a has shape [0, x] or b has shape [x, 0], the output shape
      // is a 0-element matrix, so there is nothing to do.
      return;
    }

    if (a.NumElements() == 0 && b.NumElements() == 0) {
      // If a has shape [x, 0] and b has shape [0, y], the
      // output shape is [x, y] where x and y are non-zero, so we fill
      // the output with zeros.
      functor::SetZeroFunctor<Device, T> f;
      f(ctx->eigen_device<Device>(), out->flat<T>());
      return;
    }

    auto launch = LaunchFusedMatMulOp<Device, T>();
    launch(ctx, a, b, dim_pair, fused_computation_, fused_computation_args_,
           out);
  }

 private:
  bool transpose_a_;
  bool transpose_b_;

  FusedComputationType fused_computation_ = FusedComputationType::kUndefined;
  FusedComputationArgs fused_computation_args_;

  TF_DISALLOW_COPY_AND_ASSIGN(FusedMatMulOp);
};

// Registration of the CPU implementations.
#define REGISTER_FUSED_CPU_MATMUL(T)                                  \
  REGISTER_KERNEL_BUILDER(                                            \
      Name("_FusedMatMul").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      FusedMatMulOp<CPUDevice, T>);

TF_CALL_float(REGISTER_FUSED_CPU_MATMUL);

#undef REGISTER_FUSED_CPU_MATMUL

}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_KERNELS_MATMUL_OP_FUSED_H_
