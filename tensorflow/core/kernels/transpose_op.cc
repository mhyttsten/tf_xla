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
class MHTracer_DTPStensorflowPScorePSkernelsPStranspose_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPStranspose_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPStranspose_opDTcc() {
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

#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/transpose_op.h"

#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {

// inv = InvertPermutationOp(T<int32/int64> p) takes a permutation of
// integers 0, 1, ..., n - 1 and returns the inverted
// permutation of p. I.e., inv[p[i]] == i, for i in [0 .. n).
//
// REQUIRES: input is a vector of int32 or int64.
// REQUIRES: input is a permutation of 0, 1, ..., n-1.

template <typename T>
class InvertPermutationOp : public OpKernel {
 public:
  explicit InvertPermutationOp(OpKernelConstruction* context)
      : OpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStranspose_opDTcc mht_0(mht_0_v, 214, "", "./tensorflow/core/kernels/transpose_op.cc", "InvertPermutationOp");
}

  void Compute(OpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStranspose_opDTcc mht_1(mht_1_v, 219, "", "./tensorflow/core/kernels/transpose_op.cc", "Compute");

    const Tensor& input = context->input(0);
    OP_REQUIRES(
        context, TensorShapeUtils::IsVector(input.shape()),
        errors::InvalidArgument("invert_permutation expects a 1D vector."));
    auto Tin = input.vec<T>();
    OP_REQUIRES(context,
                FastBoundsCheck(Tin.size(), std::numeric_limits<int32>::max()),
                errors::InvalidArgument("permutation of nonnegative int32s "
                                        "must have <= int32 max elements"));
    const T N = static_cast<T>(Tin.size());  // Safe: bounds-checked above.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));
    auto Tout = output->vec<T>();
    std::fill_n(Tout.data(), N, -1);
    for (int i = 0; i < N; ++i) {
      const T d = internal::SubtleMustCopy(Tin(i));
      OP_REQUIRES(context, FastBoundsCheck(d, N),
                  errors::InvalidArgument(d, " is not between 0 and ", N));
      OP_REQUIRES(context, Tout(d) == -1,
                  errors::InvalidArgument(d, " is duplicated in the input."));
      Tout(d) = i;
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("InvertPermutation").Device(DEVICE_CPU).TypeConstraint<int32>("T"),
    InvertPermutationOp<int32>);
REGISTER_KERNEL_BUILDER(
    Name("InvertPermutation").Device(DEVICE_CPU).TypeConstraint<int64_t>("T"),
    InvertPermutationOp<int64_t>);

REGISTER_KERNEL_BUILDER(Name("InvertPermutation")
                            .Device(DEVICE_DEFAULT)
                            .TypeConstraint<int32>("T")
                            .HostMemory("x")
                            .HostMemory("y"),
                        InvertPermutationOp<int32>);
REGISTER_KERNEL_BUILDER(Name("InvertPermutation")
                            .Device(DEVICE_DEFAULT)
                            .TypeConstraint<int64_t>("T")
                            .HostMemory("x")
                            .HostMemory("y"),
                        InvertPermutationOp<int64_t>);

namespace {
template <typename Tperm>
Status PermutationHelper(const Tensor& perm, const int dims,
                         std::vector<int32>* permutation) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStranspose_opDTcc mht_2(mht_2_v, 272, "", "./tensorflow/core/kernels/transpose_op.cc", "PermutationHelper");

  auto Vperm = perm.vec<Tperm>();
  if (dims != Vperm.size()) {
    return errors::InvalidArgument("transpose expects a vector of size ", dims,
                                   ". But input(1) is a vector of size ",
                                   Vperm.size());
  }
  // using volatile instead of SubtleMustCopy here so that the
  // asynchrony boundary is permutation.
  const volatile Tperm* perm_begin =
      reinterpret_cast<const volatile Tperm*>(Vperm.data());
  *permutation = std::vector<int32>(perm_begin, perm_begin + dims);

  return Status::OK();
}
}  // namespace

// output = TransposeOp(T<any> input, T<int32> perm) takes a tensor
// of type T and rank N, and a permutation of 0, 1, ..., N-1. It
// shuffles the dimensions of the input tensor according to permutation.
//
// Specifically, the returned tensor output meets the following condition:
// 1) output.dims() == input.dims();
// 2) output.dim_size(i) == input.dim_size(perm[i]);
// 3) output.tensor<T, N>(i_0, i_1, ..., i_N-1) ==
//      input.tensor<T, N>(j_0, j_1, ..., j_N-1),
//    where i_s == j_{perm[s]}
//
// REQUIRES: perm is a vector of int32.
// REQUIRES: input.dims() == perm.size().
// REQUIRES: perm is a permutation.

void TransposeOp::Compute(OpKernelContext* ctx) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStranspose_opDTcc mht_3(mht_3_v, 307, "", "./tensorflow/core/kernels/transpose_op.cc", "TransposeOp::Compute");

  const Tensor& input = ctx->input(0);
  const Tensor& perm = ctx->input(1);
  // Preliminary validation of sizes.
  OP_REQUIRES(ctx, TensorShapeUtils::IsVector(perm.shape()),
              errors::InvalidArgument("perm must be rank 1, got shape ",
                                      perm.shape().DebugString()));

  // Although Tperm may be an int64 type, an int32 is sufficient to hold
  // dimension range values, so the narrowing here should be safe.
  std::vector<int32> permutation;
  const int dims = input.dims();
  if (perm.dtype() == DT_INT32) {
    OP_REQUIRES_OK(ctx, PermutationHelper<int32>(perm, dims, &permutation));
  } else {
    OP_REQUIRES_OK(ctx, PermutationHelper<int64_t>(perm, dims, &permutation));
  }
  TensorShape shape;

  // Check whether permutation is a permutation of integers of [0 .. dims).
  gtl::InlinedVector<bool, 8> bits(dims);
  bool is_identity = true;
  for (int i = 0; i < dims; ++i) {
    const int32_t d = permutation[i];
    OP_REQUIRES(
        ctx, 0 <= d && d < dims,
        errors::InvalidArgument(d, " is out of range [0 .. ", dims, ")"));
    bits[d] = true;
    const auto dim_size = input.dim_size(d);
    shape.AddDim(dim_size);
    if (d != i) {
      is_identity = false;
    }
  }
  for (int i = 0; i < dims; ++i) {
    OP_REQUIRES(ctx, bits[i],
                errors::InvalidArgument(i, " is missing from {",
                                        absl::StrJoin(permutation, ","), "}."));
  }

  // 0-D, 1-D, and identity transposes do nothing.
  if (!IsConjugate() && (dims <= 1 || is_identity)) {
    ctx->set_output(0, input);
    return;
  } else if (!IsConjugate() && internal::NonSingletonDimensionsAlign(
                                   input.shape(), permutation)) {
    Tensor output;
    OP_REQUIRES(ctx, output.CopyFrom(input, shape),
                errors::Unknown("Error reshaping Tensor."));
    ctx->set_output(0, output);
    return;
  }

  Tensor* output = nullptr;
  OP_REQUIRES_OK(ctx, ctx->allocate_output(0, shape, &output));
  if (shape.num_elements() > 0) {
    OP_REQUIRES_OK(ctx, DoTranspose(ctx, input, permutation, output));
  }
}

Status TransposeCpuOp::DoTranspose(OpKernelContext* ctx, const Tensor& in,
                                   gtl::ArraySlice<int32> perm, Tensor* out) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStranspose_opDTcc mht_4(mht_4_v, 371, "", "./tensorflow/core/kernels/transpose_op.cc", "TransposeCpuOp::DoTranspose");

  typedef Eigen::ThreadPoolDevice CPUDevice;
  return ::tensorflow::DoTranspose(ctx->eigen_device<CPUDevice>(), in, perm,
                                   out);
}

Status ConjugateTransposeCpuOp::DoTranspose(OpKernelContext* ctx,
                                            const Tensor& in,
                                            gtl::ArraySlice<int32> perm,
                                            Tensor* out) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStranspose_opDTcc mht_5(mht_5_v, 383, "", "./tensorflow/core/kernels/transpose_op.cc", "ConjugateTransposeCpuOp::DoTranspose");

  typedef Eigen::ThreadPoolDevice CPUDevice;
  return ::tensorflow::DoConjugateTranspose(ctx->eigen_device<CPUDevice>(), in,
                                            perm, out);
}

#define REGISTER(T)                                   \
  REGISTER_KERNEL_BUILDER(Name("Transpose")           \
                              .Device(DEVICE_CPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("perm"),    \
                          TransposeCpuOp);            \
  REGISTER_KERNEL_BUILDER(Name("ConjugateTranspose")  \
                              .Device(DEVICE_CPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("perm"),    \
                          ConjugateTransposeCpuOp);

TF_CALL_ALL_TYPES(REGISTER)
#undef REGISTER

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
Status TransposeGpuOp::DoTranspose(OpKernelContext* ctx, const Tensor& in,
                                   gtl::ArraySlice<int32> perm, Tensor* out) {
  typedef Eigen::GpuDevice GPUDevice;
  return ::tensorflow::DoTranspose(ctx->eigen_device<GPUDevice>(), in, perm,
                                   out);
}
Status ConjugateTransposeGpuOp::DoTranspose(OpKernelContext* ctx,
                                            const Tensor& in,
                                            gtl::ArraySlice<int32> perm,
                                            Tensor* out) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePSkernelsPStranspose_opDTcc mht_6(mht_6_v, 417, "", "./tensorflow/core/kernels/transpose_op.cc", "ConjugateTransposeGpuOp::DoTranspose");

  typedef Eigen::GpuDevice GPUDevice;
  return ::tensorflow::DoConjugateTranspose(ctx->eigen_device<GPUDevice>(), in,
                                            perm, out);
}

#define REGISTER(T)                                   \
  REGISTER_KERNEL_BUILDER(Name("Transpose")           \
                              .Device(DEVICE_GPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("perm"),    \
                          TransposeGpuOp);            \
  REGISTER_KERNEL_BUILDER(Name("ConjugateTranspose")  \
                              .Device(DEVICE_GPU)     \
                              .TypeConstraint<T>("T") \
                              .HostMemory("perm"),    \
                          ConjugateTransposeGpuOp);
TF_CALL_POD_TYPES(REGISTER);
#undef REGISTER
#endif

}  // namespace tensorflow
