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
class MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_transpose_opDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_transpose_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_transpose_opDTcc() {
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

// See docs in ../ops/array_ops.cc.

#if defined(INTEL_MKL)

#define EIGEN_USE_THREADS

#include "dnnl.hpp"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/kernels/transpose_op.h"
#include "tensorflow/core/util/mkl_util.h"

using dnnl::stream;

namespace tensorflow {

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

namespace {
// oneDNN based Transpose implementation
template <typename T>
Status MKLTransposeND(OpKernelContext* ctx, const Tensor& in, Tensor* out,
                      const gtl::ArraySlice<int32>& perm);

static inline memory::dims ReorderStrides(const memory::dims& strides,
                                          const gtl::ArraySlice<int32>& perm) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_transpose_opDTcc mht_0(mht_0_v, 223, "", "./tensorflow/core/kernels/mkl/mkl_transpose_op.cc", "ReorderStrides");

  memory::dims reordered_strides;
  reordered_strides.resize(strides.size());
  for (size_t i = 0; i < strides.size(); ++i) {
    reordered_strides[perm[i]] = strides[i];
  }
  return reordered_strides;
}

// Transpose of N-dimensional tensor using oneDNN
template <typename T>
Status MKLTransposeND(OpKernelContext* context, const Tensor& in_tensor,
                      Tensor* out_tensor, const gtl::ArraySlice<int32>& perm) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_transpose_opDTcc mht_1(mht_1_v, 238, "", "./tensorflow/core/kernels/mkl/mkl_transpose_op.cc", "MKLTransposeND");

  try {
    engine cpu_engine = engine(engine::kind::cpu, 0);
    MklDnnData<T> in(&cpu_engine);
    MklDnnData<T> out(&cpu_engine);

    memory::dims in_dims = TFShapeToMklDnnDims(in_tensor.shape());
    memory::dims out_dims = TFShapeToMklDnnDims(out_tensor->shape());
    memory::dims in_strides = CalculateTFStrides(in_dims);
    // Reorder output strides based on permutation requested.
    memory::dims out_strides =
        ReorderStrides(CalculateTFStrides(out_dims), perm);

    std::shared_ptr<stream> transpose_stream;
    in.SetUsrMem(in_dims, in_strides, &in_tensor);
    // Output dimensions are same as input dimensions. We adjust the layout
    // using strides.
    out.SetUsrMem(in_dims, out_strides, out_tensor);

    std::vector<primitive> net;
    auto* prim = FindOrCreateReorder<T>(in.GetUsrMem(), out.GetUsrMem());
    MklDnnThreadPool eigen_tp(context);
    transpose_stream.reset(CreateStream(&eigen_tp, prim->GetEngine()));
    in.SetUsrMemDataHandle(&in_tensor, transpose_stream);
    out.SetUsrMemDataHandle(out_tensor, transpose_stream);
    net.push_back(*(prim->GetPrimitive()));
    std::vector<MemoryArgsMap> net_args;
    net_args.push_back(
        {{DNNL_ARG_FROM, *in.GetUsrMem()}, {DNNL_ARG_TO, *out.GetUsrMem()}});
    execute_primitives(net, transpose_stream, net_args);

    return Status::OK();
  } catch (dnnl::error& e) {
    string error_msg = "Status: " + std::to_string(e.status) +
                       ", message: " + std::string(e.message) + ", in file " +
                       std::string(__FILE__) + ":" + std::to_string(__LINE__);
    return errors::Aborted("Operation received an exception:", error_msg);
  }
}

}  // namespace

Status MklTransposeCpuOp::DoTranspose(OpKernelContext* ctx, const Tensor& in,
                                      gtl::ArraySlice<int32> perm,
                                      Tensor* out) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_transpose_opDTcc mht_2(mht_2_v, 285, "", "./tensorflow/core/kernels/mkl/mkl_transpose_op.cc", "MklTransposeCpuOp::DoTranspose");

  // oneDNN has limit on the maximum number of dimensions in a tensor.
  // Fallback to Eigen for not supported cases.
  if (in.dims() <= DNNL_MAX_NDIMS) {
    switch (in.dtype()) {
      case DT_FLOAT:
        return MKLTransposeND<float>(ctx, in, out, perm);
        break;
      case DT_BFLOAT16:
        return MKLTransposeND<bfloat16>(ctx, in, out, perm);
        break;
      // TODO(intel-tf): support other types such as INT8.
      default:
        break;
    }
  }

  // Fallback to eigen if transpose parameters not supported by oneDNN
  typedef Eigen::ThreadPoolDevice CPUDevice;
  return ::tensorflow::DoTranspose(ctx->eigen_device<CPUDevice>(), in, perm,
                                   out);
}

Status MklConjugateTransposeCpuOp::DoTranspose(OpKernelContext* ctx,
                                               const Tensor& in,
                                               gtl::ArraySlice<int32> perm,
                                               Tensor* out) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPSmklPSmkl_transpose_opDTcc mht_3(mht_3_v, 314, "", "./tensorflow/core/kernels/mkl/mkl_transpose_op.cc", "MklConjugateTransposeCpuOp::DoTranspose");

  // oneDNN has limit on the maximum number of dimensions in a tensor.
  // Fallback to Eigen for not supported cases.
  if (in.dims() <= DNNL_MAX_NDIMS) {
    switch (in.dtype()) {
      case DT_FLOAT:
        return MKLTransposeND<float>(ctx, in, out, perm);
        break;
      case DT_BFLOAT16:
        return MKLTransposeND<bfloat16>(ctx, in, out, perm);
        break;
      // TODO(intel-tf): support other types such as INT8.
      default:
        break;
    }
  }

  // Fallback to eigen if transpose parameters not supported by oneDNN
  typedef Eigen::ThreadPoolDevice CPUDevice;
  return ::tensorflow::DoConjugateTranspose(ctx->eigen_device<CPUDevice>(), in,
                                            perm, out);
}

#define REGISTER(T)                                                           \
  REGISTER_KERNEL_BUILDER(Name("_MklTranspose")                               \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .HostMemory("perm")                             \
                              .Label(mkl_op_registry::kMklNameChangeOpLabel), \
                          MklTransposeCpuOp);                                 \
  REGISTER_KERNEL_BUILDER(Name("_MklConjugateTranspose")                      \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<T>("T")                         \
                              .HostMemory("perm")                             \
                              .Label(mkl_op_registry::kMklNameChangeOpLabel), \
                          MklConjugateTransposeCpuOp);

TF_CALL_ALL_TYPES(REGISTER)
#undef REGISTER

}  // namespace tensorflow

#endif  // INTEL_MKL
