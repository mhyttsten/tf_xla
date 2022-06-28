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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSfft_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSfft_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSfft_opsDTcc() {
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

// XLA-specific Ops for FFT.

#include "tensorflow/compiler/tf2xla/mlir_xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/ops_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_slice.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

namespace {

using xla::FftType;

class GenericFftOp : public XlaOpKernel {
 public:
  explicit GenericFftOp(OpKernelConstruction* ctx, FftType fft_type,
                        int fft_rank)
      : XlaOpKernel(ctx), fft_type_(fft_type), fft_rank_(fft_rank) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSfft_opsDTcc mht_0(mht_0_v, 214, "", "./tensorflow/compiler/tf2xla/kernels/fft_ops.cc", "GenericFftOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSfft_opsDTcc mht_1(mht_1_v, 219, "", "./tensorflow/compiler/tf2xla/kernels/fft_ops.cc", "Compile");

    const TensorShape input_shape = ctx->InputShape(0);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVectorOrHigher(input_shape),
        errors::InvalidArgument("input must be at least 1 dimensional"));

    std::vector<int64_t> fft_length;
    xla::XlaOp input = ctx->Input(0);
    if (fft_type_ == FftType::RFFT || fft_type_ == FftType::IRFFT) {
      OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(1, &fft_length));
      OP_REQUIRES(ctx, fft_length.size() == fft_rank_,
                  errors::InvalidArgument("fft_length must be length ",
                                          fft_rank_, " vector"));

      // Zero pad or truncate the axes we're doing FFT on.
      absl::InlinedVector<int64_t, 4> slice_sizes = input_shape.dim_sizes();
      std::vector<std::pair<int64_t, int64_t>> padding_sizes(
          slice_sizes.size());
      std::vector<int64_t> expected_sizes = fft_length;
      // IRFFT wants the innermost axis to be n / 2 + 1.
      if (fft_type_ == FftType::IRFFT) {
        expected_sizes[fft_rank_ - 1] = fft_length[fft_rank_ - 1] / 2 + 1;
      }
      for (int i = 0; i < fft_rank_; i++) {
        int index = input_shape.dims() - fft_rank_ + i;
        OP_REQUIRES(
            ctx,
            input_shape.dim_size(index) == 0 ||
                input_shape.dim_size(index) >= expected_sizes[i],
            errors::InvalidArgument(
                "Input dimension ", index, " must have length of at least ",
                expected_sizes[i], " but got: ", input_shape.dim_size(index)));
        if (input_shape.dim_size(index) > expected_sizes[i]) {
          slice_sizes[index] = expected_sizes[i];
        } else {
          padding_sizes[index].second =
              expected_sizes[i] - input_shape.dim_size(index);
        }
      }

      std::vector<int64_t> start_indices(input_shape.dims(), 0);
      std::vector<int64_t> strides(input_shape.dims(), 1);
      input = xla::Pad(xla::Slice(input, start_indices, slice_sizes, strides),
                       XlaHelpers::Zero(ctx->builder(), ctx->input_type(0)),
                       xla::MakeEdgePaddingConfig(padding_sizes));
    } else {
      // Innermost axis provides the FFT length.
      for (int i = 0; i < fft_rank_; i++) {
        fft_length.push_back(
            input_shape.dim_size(input_shape.dims() - fft_rank_ + i));
      }
    }

    xla::XlaOp fft = xla::Fft(input, fft_type_, fft_length);
    ctx->SetOutput(0, fft);
  }

 protected:
  const FftType fft_type_;
  const int fft_rank_;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(GenericFftOp);
};

template <int FFTRank>
class FFTOp : public GenericFftOp {
 public:
  explicit FFTOp(OpKernelConstruction* ctx)
      : GenericFftOp(ctx, /*fft_type=*/FftType::FFT, /*fft_rank=*/FFTRank) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSfft_opsDTcc mht_2(mht_2_v, 291, "", "./tensorflow/compiler/tf2xla/kernels/fft_ops.cc", "FFTOp");
}
};
REGISTER_XLA_OP(Name("FFT").TypeConstraint("Tcomplex",
                                           {DT_COMPLEX64, DT_COMPLEX128}),
                FFTOp<1>);
REGISTER_XLA_OP(Name("FFT2D").TypeConstraint("Tcomplex",
                                             {DT_COMPLEX64, DT_COMPLEX128}),
                FFTOp<2>);
REGISTER_XLA_OP(Name("FFT3D").TypeConstraint("Tcomplex",
                                             {DT_COMPLEX64, DT_COMPLEX128}),
                FFTOp<3>);

template <int FFTRank>
class IFFTOp : public GenericFftOp {
 public:
  explicit IFFTOp(OpKernelConstruction* ctx)
      : GenericFftOp(ctx, /*fft_type=*/FftType::IFFT, /*fft_rank=*/FFTRank) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSfft_opsDTcc mht_3(mht_3_v, 310, "", "./tensorflow/compiler/tf2xla/kernels/fft_ops.cc", "IFFTOp");
}
};
REGISTER_XLA_OP(Name("IFFT").TypeConstraint("Tcomplex", DT_COMPLEX64),
                MlirXlaOpKernel);
REGISTER_XLA_OP(Name("IFFT2D").TypeConstraint("Tcomplex", DT_COMPLEX64),
                IFFTOp<2>);
REGISTER_XLA_OP(Name("IFFT3D").TypeConstraint("Tcomplex", DT_COMPLEX64),
                IFFTOp<3>);

template <int FFTRank>
class RFFTOp : public GenericFftOp {
 public:
  explicit RFFTOp(OpKernelConstruction* ctx)
      : GenericFftOp(ctx, /*fft_type=*/FftType::RFFT, /*fft_rank=*/FFTRank) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSfft_opsDTcc mht_4(mht_4_v, 326, "", "./tensorflow/compiler/tf2xla/kernels/fft_ops.cc", "RFFTOp");
}
};
REGISTER_XLA_OP(Name("RFFT")
                    .TypeConstraint("Treal", DT_FLOAT)
                    .TypeConstraint("Tcomplex", DT_COMPLEX64)
                    .CompileTimeConstantInput("fft_length"),
                RFFTOp<1>);
REGISTER_XLA_OP(Name("RFFT2D")
                    .TypeConstraint("Treal", DT_FLOAT)
                    .TypeConstraint("Tcomplex", DT_COMPLEX64)
                    .CompileTimeConstantInput("fft_length"),
                RFFTOp<2>);
REGISTER_XLA_OP(Name("RFFT3D")
                    .TypeConstraint("Treal", DT_FLOAT)
                    .TypeConstraint("Tcomplex", DT_COMPLEX64)
                    .CompileTimeConstantInput("fft_length"),
                RFFTOp<3>);

template <int FFTRank>
class IRFFTOp : public GenericFftOp {
 public:
  explicit IRFFTOp(OpKernelConstruction* ctx)
      : GenericFftOp(ctx, /*fft_type=*/FftType::IRFFT, /*fft_rank=*/FFTRank) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSfft_opsDTcc mht_5(mht_5_v, 351, "", "./tensorflow/compiler/tf2xla/kernels/fft_ops.cc", "IRFFTOp");
}
};
REGISTER_XLA_OP(Name("IRFFT")
                    .TypeConstraint("Treal", DT_FLOAT)
                    .TypeConstraint("Tcomplex", DT_COMPLEX64)
                    .CompileTimeConstantInput("fft_length"),
                IRFFTOp<1>);
REGISTER_XLA_OP(Name("IRFFT2D")
                    .TypeConstraint("Treal", DT_FLOAT)
                    .TypeConstraint("Tcomplex", DT_COMPLEX64)
                    .CompileTimeConstantInput("fft_length"),
                IRFFTOp<2>);
REGISTER_XLA_OP(Name("IRFFT3D")
                    .TypeConstraint("Treal", DT_FLOAT)
                    .TypeConstraint("Tcomplex", DT_COMPLEX64)
                    .CompileTimeConstantInput("fft_length"),
                IRFFTOp<3>);

}  // namespace
}  // namespace tensorflow
