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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdata_format_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdata_format_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdata_format_opsDTcc() {
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

/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <string>
#include <vector>

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/slicing.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {
namespace {

class DataFormatDimMapOp : public XlaOpKernel {
 public:
  explicit DataFormatDimMapOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdata_format_opsDTcc mht_0(mht_0_v, 202, "", "./tensorflow/compiler/tf2xla/kernels/data_format_ops.cc", "DataFormatDimMapOp");

    string src_format;
    OP_REQUIRES_OK(context, context->GetAttr("src_format", &src_format));
    string dst_format;
    OP_REQUIRES_OK(context, context->GetAttr("dst_format", &dst_format));
    OP_REQUIRES(context, src_format.size() == 4 or src_format.size() == 5,
                errors::InvalidArgument(
                    absl::StrCat("Source format must of length 4 or 5, "
                                 "received src_format = ",
                                 src_format)));
    OP_REQUIRES(
        context, dst_format.size() == 4 or dst_format.size() == 5,
        errors::InvalidArgument(absl::StrCat(
            "Destination format must of length 4 or 5, received dst_format = ",
            dst_format)));
    for (int i = 0; i < src_format.size(); ++i) {
      dst_idx_.push_back(-1);
    }
    for (int i = 0; i < src_format.size(); ++i) {
      for (int j = 0; j < dst_format.size(); ++j) {
        if (dst_format[j] == src_format[i]) {
          dst_idx_[i] = j;
          break;
        }
      }
      OP_REQUIRES(context, dst_idx_[i] != -1,
                  errors::InvalidArgument(absl::StrCat(
                      src_format, " is not a permutation of ", dst_format)));
    }
  }

  void Compile(XlaOpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdata_format_opsDTcc mht_1(mht_1_v, 236, "", "./tensorflow/compiler/tf2xla/kernels/data_format_ops.cc", "Compile");

    auto builder = context->builder();
    xla::XlaOp dst_indices =
        xla::ConstantR1(builder, absl::Span<const int32>(dst_idx_));
    const int dims = dst_idx_.size();
    xla::XlaOp rank = xla::ConstantR0<int32>(builder, dims);
    xla::XlaOp src_indices =
        (xla::ConvertElementType(context->Input(0), xla::S32) + rank) % rank;
    xla::XlaOp output =
        xla::TorchIndexSelect(dst_indices, src_indices, /*dim=*/0);
    context->SetOutput(
        0, xla::ConvertElementType(output, context->input_xla_type(0)));
  }

 private:
  std::vector<int32> dst_idx_;

  TF_DISALLOW_COPY_AND_ASSIGN(DataFormatDimMapOp);
};

REGISTER_XLA_OP(
    Name("DataFormatDimMap").TypeConstraint("T", {DT_INT32, DT_INT64}),
    DataFormatDimMapOp);

class DataFormatVecPermuteOp : public XlaOpKernel {
 public:
  explicit DataFormatVecPermuteOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdata_format_opsDTcc mht_2(mht_2_v, 266, "", "./tensorflow/compiler/tf2xla/kernels/data_format_ops.cc", "DataFormatVecPermuteOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("src_format", &src_format_));
    OP_REQUIRES(
        ctx, src_format_.size() == 4 || src_format_.size() == 5,
        errors::InvalidArgument("Data format should have 4 or 5 characters"));
    TensorFormat data_format;
    OP_REQUIRES(ctx, FormatFromString(src_format_, &data_format),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dst_format", &dst_format_));
    OP_REQUIRES(
        ctx, dst_format_.size() == 4 || dst_format_.size() == 5,
        errors::InvalidArgument("Data format should have 4 or 5 characters"));
    OP_REQUIRES(ctx, FormatFromString(dst_format_, &data_format),
                errors::InvalidArgument("Invalid data format"));
  }
  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdata_format_opsDTcc mht_3(mht_3_v, 284, "", "./tensorflow/compiler/tf2xla/kernels/data_format_ops.cc", "Compile");

    auto builder = ctx->builder();
    const TensorShape input_tensor_shape = ctx->InputShape(0);
    int input_rank = input_tensor_shape.dims();
    OP_REQUIRES(ctx, input_rank == 1 || input_rank == 2,
                errors::InvalidArgument(
                    "Input must be a vector or matrix, but got shape ",
                    input_tensor_shape.DebugString()));
    const int dim0 = input_tensor_shape.dim_size(0);

    const int full_dim_count = src_format_.size();
    const int spatial_dim_count = full_dim_count - 2;

    if (input_rank == 1) {
      OP_REQUIRES(ctx,
                  input_tensor_shape.num_elements() == spatial_dim_count ||
                      input_tensor_shape.num_elements() == full_dim_count,
                  errors::InvalidArgument("1D input must be of size ",
                                          spatial_dim_count, " or ",
                                          full_dim_count, ", but got shape ",
                                          input_tensor_shape.DebugString()));
    } else if (input_rank == 2) {
      OP_REQUIRES(ctx,
                  input_tensor_shape.dim_size(0) == spatial_dim_count ||
                      input_tensor_shape.dim_size(0) == full_dim_count,
                  errors::InvalidArgument("First dimension of 2D input must be "
                                          "of size ",
                                          spatial_dim_count, " or ",
                                          full_dim_count, ", but got shape ",
                                          input_tensor_shape.DebugString()));
      OP_REQUIRES(
          ctx, input_tensor_shape.dim_size(1) == 2,
          errors::InvalidArgument(
              "Second dimension of 2D input must be of size 2, but got shape ",
              input_tensor_shape.DebugString()));
    }

    string src_format_str = src_format_;
    string dst_format_str = dst_format_;
    if (input_tensor_shape.dim_size(0) == spatial_dim_count) {
      // If the input is a vector of size spatial_dim_count, treat the elements
      // as spatial dimensions.
      auto keep_only_spatial_dimensions =
          [spatial_dim_count](string* format_str) -> void {
        auto new_end =
            std::remove_if(format_str->begin(), format_str->end(),
                           [spatial_dim_count](const char dim) {
                             return dim != 'H' && dim != 'W' &&
                                    (spatial_dim_count == 2 || dim != 'D');
                           });
        format_str->erase(new_end, format_str->end());
      };
      keep_only_spatial_dimensions(&src_format_str);
      keep_only_spatial_dimensions(&dst_format_str);
    }
    std::vector<int32> dst_indices(dim0);
    for (int i = 0; i < dim0; ++i) {
      for (int j = 0; j < dim0; ++j) {
        if (src_format_str[i] == dst_format_str[j]) {
          dst_indices[j] = i;
          break;
        }
      }
    }
    xla::XlaOp indices =
        xla::ConstantR1(builder, absl::Span<const int32>(dst_indices));
    xla::XlaOp output = xla::TorchIndexSelect(ctx->Input(0), indices, 0);
    ctx->SetOutput(0, output);
  }

 private:
  string src_format_;
  string dst_format_;

  TF_DISALLOW_COPY_AND_ASSIGN(DataFormatVecPermuteOp);
};

REGISTER_XLA_OP(
    Name("DataFormatVecPermute").TypeConstraint("T", {DT_INT32, DT_INT64}),
    DataFormatVecPermuteOp);
REGISTER_XLA_OP(Name("DataFormatVecPermute")
                    .Label("host")
                    .TypeConstraint("T", {DT_INT32, DT_INT64}),
                DataFormatVecPermuteOp);

}  // namespace
}  // namespace tensorflow
