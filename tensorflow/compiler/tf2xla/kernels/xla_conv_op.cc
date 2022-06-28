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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSxla_conv_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSxla_conv_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSxla_conv_opDTcc() {
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

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace {

class XlaConvOp : public XlaOpKernel {
 public:
  explicit XlaConvOp(OpKernelConstruction* context) : XlaOpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSxla_conv_opDTcc mht_0(mht_0_v, 199, "", "./tensorflow/compiler/tf2xla/kernels/xla_conv_op.cc", "XlaConvOp");

    string dnums_attr;
    OP_REQUIRES_OK(context, context->GetAttr("dimension_numbers", &dnums_attr));
    OP_REQUIRES(
        context, dnums_.ParsePartialFromString(dnums_attr),
        errors::InvalidArgument("Error parsing convolution dimension numbers"));
    string precision_config_attr;
    OP_REQUIRES_OK(
        context, context->GetAttr("precision_config", &precision_config_attr));
    OP_REQUIRES(context,
                precision_config_.ParsePartialFromString(precision_config_attr),
                errors::InvalidArgument("Error parsing precision config."));
    preferred_element_type_ = absl::nullopt;
    batch_group_count_ = 1;
  }

  void Compile(XlaOpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSxla_conv_opDTcc mht_1(mht_1_v, 218, "", "./tensorflow/compiler/tf2xla/kernels/xla_conv_op.cc", "Compile");

    const TensorShape lhs_shape = context->InputShape(0);
    const TensorShape rhs_shape = context->InputShape(1);
    const TensorShape padding_shape = context->InputShape("padding");
    std::vector<int64_t> window_strides;
    std::vector<int64_t> lhs_dilation;
    std::vector<int64_t> rhs_dilation;
    int64_t feature_group_count;
    OP_REQUIRES_OK(context, context->ConstantInputAsIntVector("window_strides",
                                                              &window_strides));
    OP_REQUIRES_OK(context, context->ConstantInputAsIntVector("lhs_dilation",
                                                              &lhs_dilation));
    OP_REQUIRES_OK(context, context->ConstantInputAsIntVector("rhs_dilation",
                                                              &rhs_dilation));
    OP_REQUIRES_OK(context, context->ConstantInputAsIntScalar(
                                "feature_group_count", &feature_group_count));

    OP_REQUIRES(context,
                TensorShapeUtils::IsMatrix(padding_shape) &&
                    padding_shape.dim_size(1) == 2,
                errors::InvalidArgument(
                    "padding must be a matrix with minor dimension 2, got ",
                    padding_shape.DebugString()));
    xla::Literal padding_literal;
    OP_REQUIRES_OK(context, context->ConstantInputAsInt64Literal(
                                "padding", &padding_literal));
    std::vector<std::pair<int64_t, int64_t>> padding(padding_shape.dim_size(0));
    for (int i = 0; i < padding.size(); ++i) {
      padding[i] = {padding_literal.Get<int64_t>({i, 0}),
                    padding_literal.Get<int64_t>({i, 1})};
    }

    // We do only minimal checking, relying on XLA to check the shape
    // invariants.
    xla::XlaOp output = xla::ConvGeneralDilated(
        context->Input(0), context->Input(1), window_strides, padding,
        lhs_dilation, rhs_dilation, dnums_, feature_group_count,
        batch_group_count_, &precision_config_, preferred_element_type_);
    context->SetOutput(0, output);
  }

 protected:
  absl::optional<xla::PrimitiveType> preferred_element_type_;
  int64_t batch_group_count_;

 private:
  xla::ConvolutionDimensionNumbers dnums_;
  xla::PrecisionConfig precision_config_;

  TF_DISALLOW_COPY_AND_ASSIGN(XlaConvOp);
};

REGISTER_XLA_OP(Name("XlaConv")
                    .CompileTimeConstantInput("window_strides")
                    .CompileTimeConstantInput("lhs_dilation")
                    .CompileTimeConstantInput("rhs_dilation")
                    .CompileTimeConstantInput("feature_group_count")
                    .CompileTimeConstantInput("padding"),
                XlaConvOp);

class XlaConvV2Op : public XlaConvOp {
 public:
  explicit XlaConvV2Op(OpKernelConstruction* context) : XlaConvOp(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSxla_conv_opDTcc mht_2(mht_2_v, 283, "", "./tensorflow/compiler/tf2xla/kernels/xla_conv_op.cc", "XlaConvV2Op");

    DataType preferred_element_dtype;
    OP_REQUIRES_OK(context, context->GetAttr("preferred_element_type",
                                             &preferred_element_dtype));
    xla::PrimitiveType preferred_element_type;
    OP_REQUIRES_OK(context, DataTypeToPrimitiveType(preferred_element_dtype,
                                                    &preferred_element_type));
    preferred_element_type_ = preferred_element_type;

    OP_REQUIRES_OK(context,
                   context->GetAttr("batch_group_count", &batch_group_count_));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(XlaConvV2Op);
};

REGISTER_XLA_OP(Name("XlaConvV2")
                    .CompileTimeConstantInput("window_strides")
                    .CompileTimeConstantInput("lhs_dilation")
                    .CompileTimeConstantInput("rhs_dilation")
                    .CompileTimeConstantInput("feature_group_count")
                    .CompileTimeConstantInput("padding"),
                XlaConvV2Op);

}  // namespace
}  // namespace tensorflow
