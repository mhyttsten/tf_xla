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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreduction_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreduction_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreduction_opsDTcc() {
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

// XLA-specific reduction Ops.

#include "tensorflow/compiler/tf2xla/kernels/reduction_ops.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/core/framework/kernel_def_builder.h"

namespace tensorflow {
namespace {

class SumOp : public XlaReductionOp {
 public:
  explicit SumOp(OpKernelConstruction* ctx)
      : XlaReductionOp(ctx,
                       XlaHelpers::SumAccumulationType(ctx->input_type(0))) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreduction_opsDTcc mht_0(mht_0_v, 203, "", "./tensorflow/compiler/tf2xla/kernels/reduction_ops.cc", "SumOp");
}
  xla::XlaOp InitialValue(xla::XlaBuilder* builder) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreduction_opsDTcc mht_1(mht_1_v, 207, "", "./tensorflow/compiler/tf2xla/kernels/reduction_ops.cc", "InitialValue");

    return xla::Zero(builder, xla_reduction_type_);
  }
  void BuildReducer(xla::XlaBuilder* builder, const xla::XlaOp& scalar_lhs,
                    const xla::XlaOp& scalar_rhs) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreduction_opsDTcc mht_2(mht_2_v, 214, "", "./tensorflow/compiler/tf2xla/kernels/reduction_ops.cc", "BuildReducer");

    xla::Add(scalar_lhs, scalar_rhs);
  }
};

REGISTER_XLA_OP(Name("Sum").CompileTimeConstantInput("reduction_indices"),
                SumOp);

class ProdOp : public XlaReductionOp {
 public:
  explicit ProdOp(OpKernelConstruction* ctx)
      : XlaReductionOp(ctx,
                       XlaHelpers::SumAccumulationType(ctx->input_type(0))) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreduction_opsDTcc mht_3(mht_3_v, 229, "", "./tensorflow/compiler/tf2xla/kernels/reduction_ops.cc", "ProdOp");
}

  xla::XlaOp InitialValue(xla::XlaBuilder* builder) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreduction_opsDTcc mht_4(mht_4_v, 234, "", "./tensorflow/compiler/tf2xla/kernels/reduction_ops.cc", "InitialValue");

    return xla::One(builder, xla_reduction_type_);
  }

  void BuildReducer(xla::XlaBuilder* builder, const xla::XlaOp& scalar_lhs,
                    const xla::XlaOp& scalar_rhs) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreduction_opsDTcc mht_5(mht_5_v, 242, "", "./tensorflow/compiler/tf2xla/kernels/reduction_ops.cc", "BuildReducer");

    xla::Mul(scalar_lhs, scalar_rhs);
  }
};

REGISTER_XLA_OP(Name("Prod").CompileTimeConstantInput("reduction_indices"),
                ProdOp);

class MinOp : public XlaReductionOp {
 public:
  explicit MinOp(OpKernelConstruction* ctx)
      : XlaReductionOp(ctx, ctx->input_type(0)) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreduction_opsDTcc mht_6(mht_6_v, 256, "", "./tensorflow/compiler/tf2xla/kernels/reduction_ops.cc", "MinOp");
}

  xla::XlaOp InitialValue(xla::XlaBuilder* builder) override {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreduction_opsDTcc mht_7(mht_7_v, 261, "", "./tensorflow/compiler/tf2xla/kernels/reduction_ops.cc", "InitialValue");

    return xla::MaxValue(builder, xla_reduction_type_);
  }

  void BuildReducer(xla::XlaBuilder* builder, const xla::XlaOp& scalar_lhs,
                    const xla::XlaOp& scalar_rhs) override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreduction_opsDTcc mht_8(mht_8_v, 269, "", "./tensorflow/compiler/tf2xla/kernels/reduction_ops.cc", "BuildReducer");

    xla::Min(scalar_lhs, scalar_rhs);
  }
};

REGISTER_XLA_OP(Name("Min").CompileTimeConstantInput("reduction_indices"),
                MinOp);

class MaxOp : public XlaReductionOp {
 public:
  explicit MaxOp(OpKernelConstruction* ctx)
      : XlaReductionOp(ctx, ctx->input_type(0)) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreduction_opsDTcc mht_9(mht_9_v, 283, "", "./tensorflow/compiler/tf2xla/kernels/reduction_ops.cc", "MaxOp");

    OP_REQUIRES_OK(ctx, PrimitiveTypeCheck(xla_reduction_type_));
  }

  static Status PrimitiveTypeCheck(xla::PrimitiveType xla_reduction_type) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreduction_opsDTcc mht_10(mht_10_v, 290, "", "./tensorflow/compiler/tf2xla/kernels/reduction_ops.cc", "PrimitiveTypeCheck");

    if (xla_reduction_type == xla::C64 || xla_reduction_type == xla::C128 ||
        xla_reduction_type == xla::TUPLE ||
        xla_reduction_type == xla::OPAQUE_TYPE) {
      return errors::InvalidArgument(
          "Unsupported PrimitiveType in MaxOp: '",
          xla::PrimitiveType_Name(xla_reduction_type), "'");
    } else {
      return Status::OK();
    }
  }

  xla::XlaOp InitialValue(xla::XlaBuilder* builder) override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreduction_opsDTcc mht_11(mht_11_v, 305, "", "./tensorflow/compiler/tf2xla/kernels/reduction_ops.cc", "InitialValue");

    return xla::MinValue(builder, xla_reduction_type_);
  }

  void BuildReducer(xla::XlaBuilder* builder, const xla::XlaOp& scalar_lhs,
                    const xla::XlaOp& scalar_rhs) override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreduction_opsDTcc mht_12(mht_12_v, 313, "", "./tensorflow/compiler/tf2xla/kernels/reduction_ops.cc", "BuildReducer");

    xla::Max(scalar_lhs, scalar_rhs);
  }
};

REGISTER_XLA_OP(Name("Max").CompileTimeConstantInput("reduction_indices"),
                MaxOp);

class MeanOp : public XlaReductionOp {
 public:
  explicit MeanOp(OpKernelConstruction* ctx)
      : XlaReductionOp(ctx,
                       XlaHelpers::SumAccumulationType(ctx->input_type(0))) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreduction_opsDTcc mht_13(mht_13_v, 328, "", "./tensorflow/compiler/tf2xla/kernels/reduction_ops.cc", "MeanOp");
}

  xla::XlaOp InitialValue(xla::XlaBuilder* builder) override {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreduction_opsDTcc mht_14(mht_14_v, 333, "", "./tensorflow/compiler/tf2xla/kernels/reduction_ops.cc", "InitialValue");

    return xla::Zero(builder, xla_reduction_type_);
  }
  void BuildReducer(xla::XlaBuilder* builder, const xla::XlaOp& scalar_lhs,
                    const xla::XlaOp& scalar_rhs) override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreduction_opsDTcc mht_15(mht_15_v, 340, "", "./tensorflow/compiler/tf2xla/kernels/reduction_ops.cc", "BuildReducer");

    xla::Add(scalar_lhs, scalar_rhs);
  }

  xla::XlaOp BuildFinalizer(
      xla::XlaBuilder* /*builder*/, const xla::XlaOp& input,
      const xla::XlaOp& reduce_output,
      const std::vector<int64_t>& dimensions_to_reduce) override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreduction_opsDTcc mht_16(mht_16_v, 350, "", "./tensorflow/compiler/tf2xla/kernels/reduction_ops.cc", "BuildFinalizer");

    if (dimensions_to_reduce.empty()) {
      return reduce_output;
    }
    auto divisor = xla::GetDimensionSize(input, dimensions_to_reduce[0]);
    for (int i = 1; i < dimensions_to_reduce.size(); i++) {
      auto size = xla::GetDimensionSize(input, dimensions_to_reduce[i]);
      divisor = xla::Mul(divisor, size);
    }
    divisor = xla::ConvertElementType(divisor, xla_reduction_type_);
    return XlaHelpers::ConvertElementType(reduce_output / divisor,
                                          input_type(0));
  }
};

REGISTER_XLA_OP(Name("Mean").CompileTimeConstantInput("reduction_indices"),
                MeanOp);

class AllOp : public XlaReductionOp {
 public:
  explicit AllOp(OpKernelConstruction* ctx)
      : XlaReductionOp(ctx, ctx->input_type(0)) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreduction_opsDTcc mht_17(mht_17_v, 374, "", "./tensorflow/compiler/tf2xla/kernels/reduction_ops.cc", "AllOp");
}

  xla::XlaOp InitialValue(xla::XlaBuilder* builder) override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreduction_opsDTcc mht_18(mht_18_v, 379, "", "./tensorflow/compiler/tf2xla/kernels/reduction_ops.cc", "InitialValue");

    return xla::ConstantR0<bool>(builder, true);
  }

  void BuildReducer(xla::XlaBuilder* builder, const xla::XlaOp& scalar_lhs,
                    const xla::XlaOp& scalar_rhs) override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreduction_opsDTcc mht_19(mht_19_v, 387, "", "./tensorflow/compiler/tf2xla/kernels/reduction_ops.cc", "BuildReducer");

    xla::And(scalar_lhs, scalar_rhs);
  }
};

REGISTER_XLA_OP(Name("All").CompileTimeConstantInput("reduction_indices"),
                AllOp);

class AnyOp : public XlaReductionOp {
 public:
  explicit AnyOp(OpKernelConstruction* ctx)
      : XlaReductionOp(ctx, ctx->input_type(0)) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreduction_opsDTcc mht_20(mht_20_v, 401, "", "./tensorflow/compiler/tf2xla/kernels/reduction_ops.cc", "AnyOp");
}

  xla::XlaOp InitialValue(xla::XlaBuilder* builder) override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreduction_opsDTcc mht_21(mht_21_v, 406, "", "./tensorflow/compiler/tf2xla/kernels/reduction_ops.cc", "InitialValue");

    return xla::ConstantR0<bool>(builder, false);
  }

  void BuildReducer(xla::XlaBuilder* builder, const xla::XlaOp& scalar_lhs,
                    const xla::XlaOp& scalar_rhs) override {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSreduction_opsDTcc mht_22(mht_22_v, 414, "", "./tensorflow/compiler/tf2xla/kernels/reduction_ops.cc", "BuildReducer");

    xla::Or(scalar_lhs, scalar_rhs);
  }
};

REGISTER_XLA_OP(Name("Any").CompileTimeConstantInput("reduction_indices"),
                AnyOp);

}  // namespace
}  // namespace tensorflow
