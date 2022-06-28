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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSxla_pad_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSxla_pad_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSxla_pad_opDTcc() {
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

#include "absl/algorithm/container.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_compiler.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace {

class XlaPadOp : public XlaOpKernel {
 public:
  explicit XlaPadOp(OpKernelConstruction* context) : XlaOpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSxla_pad_opDTcc mht_0(mht_0_v, 199, "", "./tensorflow/compiler/tf2xla/kernels/xla_pad_op.cc", "XlaPadOp");
}

  void Compile(XlaOpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSxla_pad_opDTcc mht_1(mht_1_v, 204, "", "./tensorflow/compiler/tf2xla/kernels/xla_pad_op.cc", "Compile");

    const TensorShape input_shape = context->InputShape("input");
    const TensorShape padding_value_shape =
        context->InputShape("padding_value");

    std::vector<int64_t> padding_low;
    std::vector<int64_t> padding_high;
    std::vector<int64_t> padding_interior;
    OP_REQUIRES_OK(context, context->ConstantInputAsIntVector("padding_low",
                                                              &padding_low));
    OP_REQUIRES_OK(context, context->ConstantInputAsIntVector("padding_high",
                                                              &padding_high));
    OP_REQUIRES_OK(context, context->ConstantInputAsIntVector(
                                "padding_interior", &padding_interior));

    OP_REQUIRES(context, TensorShapeUtils::IsScalar(padding_value_shape),
                errors::InvalidArgument("padding_value must be a scalar"));
    const int rank = input_shape.dims();
    OP_REQUIRES(context, rank == padding_low.size(),
                errors::InvalidArgument(
                    "The size of padding_low must be equal to the input "
                    "rank (",
                    padding_low.size(), " vs. ", rank, ")"));
    OP_REQUIRES(context, rank == padding_high.size(),
                errors::InvalidArgument(
                    "The size of padding_high must be equal to the input "
                    "rank (",
                    padding_high.size(), " vs. ", rank, ")"));
    OP_REQUIRES(context, rank == padding_interior.size(),
                errors::InvalidArgument(
                    "The size of padding_interior must be equal to the input "
                    "rank (",
                    padding_interior.size(), " vs. ", rank, ")"));

    auto non_negative = [](int64_t x) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSxla_pad_opDTcc mht_2(mht_2_v, 241, "", "./tensorflow/compiler/tf2xla/kernels/xla_pad_op.cc", "lambda");
 return x >= 0; };
    OP_REQUIRES(
        context, absl::c_all_of(padding_interior, non_negative),
        errors::InvalidArgument("padding_interior must be non-negative, got [",
                                absl::StrJoin(padding_interior, ","), "]"));

    xla::PaddingConfig padding_config;
    for (int i = 0; i < rank; ++i) {
      auto* dim = padding_config.add_dimensions();
      dim->set_edge_padding_low(padding_low[i]);
      dim->set_edge_padding_high(padding_high[i]);
      dim->set_interior_padding(padding_interior[i]);
    }

    xla::XlaOp output =
        xla::Pad(context->Input("input"), context->Input("padding_value"),
                 padding_config);
    context->SetOutput(0, output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(XlaPadOp);
};

REGISTER_XLA_OP(Name("XlaPad")
                    .CompileTimeConstantInput("padding_low")
                    .CompileTimeConstantInput("padding_high")
                    .CompileTimeConstantInput("padding_interior"),
                XlaPadOp);

}  // namespace
}  // namespace tensorflow
