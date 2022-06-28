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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSunary_ops_compositionDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSunary_ops_compositionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSunary_ops_compositionDTcc() {
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

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/tf2xla/kernels/cwise_ops.h"
#include "tensorflow/compiler/tf2xla/kernels/elu_op.h"
#include "tensorflow/compiler/tf2xla/kernels/relu_op.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/math.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/kernel_def_builder.h"

namespace tensorflow {
namespace {

using XlaUnaryOpGenerator = std::function<xla::XlaOp(xla::XlaOp)>;
using XlaOpGeneratorMap = absl::flat_hash_map<string, XlaUnaryOpGenerator>;

void PopulateXlaOpGeneratorMap(XlaOpGeneratorMap* op_generator_map) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSunary_ops_compositionDTcc mht_0(mht_0_v, 206, "", "./tensorflow/compiler/tf2xla/kernels/unary_ops_composition.cc", "PopulateXlaOpGeneratorMap");

  auto add_xla_op_generator = [&](std::string name,
                                  XlaUnaryOpGenerator xla_op_generator) {
   std::vector<std::string> mht_1_v;
   mht_1_v.push_back("name: \"" + name + "\"");
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSunary_ops_compositionDTcc mht_1(mht_1_v, 212, "", "./tensorflow/compiler/tf2xla/kernels/unary_ops_composition.cc", "lambda");

    CHECK(op_generator_map->insert({name, xla_op_generator}).second);
  };

#define ADD_XLA_OP_GENERATOR(Name) add_xla_op_generator(#Name, xla::Name);

  ADD_XLA_OP_GENERATOR(Abs);
  ADD_XLA_OP_GENERATOR(Acos);
  ADD_XLA_OP_GENERATOR(Acosh);
  ADD_XLA_OP_GENERATOR(Asin);
  ADD_XLA_OP_GENERATOR(Asinh);
  ADD_XLA_OP_GENERATOR(Atan);
  ADD_XLA_OP_GENERATOR(Atanh);
  ADD_XLA_OP_GENERATOR(Ceil);
  ADD_XLA_OP_GENERATOR(Cos);
  ADD_XLA_OP_GENERATOR(Cosh);
  ADD_XLA_OP_GENERATOR(Expm1);
  ADD_XLA_OP_GENERATOR(Exp);
  ADD_XLA_OP_GENERATOR(Floor);
  add_xla_op_generator(
      "Inv", [](xla::XlaOp x) { return xla::ScalarLike(x, 1.0) / x; });
  ADD_XLA_OP_GENERATOR(Log);
  ADD_XLA_OP_GENERATOR(Log1p);
  ADD_XLA_OP_GENERATOR(Neg);
  ADD_XLA_OP_GENERATOR(Reciprocal);
  add_xla_op_generator("Rint", xla::RoundToEven);
  ADD_XLA_OP_GENERATOR(Round);
  ADD_XLA_OP_GENERATOR(Rsqrt);
  add_xla_op_generator("Sigmoid", xla::Logistic);
  ADD_XLA_OP_GENERATOR(Sin);
  ADD_XLA_OP_GENERATOR(Sinh);
  ADD_XLA_OP_GENERATOR(Sqrt);
  ADD_XLA_OP_GENERATOR(Square);
  ADD_XLA_OP_GENERATOR(Tan);
  ADD_XLA_OP_GENERATOR(Tanh);

  ADD_XLA_OP_GENERATOR(Elu);
  ADD_XLA_OP_GENERATOR(Relu);
  ADD_XLA_OP_GENERATOR(Relu6);
  ADD_XLA_OP_GENERATOR(Selu);

#undef ADD_XLA_OP_GENERATOR
}

const XlaOpGeneratorMap& GetXlaOpGeneratorMap() {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSunary_ops_compositionDTcc mht_2(mht_2_v, 259, "", "./tensorflow/compiler/tf2xla/kernels/unary_ops_composition.cc", "GetXlaOpGeneratorMap");

  static XlaOpGeneratorMap* result = []() {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSunary_ops_compositionDTcc mht_3(mht_3_v, 263, "", "./tensorflow/compiler/tf2xla/kernels/unary_ops_composition.cc", "lambda");

    auto* result = new XlaOpGeneratorMap;
    PopulateXlaOpGeneratorMap(result);
    return result;
  }();

  return *result;
}

class UnaryOpsCompositionOp : public XlaOpKernel {
 public:
  explicit UnaryOpsCompositionOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSunary_ops_compositionDTcc mht_4(mht_4_v, 277, "", "./tensorflow/compiler/tf2xla/kernels/unary_ops_composition.cc", "UnaryOpsCompositionOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("op_names", &op_names_));

    const XlaOpGeneratorMap& op_generator_map = GetXlaOpGeneratorMap();
    for (absl::string_view op_name : op_names_) {
      OP_REQUIRES(ctx, op_generator_map.contains(op_name),
                  errors::Unimplemented(
                      op_name, " not supported in _UnaryOpsComposition"));
    }
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSunary_ops_compositionDTcc mht_5(mht_5_v, 291, "", "./tensorflow/compiler/tf2xla/kernels/unary_ops_composition.cc", "Compile");

    xla::XlaOp x = ctx->Input(0);
    const XlaOpGeneratorMap& op_generator_map = GetXlaOpGeneratorMap();
    for (absl::string_view op_name : op_names_) {
      x = op_generator_map.find(op_name)->second(x);
    }
    ctx->SetOutput(0, x);
  }

 private:
  std::vector<string> op_names_;
};

REGISTER_XLA_OP(Name("_UnaryOpsComposition"), UnaryOpsCompositionOp);

}  // namespace
}  // namespace tensorflow
