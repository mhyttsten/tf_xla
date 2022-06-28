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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStile_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStile_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStile_opsDTcc() {
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

// XLA-specific Tile Op.

#include <vector>

#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/tf2xla/lib/broadcast.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/value_inference.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_index.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace {

// --------------------------------------------------------------------------
class TileOp : public XlaOpKernel {
 public:
  explicit TileOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStile_opsDTcc mht_0(mht_0_v, 212, "", "./tensorflow/compiler/tf2xla/kernels/tile_ops.cc", "TileOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPStile_opsDTcc mht_1(mht_1_v, 217, "", "./tensorflow/compiler/tf2xla/kernels/tile_ops.cc", "Compile");

    const TensorShape input_shape = ctx->InputShape("input");
    const TensorShape multiples_shape = ctx->InputShape("multiples");

    OP_REQUIRES(
        ctx, TensorShapeUtils::IsVector(multiples_shape),
        errors::InvalidArgument("Expected multiples to be 1-D, but got shape ",
                                multiples_shape.DebugString()));
    OP_REQUIRES(ctx, input_shape.dims() == multiples_shape.num_elements(),
                errors::InvalidArgument(
                    "Expected multiples argument to be a vector of length ",
                    input_shape.dims(), " but got length ",
                    multiples_shape.dim_size(0)));
    const int input_dims = input_shape.dims();
    auto input = ctx->Input(0);
    // If input is a scalar then multiples has 0 elements and this is
    // a NoOp.
    if (input_dims == 0) {
      ctx->SetOutput(0, input);
      return;
    }

    std::vector<int64_t> multiples_bounds;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputAsIntVector(
                            "multiples", &multiples_bounds,
                            xla::ValueInferenceMode::kUpperBound));

    std::vector<int64_t> output_dims(input_shape.dims());
    for (int64_t i = 0; i < input_shape.dims(); ++i) {
      OP_REQUIRES(ctx, multiples_bounds[i] >= 0,
                  errors::InvalidArgument("Expected multiples[", i,
                                          "] >= 0, but got ", output_dims[i]));
      output_dims[i] = input_shape.dim_size(i) * multiples_bounds[i];
    }

    std::vector<bool> multiples_are_dynamic;

    OP_REQUIRES_OK(ctx, ctx->ResolveInputDynamismIntoPredVector(
                            1, &multiples_are_dynamic));

    bool all_multiples_are_static = absl::c_all_of(
        multiples_are_dynamic, [](bool dynamic) { return !dynamic; });
    // If a value is static, it means the upper bound is the value itself:
    // constant_value = constant_upper_boudn = counstant_lower_bound
    if (all_multiples_are_static) {
      // If all multiples are 1, than the input is the same as the output.
      if (absl::c_all_of(multiples_bounds,
                         [](int64_t multiple) { return multiple == 1; })) {
        ctx->SetOutput(0, input);
        return;
      }
    }

    auto result_or = BroadcastTo(ctx->Input("input"), output_dims);

    OP_REQUIRES_OK(ctx, result_or.status());
    auto result = result_or.ValueOrDie();
    if (!all_multiples_are_static) {
      // Some values of multiples are unknown at compile time, this is a dynamic
      // tile op. We need to call set dimension size.
      for (int64_t i = 0; i < multiples_are_dynamic.size(); ++i) {
        if (!multiples_are_dynamic[i]) {
          continue;
        }
        // If a dimension is dynamic, call set-dimension-size on the output.
        auto dynamic_dim_size =
            xla::Slice(ctx->Input("multiples"), {i}, {i + 1}, {1});
        dynamic_dim_size = xla::Reshape(dynamic_dim_size, {});
        dynamic_dim_size = xla::ConvertElementType(dynamic_dim_size, xla::S32);
        result = xla::SetDimensionSize(result, dynamic_dim_size, i);
      }
    }

    ctx->SetOutput(0, result);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(TileOp);
};

REGISTER_XLA_OP(Name("Tile").CompileTimeConstantInput("multiples"), TileOp);

}  // namespace
}  // namespace tensorflow
