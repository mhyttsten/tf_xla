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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPScross_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPScross_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPScross_opDTcc() {
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

#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"

namespace tensorflow {
namespace {

class CrossOp : public XlaOpKernel {
 public:
  explicit CrossOp(OpKernelConstruction* context) : XlaOpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPScross_opDTcc mht_0(mht_0_v, 195, "", "./tensorflow/compiler/tf2xla/kernels/cross_op.cc", "CrossOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPScross_opDTcc mht_1(mht_1_v, 200, "", "./tensorflow/compiler/tf2xla/kernels/cross_op.cc", "Compile");

    TensorShape in0_shape = ctx->InputShape(0);
    TensorShape in1_shape = ctx->InputShape(1);
    OP_REQUIRES(ctx, in0_shape == in1_shape,
                errors::InvalidArgument("Both inputs must be of same shape: ",
                                        in0_shape.DebugString(), " vs. ",
                                        in1_shape.DebugString()));
    OP_REQUIRES(ctx, in0_shape.dims() >= 1,
                errors::InvalidArgument("Input must be at least 1D",
                                        in0_shape.DebugString()));

    auto inner_dim = in0_shape.dim_size(in0_shape.dims() - 1);
    OP_REQUIRES(ctx, inner_dim == 3,
                errors::FailedPrecondition(
                    "Cross-products are only defined for 3-element vectors."));

    // in0 is a [...,X,Y,Z,3]
    // in1 is the same shape as in0
    // So slice 0 is: in0[...,:,:,:,0:1]
    // So slice 1 is: in0[...,:,:,:,1:2]
    // So slice 2 is: in0[...,:,:,:,2:3]

    std::vector<int64_t> starts(in0_shape.dims(), 0);
    std::vector<int64_t> limits;
    const auto& dim_sizes = in0_shape.dim_sizes();
    limits.reserve(dim_sizes.size());
    for (auto dim_size : in0_shape.dim_sizes()) {
      limits.push_back(dim_size);
    }
    std::vector<int64_t> strides(in0_shape.dims(), 1);

    xla::XlaBuilder* b = ctx->builder();
    auto in0 = ctx->Input(0);
    auto in1 = ctx->Input(1);
    starts.back() = 0;
    limits.back() = 1;
    auto u1 = xla::Slice(in0, starts, limits, strides);
    auto v1 = xla::Slice(in1, starts, limits, strides);
    starts.back() = 1;
    limits.back() = 2;
    auto u2 = xla::Slice(in0, starts, limits, strides);
    auto v2 = xla::Slice(in1, starts, limits, strides);
    starts.back() = 2;
    limits.back() = 3;
    auto u3 = xla::Slice(in0, starts, limits, strides);
    auto v3 = xla::Slice(in1, starts, limits, strides);

    auto s1 = xla::Sub(xla::Mul(u2, v3), xla::Mul(u3, v2));
    auto s2 = xla::Sub(xla::Mul(u3, v1), xla::Mul(u1, v3));
    auto s3 = xla::Sub(xla::Mul(u1, v2), xla::Mul(u2, v1));
    auto output = xla::ConcatInDim(b, {s1, s2, s3}, in0_shape.dims() - 1);

    ctx->SetOutput(0, output);
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CrossOp);
};

REGISTER_XLA_OP(Name("Cross"), CrossOp);

}  // namespace
}  // namespace tensorflow
