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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdiag_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdiag_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdiag_opDTcc() {
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

#include "tensorflow/compiler/tf2xla/lib/util.h"
#include "tensorflow/compiler/tf2xla/mlir_xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/lib/matrix.h"
#include "tensorflow/compiler/xla/client/lib/pooling.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace {

// Create a diagonal / batch diagonal matrix with 'input' on the diagonal.
xla::XlaOp CreateDiagonal(xla::XlaOp input, int64_t last_dim_size,
                          absl::Span<const int64_t> other_dims) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdiag_opDTcc mht_0(mht_0_v, 204, "", "./tensorflow/compiler/tf2xla/kernels/diag_op.cc", "CreateDiagonal");

  xla::XlaBuilder* builder = input.builder();
  // Create two matrices that have the following forms, and compare them:
  //
  // [[0, 0, 0, 0]            [[0, 1, 2, 3]
  //  [1, 1, 1, 1]             [0, 1, 2, 3]
  //  [2, 2, 2, 2]             [0, 1, 2, 3]
  //  [3, 3, 3, 3]]            [0, 1, 2, 3]]
  //
  // This produces a predicate matrix of the right size, with "true" on the
  // diagonal.
  xla::XlaOp iota = xla::Iota(builder, xla::S32, last_dim_size);
  xla::XlaOp iota_broadcast = xla::Broadcast(iota, {last_dim_size});
  xla::XlaOp mask = xla::Eq(iota_broadcast, iota, {0});

  // If this is a batched diagonal, broadcast the mask across the other
  // dimensions.
  if (!other_dims.empty()) {
    mask = xla::Broadcast(mask, other_dims);
  }

  // Broadcast the input, and then use the mask computed above to select the
  // diagonal:
  // e.g, in 2D:
  //         [[t, f, f]    [[1, 1, 1]    [[0, 0, 0]      [[1, 0, 0]
  // select(  [f, t, f]  ,  [4, 4, 4]  ,  [0, 0, 0]  ) =  [0, 4, 0]
  //          [f, f, t]]    [9, 9, 9]]    [0, 0, 0]]      [0, 0, 9]]
  //
  std::vector<int64_t> out_dim_sizes(other_dims.begin(), other_dims.end());
  out_dim_sizes.push_back(last_dim_size);
  out_dim_sizes.push_back(last_dim_size);

  // Broadcast into the second to last dimension.
  std::vector<int64_t> broadcast_dimensions(other_dims.size() + 1);
  absl::c_iota(broadcast_dimensions, 0);
  ++broadcast_dimensions.back();
  xla::XlaOp input_broadcast =
      xla::BroadcastInDim(input, out_dim_sizes, broadcast_dimensions);
  return xla::Select(mask, input_broadcast, xla::ZerosLike(input_broadcast));
}

class DiagOp : public XlaOpKernel {
 public:
  explicit DiagOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdiag_opDTcc mht_1(mht_1_v, 250, "", "./tensorflow/compiler/tf2xla/kernels/diag_op.cc", "DiagOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdiag_opDTcc mht_2(mht_2_v, 255, "", "./tensorflow/compiler/tf2xla/kernels/diag_op.cc", "Compile");

    OP_REQUIRES(ctx, ctx->num_inputs() >= 1,
                errors::InvalidArgument("Diag op must have at an input"));
    const TensorShape input_shape = ctx->InputShape(0);

    auto dims = input_shape.dim_sizes();
    OP_REQUIRES(ctx, !dims.empty(),
                errors::InvalidArgument("Expected 1 <= dims, got shape ",
                                        input_shape.DebugString()));

    xla::XlaOp input = ctx->Input(0);

    // Picture:
    // tf.diag([1, 2, 3, 4]) ==> [[1, 0, 0, 0]
    //                            [0, 2, 0, 0]
    //                            [0, 0, 3, 0]
    //                            [0, 0, 0, 4]]

    // Flattens the input to 1D.
    int64_t size = input_shape.num_elements();
    input = xla::Reshape(input, {size});

    // Create an R2 with the R1 diagonal.
    xla::XlaOp diag = CreateDiagonal(input, size, /*other_dims=*/{});

    // Reshapes to the final shape.
    std::vector<int64_t> new_dims(dims.size() * 2);
    std::copy(dims.begin(), dims.end(), new_dims.begin());
    std::copy(dims.begin(), dims.end(), new_dims.begin() + dims.size());
    diag = xla::Reshape(diag, new_dims);

    ctx->SetOutput(0, diag);
  }
};

REGISTER_XLA_OP(Name("Diag"), DiagOp);

REGISTER_XLA_OP(Name("DiagPart"), MlirXlaOpKernel);

}  // namespace
}  // namespace tensorflow
