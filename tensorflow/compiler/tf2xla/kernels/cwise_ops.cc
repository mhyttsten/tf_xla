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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPScwise_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPScwise_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPScwise_opsDTcc() {
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

// XLA-specific base classes for Unary and Binary Ops.

#include "tensorflow/compiler/tf2xla/kernels/cwise_ops.h"

#include "tensorflow/compiler/tf2xla/lib/broadcast.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {

void XlaBinaryOp::Compile(XlaOpKernelContext* ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPScwise_opsDTcc mht_0(mht_0_v, 205, "", "./tensorflow/compiler/tf2xla/kernels/cwise_ops.cc", "XlaBinaryOp::Compile");

  TensorShape lhs_shape = ctx->InputShape(0);
  TensorShape rhs_shape = ctx->InputShape(1);
  xla::Shape lhs_xla_shape = ctx->InputXlaShape(0).ValueOrDie();
  xla::Shape rhs_xla_shape = ctx->InputXlaShape(1).ValueOrDie();
  // Fetch the expressions containing the input tensors.
  auto lhs_handle = ctx->Input(0);
  auto rhs_handle = ctx->Input(1);
  if (lhs_shape.dims() == rhs_shape.dims()) {
    auto reconcile_tensor_mismatched_dims =
        [ctx](xla::XlaOp op, const xla::Shape& lhs_xla_shape,
              const xla::Shape& rhs_xla_shape, TensorShape* lhs_tensor_shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPScwise_opsDTcc mht_1(mht_1_v, 219, "", "./tensorflow/compiler/tf2xla/kernels/cwise_ops.cc", "lambda");

          // Find out mismatched dimensions that are non-broadcastable.
          // Reconcile the
          // difference by slicing the bigger dimension.
          for (int64_t i = 0; i < lhs_xla_shape.rank(); ++i) {
            if (lhs_xla_shape.is_dynamic_dimension(i)) {
              if (!rhs_xla_shape.is_dynamic_dimension(i) &&
                  lhs_xla_shape.dimensions(i) > rhs_xla_shape.dimensions(i) &&
                  rhs_xla_shape.dimensions(i) != 1) {
                // e.g., :
                // lhs = [..., <=N, ...]
                // rhs = [..., 2  , ...]
                // Slice N into 2.
                // Size 1 dim doesn't need slice as the other side is
                // broadcastable.
                auto size = xla::GetDimensionSize(op, i);
                op = xla::SliceInDim(op, 0, rhs_xla_shape.dimensions(i), 1,
                                     /*dimno=*/i);
                lhs_tensor_shape->set_dim(i, rhs_xla_shape.dimensions(i));
                // Propagate dynamic dimension.
                op = xla::SetDimensionSize(op, size, i);
              }
              if (rhs_xla_shape.is_dynamic_dimension(i) &&
                  lhs_xla_shape.dimensions(i) < rhs_xla_shape.dimensions(i) &&
                  rhs_xla_shape.dimensions(i) != 1 &&
                  lhs_xla_shape.dimensions(i) != 1) {
                // e.g., :
                // lhs = [..., <=M, ...]
                // rhs = [..., <=N  , ...]
                // where M < N
                //
                // In this case we pad M into N to make the bounds the same.
                // Note that we can't slice N into M because M could be a
                // dynamic size 1 dim that's meant to be broadcasted to N.
                auto size = xla::GetDimensionSize(op, i);
                int64_t diff =
                    rhs_xla_shape.dimensions(i) - lhs_xla_shape.dimensions(i);
                op = xla::PadInDim(
                    op, xla::Zero(ctx->builder(), lhs_xla_shape.element_type()),
                    i, 0, diff);
                lhs_tensor_shape->set_dim(i, rhs_xla_shape.dimensions(i));
                // Propagate dynamic dimension.
                op = xla::SetDimensionSize(op, size, i);
              }
            }
          }
          return op;
        };
    lhs_handle = reconcile_tensor_mismatched_dims(lhs_handle, lhs_xla_shape,
                                                  rhs_xla_shape, &lhs_shape);
    rhs_handle = reconcile_tensor_mismatched_dims(rhs_handle, rhs_xla_shape,
                                                  lhs_xla_shape, &rhs_shape);
  }
  // By TensorFlow conventions the inputs may not have the same
  // shapes, in which case they will be automatically broadcast if
  // possible before mapping. Use the standard TensorFlow helper to
  // compute valid broadcast shapes, but rely below on XLA to
  // automatically perform the broadcast assuming its valid shapes are
  // a superset of TensorFlow's valid shapes.
  BCast bcast(BCast::FromShape(lhs_shape), BCast::FromShape(rhs_shape),
              /*fewer_dims_optimization=*/false);
  if (!bcast.IsValid()) {
    ctx->SetStatus(errors::InvalidArgument("Incompatible shapes: ",
                                           lhs_shape.DebugString(), " vs. ",
                                           rhs_shape.DebugString()));
    return;
  }

  // If the ranks of the inputs don't match, TensorFlow automatically
  // reshapes the smaller by padding with dimensions of size 1 as a
  // prefix. In other words to pad a 5-vector to a 3-dimensional
  // tensor it is reshaped to have shape [1,1,5]. XLA's automatic
  // broadcast code is able to broadcast from lower to higher rank,
  // but doesn't assume you want to pad as a prefix of the dimensions,
  // and instead needs to be told which dimensions of the higher rank
  // tensor to match to the lower rank tensor. In this example it
  // would be dimensions [2]. If we were matching a matrix against a
  // 4-D tensor the dimensions to match would be [2,3],
  // etc. extend_dimension encodes the general case.
  std::vector<int64_t> extend_dimension;
  int max_rank = std::max(lhs_shape.dims(), rhs_shape.dims());
  int min_rank = std::min(lhs_shape.dims(), rhs_shape.dims());
  if (min_rank != max_rank) {
    for (int i = 0; i < min_rank; ++i) {
      // Match the lower rank tensor along the larger-numbered
      // dimensions of the higher rank tensor.
      extend_dimension.push_back(max_rank - min_rank + i);
    }
  }

  // Call virtual method to emit the computation.
  xla::XlaOp output =
      Computation(ctx, lhs_handle, lhs_shape.dim_sizes(), rhs_handle,
                  rhs_shape.dim_sizes(), bcast, extend_dimension);

  // The TensorFlow helper computed the post-broadcast shape in
  // output_shape: we rely on subclassed Computations to implement the
  // same broadcast semantics.
  ctx->SetOutput(0, output);
}

/* static */ std::pair<xla::XlaOp, xla::XlaOp> XlaBinaryOp::Broadcast(
    xla::XlaOp lhs, xla::XlaOp rhs, const BCast& broadcast_helper) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPScwise_opsDTcc mht_2(mht_2_v, 324, "", "./tensorflow/compiler/tf2xla/kernels/cwise_ops.cc", "XlaBinaryOp::Broadcast");

  auto lhs_output = BroadcastTo(lhs, broadcast_helper.output_shape());
  if (!lhs_output.ok()) {
    xla::XlaOp error = lhs.builder()->ReportError(lhs_output.status());
    return {error, error};
  }
  auto rhs_output = BroadcastTo(rhs, broadcast_helper.output_shape());
  if (!rhs_output.ok()) {
    xla::XlaOp error = rhs.builder()->ReportError(rhs_output.status());
    return {error, error};
  }
  return {lhs_output.ValueOrDie(), rhs_output.ValueOrDie()};
}

}  // namespace tensorflow
