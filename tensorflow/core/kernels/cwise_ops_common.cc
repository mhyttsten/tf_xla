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
class MHTracer_DTPStensorflowPScorePSkernelsPScwise_ops_commonDTcc {
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
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_ops_commonDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePSkernelsPScwise_ops_commonDTcc() {
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

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/cwise_ops_common.h"

namespace tensorflow {

BinaryOpShared::BinaryOpShared(OpKernelConstruction* ctx, DataType out,
                               DataType in)
    : OpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_ops_commonDTcc mht_0(mht_0_v, 191, "", "./tensorflow/core/kernels/cwise_ops_common.cc", "BinaryOpShared::BinaryOpShared");

#if !defined(INTEL_MKL) || !defined(ENABLE_MKL)
  OP_REQUIRES_OK(ctx, ctx->MatchSignature({in, in}, {out}));
#endif  // !INTEL_MKL || !ENABLE_MKL
}

void BinaryOpShared::SetUnimplementedError(OpKernelContext* ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_ops_commonDTcc mht_1(mht_1_v, 200, "", "./tensorflow/core/kernels/cwise_ops_common.cc", "BinaryOpShared::SetUnimplementedError");

  ctx->SetStatus(errors::Unimplemented(
      "Broadcast between ", ctx->input(0).shape().DebugString(), " and ",
      ctx->input(1).shape().DebugString(), " is not supported yet."));
}

void BinaryOpShared::SetComputeError(OpKernelContext* ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_ops_commonDTcc mht_2(mht_2_v, 209, "", "./tensorflow/core/kernels/cwise_ops_common.cc", "BinaryOpShared::SetComputeError");

  // For speed, errors during compute are caught only via boolean flag, with no
  // associated information.  This is sufficient for now, since the only binary
  // ops that have compute errors are integer division and mod, and the only
  // error they produce is zero division.
  const string& op = ctx->op_kernel().type_string();
  if ((op == "Div" || op == "Mod" || op == "FloorMod" || op == "FloorDiv") &&
      DataTypeIsInteger(ctx->op_kernel().input_type(0))) {
    ctx->CtxFailure(errors::InvalidArgument("Integer division by zero"));
  } else if ((op == "Pow") &&
             DataTypeIsInteger(ctx->op_kernel().input_type(0)) &&
             DataTypeIsSigned(ctx->op_kernel().input_type(1))) {
    ctx->CtxFailure(errors::InvalidArgument(
        "Integers to negative integer powers are not allowed"));
  } else {
    ctx->CtxFailure(
        errors::Internal("Unexpected error in binary operator "
                         "(only integer div and mod should have errors)"));
  }
}

BinaryOpShared::BinaryOpState::BinaryOpState(OpKernelContext* ctx)
    : in0(ctx->input(0)),
      in1(ctx->input(1)),
      bcast(BCast::FromShape(in0.shape()), BCast::FromShape(in1.shape())) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePSkernelsPScwise_ops_commonDTcc mht_3(mht_3_v, 236, "", "./tensorflow/core/kernels/cwise_ops_common.cc", "BinaryOpShared::BinaryOpState::BinaryOpState");

  if (!bcast.IsValid()) {
    bool incompatible_shape_error;
    bool has_attr =
        TryGetNodeAttr(ctx->op_kernel().def(), "incompatible_shape_error",
                       &(incompatible_shape_error));
    if (has_attr && !incompatible_shape_error) {
      const string& op = ctx->op_kernel().type_string();
      OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape({}), &out));
      result = (op == "NotEqual");
      return;
    }

    ctx->SetStatus(errors::InvalidArgument(
        "Incompatible shapes: ", in0.shape().DebugString(), " vs. ",
        in1.shape().DebugString()));
    return;
  }

  const TensorShape output_shape = BCast::ToShape(bcast.output_shape());
  out_num_elements = output_shape.num_elements();
  in0_num_elements = in0.NumElements();
  in1_num_elements = in1.NumElements();
  OP_REQUIRES_OK(ctx, ctx->forward_input_or_allocate_output(
                          {0, 1}, 0, output_shape, &out));

  ndims = static_cast<int>(bcast.x_reshape().size());
}

}  // namespace tensorflow
