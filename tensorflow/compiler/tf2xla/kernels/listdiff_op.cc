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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSlistdiff_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSlistdiff_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSlistdiff_opDTcc() {
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

// XLA-specific ListDiff Op. This only supports constant DT_INT32 and DT_INT64
// input.

#include <unordered_set>

#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace {

constexpr std::array<DataType, 2> kListDiffTypes = {DT_INT32, DT_INT64};

// ListDiffOp is an XLA kernel that supports constant-only x and y input.
class ListDiffOp : public XlaOpKernel {
 public:
  explicit ListDiffOp(OpKernelConstruction* context) : XlaOpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSlistdiff_opDTcc mht_0(mht_0_v, 207, "", "./tensorflow/compiler/tf2xla/kernels/listdiff_op.cc", "ListDiffOp");
}

  void Compile(XlaOpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSlistdiff_opDTcc mht_1(mht_1_v, 212, "", "./tensorflow/compiler/tf2xla/kernels/listdiff_op.cc", "Compile");

    OP_REQUIRES(context, TensorShapeUtils::IsVector(context->InputShape(0)),
                errors::InvalidArgument("ListDiff expects x as a vector, not ",
                                        context->InputShape(0).DebugString()));

    OP_REQUIRES(context, TensorShapeUtils::IsVector(context->InputShape(1)),
                errors::InvalidArgument("ListDiff expects y as a vector, not ",
                                        context->InputShape(1).DebugString()));

    DataType val_type = context->expected_output_dtype(0);
    DataType idx_type = context->expected_output_dtype(1);

    Status status;
    switch (val_type) {
      case DT_INT32:
        status = ListDiffWithIndexType<int32>(context, idx_type);
        break;
      case DT_INT64:
        status = ListDiffWithIndexType<int64_t>(context, idx_type);
        break;
      default:
        // This should never happen since we restrict this kernel to only match
        // inputs with supported Tensor datatype.
        status = errors::InvalidArgument("ListDiff expects x and y as either ",
                                         "int32 or int64, not ",
                                         DataTypeString(val_type));
    }
    OP_REQUIRES_OK(context, status);
  }

 private:
  template <typename Tval, typename Tidx>
  Status ListDiff(XlaOpKernelContext* context) {
    std::vector<int64_t> x_input, y_input;
    TF_RETURN_IF_ERROR(context->ConstantInputAsIntVector(0, &x_input));
    TF_RETURN_IF_ERROR(context->ConstantInputAsIntVector(1, &y_input));

    std::unordered_set<Tval> y_input_set;
    y_input_set.reserve(y_input.size());
    for (auto y : y_input) {
      y_input_set.insert(y);
    }

    std::vector<Tval> val_output;
    std::vector<Tidx> idx_output;
    auto x_size = x_input.size();
    for (Tidx i = 0; i < x_size; ++i) {
      if (y_input_set.count(x_input[i]) > 0) {
        continue;
      }
      val_output.push_back(x_input[i]);
      idx_output.push_back(i);
    }

    context->SetOutput(0,
                       xla::ConstantR1<Tval>(context->builder(), val_output));
    context->SetOutput(1,
                       xla::ConstantR1<Tidx>(context->builder(), idx_output));
    return Status::OK();
  }

  template <typename Tval>
  Status ListDiffWithIndexType(XlaOpKernelContext* context, DataType idx_type) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSlistdiff_opDTcc mht_2(mht_2_v, 277, "", "./tensorflow/compiler/tf2xla/kernels/listdiff_op.cc", "ListDiffWithIndexType");

    switch (idx_type) {
      case DT_INT32:
        return ListDiff<Tval, int32>(context);
      case DT_INT64:
        return ListDiff<Tval, int64_t>(context);
      default:
        return errors::InvalidArgument(
            "ListDiff expects idx_out as either int32 or int64, not ",
            DataTypeString(idx_type));
    }
  }
};

REGISTER_XLA_OP(Name("ListDiff")
                    .TypeConstraint("T", kListDiffTypes)
                    .CompileTimeConstantInput("x")
                    .CompileTimeConstantInput("y"),
                ListDiffOp);

}  // namespace
}  // namespace tensorflow
