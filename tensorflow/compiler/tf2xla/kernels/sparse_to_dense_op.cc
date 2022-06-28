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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsparse_to_dense_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsparse_to_dense_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsparse_to_dense_opDTcc() {
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

#include "tensorflow/compiler/tf2xla/lib/scatter.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"

namespace tensorflow {
namespace {

// Operator to convert sparse representations to dense.
class SparseToDenseOp : public XlaOpKernel {
 public:
  explicit SparseToDenseOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsparse_to_dense_opDTcc mht_0(mht_0_v, 196, "", "./tensorflow/compiler/tf2xla/kernels/sparse_to_dense_op.cc", "SparseToDenseOp");
}

  void Compile(XlaOpKernelContext* context) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSsparse_to_dense_opDTcc mht_1(mht_1_v, 201, "", "./tensorflow/compiler/tf2xla/kernels/sparse_to_dense_op.cc", "Compile");

    // sparse_indices
    const TensorShape indices_shape = context->InputShape(0);
    OP_REQUIRES(context, indices_shape.dims() <= 2,
                errors::InvalidArgument(
                    "sparse_indices should be a scalar, vector, or matrix, "
                    "got shape ",
                    indices_shape.DebugString()));
    const int64_t num_elems =
        indices_shape.dims() > 0 ? indices_shape.dim_size(0) : 1;
    const int64_t num_dims =
        indices_shape.dims() > 1 ? indices_shape.dim_size(1) : 1;

    // output_shape
    TensorShape output_shape;
    OP_REQUIRES_OK(context,
                   context->ConstantInputAsShape(
                       1, &output_shape, xla::ValueInferenceMode::kUpperBound));
    OP_REQUIRES(context, output_shape.dims() == num_dims,
                errors::InvalidArgument(
                    "output_shape has incorrect number of elements: ",
                    output_shape.num_elements(), " should be: ", num_dims));

    // sparse_values
    const TensorShape sparse_values_shape = context->InputShape(2);
    const int64_t num_values = sparse_values_shape.num_elements();
    OP_REQUIRES(
        context,
        sparse_values_shape.dims() == 0 ||
            (sparse_values_shape.dims() == 1 && num_values == num_elems),
        errors::InvalidArgument("sparse_values has incorrect shape ",
                                sparse_values_shape.DebugString(),
                                ", should be [] or [", num_elems, "]"));

    // default_value
    const TensorShape default_value_shape = context->InputShape(3);
    OP_REQUIRES(context, TensorShapeUtils::IsScalar(default_value_shape),
                errors::InvalidArgument("default_value should be a scalar."));

    xla::XlaOp indices = context->Input(0);
    xla::XlaOp sparse_values = context->Input(2);
    xla::XlaOp default_value = context->Input(3);

    if (sparse_values_shape.dims() == 0 && num_elems != 1) {
      sparse_values = Broadcast(sparse_values, {num_elems});
    }
    xla::XlaBuilder* builder = context->builder();
    auto buffer = Broadcast(default_value, output_shape.dim_sizes());
    std::vector<bool> dynamic_dims;
    OP_REQUIRES_OK(
        context, context->ResolveInputDynamismIntoPredVector(1, &dynamic_dims));

    for (int64_t i = 0; i < dynamic_dims.size(); ++i) {
      // If a dimension is dynamic, call set-dimension-size on the output.
      if (dynamic_dims[i]) {
        auto dynamic_dim_size =
            xla::Slice(context->Input(1), {i}, {i + 1}, {1});
        dynamic_dim_size = xla::Reshape(dynamic_dim_size, {});
        dynamic_dim_size = xla::ConvertElementType(dynamic_dim_size, xla::S32);
        buffer = xla::SetDimensionSize(buffer, dynamic_dim_size, i);
      }
    }
    auto result = XlaScatter(buffer, sparse_values, indices,
                             /*indices_are_vectors=*/indices_shape.dims() > 1,
                             /*combiner=*/{}, builder);
    context->SetOutput(0, builder->ReportErrorOrReturn(result));
  }
};

REGISTER_XLA_OP(Name("SparseToDense").CompileTimeConstantInput("output_shape"),
                SparseToDenseOp);

}  // namespace

}  // namespace tensorflow
