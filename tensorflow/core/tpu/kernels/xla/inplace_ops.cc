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
class MHTracer_DTPStensorflowPScorePStpuPSkernelsPSxlaPSinplace_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSxlaPSinplace_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSkernelsPSxlaPSinplace_opsDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include <algorithm>

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/op_kernel.h"

#include "tensorflow/compiler/tf2xla/lib/scatter.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/core/framework/kernel_def_builder.h"

namespace tensorflow {
namespace {

class InplaceUpdateOp : public XlaOpKernel {
 public:
  explicit InplaceUpdateOp(OpKernelConstruction* context)
      : XlaOpKernel(context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSxlaPSinplace_opsDTcc mht_0(mht_0_v, 204, "", "./tensorflow/core/tpu/kernels/xla/inplace_ops.cc", "InplaceUpdateOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSxlaPSinplace_opsDTcc mht_1(mht_1_v, 209, "", "./tensorflow/core/tpu/kernels/xla/inplace_ops.cc", "Compile");

    VLOG(3) << "InplaceUpdateOp::Compile";

    DataType index_type = input_type(1);
    OP_REQUIRES(ctx, index_type == DT_INT32 || index_type == DT_INT64,
                errors::InvalidArgument("index must be int32 or int64"));

    // TF Args are X, I, V
    const TensorShape x_shape = ctx->InputShape(0);
    const TensorShape i_shape = ctx->InputShape(1);
    const TensorShape v_shape = ctx->InputShape(2);

    OP_REQUIRES(ctx,
                TensorShapeUtils::IsScalar(i_shape) ||
                    TensorShapeUtils::IsVector(i_shape),
                errors::InvalidArgument("index must be Rank 0 or 1"));
    OP_REQUIRES(ctx, (x_shape.dims() == v_shape.dims()),
                errors::InvalidArgument("X and V must have the same Rank,"
                                        " X.shape=",
                                        x_shape.DebugString(),
                                        " V.shape=", v_shape.DebugString()));

    auto* builder = ctx->builder();
    auto const_zero = xla::ConstantR0(builder, 0);
    auto current = ctx->Input(0);

    for (int64_t i = 0; i < i_shape.num_elements(); i++) {
      std::vector<xla::XlaOp> update_indices;
      update_indices.push_back(
          xla::Reshape(xla::SliceInDim(ctx->Input(1), i, i + 1, 1, 0), {}));
      for (int xi = 1; xi < x_shape.dims(); xi++) {
        update_indices.push_back(const_zero);
      }
      current = xla::DynamicUpdateSlice(
          current, xla::SliceInDim(ctx->Input(2), i, i + 1, 1, 0),
          update_indices);
    }
    ctx->SetOutput(0, current);

    // TODO(b/118122460): Uncomment+format this code to use XLA Scatter.
    //     auto* builder = ctx->builder();
    //     const auto initial = ctx->Input(0);
    //     const auto indices = ctx->Input(1);
    //     const auto updates = ctx->Input(2);
    //
    //     auto result = XlaScatter(
    //         initial, updates, indices, /*indices_are_vectors=*/false,
    //         [](xla::XlaOp, xla::XlaOp second, xla::XlaBuilder*) { return
    //         second; }, builder);
    //     OP_REQUIRES_OK(ctx, result.status());
    //     ctx->SetOutput(0, result.ValueOrDie());
  }
};

REGISTER_XLA_OP(Name("InplaceUpdate"), InplaceUpdateOp);

class InplaceAddOp : public XlaOpKernel {
 public:
  explicit InplaceAddOp(OpKernelConstruction* context) : XlaOpKernel(context) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSxlaPSinplace_opsDTcc mht_2(mht_2_v, 270, "", "./tensorflow/core/tpu/kernels/xla/inplace_ops.cc", "InplaceAddOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSxlaPSinplace_opsDTcc mht_3(mht_3_v, 275, "", "./tensorflow/core/tpu/kernels/xla/inplace_ops.cc", "Compile");

    VLOG(3) << "InplaceAddOp::Compile";

    DataType index_type = input_type(1);
    OP_REQUIRES(ctx, index_type == DT_INT32 || index_type == DT_INT64,
                errors::InvalidArgument("index must be int32 or int64"));

    // TF Args are X, I, V
    const TensorShape x_shape = ctx->InputShape(0);
    const TensorShape i_shape = ctx->InputShape(1);
    const TensorShape v_shape = ctx->InputShape(2);
    OP_REQUIRES(ctx,
                (TensorShapeUtils::IsScalar(i_shape) ||
                 ((i_shape.dims() == 1) && (i_shape.num_elements() == 1))),
                errors::InvalidArgument("index must be Rank 1 and size 1"));
    OP_REQUIRES(ctx, (x_shape.dims() == v_shape.dims()),
                errors::InvalidArgument("X and V must have the same Rank,"
                                        " X.shape=",
                                        x_shape.DebugString(),
                                        " V.shape=", v_shape.DebugString()));
    // Pad the indices out to the match the rank of params.
    auto* builder = ctx->builder();
    std::vector<xla::XlaOp> padded_indices;
    padded_indices.push_back(xla::Reshape(ctx->Input(1), {}));
    for (int i = 0; i < x_shape.dims() - 1; ++i) {
      padded_indices.push_back(XlaHelpers::Zero(builder, index_type));
    }

    std::vector<int64_t> sizes;
    sizes.push_back(1);
    for (int i = 1; i < x_shape.dims(); i++) {
      sizes.push_back(x_shape.dim_size(i));
    }

    auto prev = xla::DynamicSlice(ctx->Input(0), padded_indices, sizes);
    auto updated = xla::Add(prev, ctx->Input(2));
    auto result =
        xla::DynamicUpdateSlice(ctx->Input(0), updated, padded_indices);
    ctx->SetOutput(0, result);
  }
};

REGISTER_XLA_OP(Name("InplaceAdd"), InplaceAddOp);

}  // namespace
}  // namespace tensorflow
