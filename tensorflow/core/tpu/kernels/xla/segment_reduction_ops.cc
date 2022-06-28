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
class MHTracer_DTPStensorflowPScorePStpuPSkernelsPSxlaPSsegment_reduction_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSxlaPSsegment_reduction_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSkernelsPSxlaPSsegment_reduction_opsDTcc() {
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

#include "tensorflow/compiler/tf2xla/lib/scatter.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/tpu/tpu_defs.h"

namespace tensorflow {
namespace {

class UnsortedSegmentSum : public XlaOpKernel {
 public:
  explicit UnsortedSegmentSum(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSxlaPSsegment_reduction_opsDTcc mht_0(mht_0_v, 197, "", "./tensorflow/core/tpu/kernels/xla/segment_reduction_ops.cc", "UnsortedSegmentSum");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("T", &dtype_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSxlaPSsegment_reduction_opsDTcc mht_1(mht_1_v, 204, "", "./tensorflow/core/tpu/kernels/xla/segment_reduction_ops.cc", "Compile");

    // output = unsorted_segment_sum(data, indices, num_segments)
    // Compute a tensor such that:
    //    output[i] = sum over {j where indices[j] == i} of data[j]
    //    output[i] == 0 if i does not appear in indices
    //
    // Contrast with segment_sum(), which assumes indices are sorted and that
    // max(indices)+1 is the desired size of the output.
    //
    // The returned output tensor has the same type as data, and the same shape
    // as data with the first indices.rank dimensions are replaced
    // by a single dimension with size num_segments.
    xla::XlaOp data = ctx->Input(0);
    TensorShape data_shape = ctx->InputShape(0);

    xla::XlaOp indices = ctx->Input(1);
    TensorShape indices_shape = ctx->InputShape(1);

    int64_t num_segments;
    OP_REQUIRES_OK(ctx,
                   ctx->ConstantInputAsIntScalar(
                       2, &num_segments, xla::ValueInferenceMode::kUpperBound));

    OP_REQUIRES(ctx, data_shape.dims() >= indices_shape.dims(),
                errors::InvalidArgument(
                    "UnsortedSegmentSum requires that indices' rank be"
                    " less than or equal to data's rank."));
    // Validate that indices.shape is a prefix of data.shape.
    for (int d = 0; d < indices_shape.dims(); ++d) {
      OP_REQUIRES(ctx, (data_shape.dim_size(d) == indices_shape.dim_size(d)),
                  errors::InvalidArgument(
                      "UnsortedSegmentSum requires indices shape to be prefix"
                      " of data_shape, but dimension ",
                      d, " differs ", data_shape.dim_size(d), " vs. ",
                      indices_shape.dim_size(d)));
    }
    xla::XlaBuilder* builder = ctx->builder();
    // data shape = [indices_shape, segment_shape]
    // buffer shape = [num_segment, segment_shape]
    // We now create the buffer shape by reverse enginerring data shape into
    // indices shape and segment shape.
    TensorShape buffer_shape = data_shape;
    buffer_shape.RemoveDimRange(0, indices_shape.dims());
    buffer_shape.InsertDim(0, num_segments);

    auto buffer = xla::Broadcast(XlaHelpers::Zero(builder, dtype_),
                                 buffer_shape.dim_sizes());

    // Build dynamic dim sizes for buffer, as well as whether each dimension
    // size is dynamic or static. We build two parts: num_sgement part and
    // segment_shape part.
    std::vector<xla::XlaOp> buffer_dims;
    std::vector<bool> buffer_dims_are_dynamic;
    // Build the "num_segment" part.
    bool num_segments_is_dynamic;
    OP_REQUIRES_OK(
        ctx, ctx->ResolveInputDynamismIntoPred(2, &num_segments_is_dynamic));

    buffer_dims.insert(buffer_dims.begin(), ctx->Input(2));
    buffer_dims_are_dynamic.insert(buffer_dims_are_dynamic.begin(),
                                   num_segments_is_dynamic);
    // Build the segment shape part.
    for (int64_t i = indices_shape.dims(); i < data_shape.dims(); ++i) {
      buffer_dims.push_back(xla::GetDimensionSize(data, i));
      buffer_dims_are_dynamic.push_back(
          ctx->InputXlaShape(0)->is_dynamic_dimension(i));
    }

    for (int64_t i = 0; i < buffer_dims.size(); ++i) {
      if (buffer_dims_are_dynamic[i]) {
        // For each dynamic dimension, call set-dimension-size on it.
        buffer = xla::SetDimensionSize(buffer, buffer_dims[i], i);
      }
    }

    auto combiner = [](xla::XlaOp a, xla::XlaOp b, xla::XlaBuilder* builder) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSxlaPSsegment_reduction_opsDTcc mht_2(mht_2_v, 282, "", "./tensorflow/core/tpu/kernels/xla/segment_reduction_ops.cc", "lambda");

      return a + b;
    };

    auto result = XlaScatter(buffer, /*updates=*/data, indices,
                             /*indices_are_vectors=*/false, combiner, builder);
    OP_REQUIRES_OK(ctx, result.status());
    ctx->SetOutput(0, result.ValueOrDie());
  }

 private:
  DataType dtype_;
};

REGISTER_XLA_OP(Name("UnsortedSegmentSum")
                    .Device(DEVICE_TPU_XLA_JIT)
                    .CompileTimeConstantInput("num_segments"),
                UnsortedSegmentSum);

}  // namespace
}  // namespace tensorflow
