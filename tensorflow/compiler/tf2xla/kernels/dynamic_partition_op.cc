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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdynamic_partition_opDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdynamic_partition_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdynamic_partition_opDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include <vector>

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/tf2xla/literal_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/lib/arithmetic.h"
#include "tensorflow/compiler/xla/client/lib/comparators.h"
#include "tensorflow/compiler/xla/client/lib/constants.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/comparison_util.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/ops_util.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/tpu/tpu_defs.h"

namespace tensorflow {
namespace {

class DynamicPartitionOp : public XlaOpKernel {
 public:
  explicit DynamicPartitionOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdynamic_partition_opDTcc mht_0(mht_0_v, 213, "", "./tensorflow/compiler/tf2xla/kernels/dynamic_partition_op.cc", "DynamicPartitionOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("num_partitions", &num_partitions_));
  }

  // Returns a S32 tensor representing how many items in `input` are equal to
  // `target`
  xla::XlaOp CountS32(XlaOpKernelContext* ctx, xla::XlaOp input,
                      int64_t target) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdynamic_partition_opDTcc mht_1(mht_1_v, 223, "", "./tensorflow/compiler/tf2xla/kernels/dynamic_partition_op.cc", "CountS32");

    xla::XlaOp equal_dim =
        xla::Compare(input, xla::ConstantR0<int32>(ctx->builder(), target), {},
                     xla::ComparisonDirection::kEq);
    xla::XlaOp casted = xla::ConvertElementType(equal_dim, xla::S32);
    return xla::ReduceAll(
        casted, xla::Zero(ctx->builder(), xla::S32),
        xla::CreateScalarAddComputation(xla::S32, ctx->builder()));
  }

  std::pair<std::vector<xla::XlaOp>, std::vector<xla::XlaOp>>
  DynamicPartition1D(XlaOpKernelContext* ctx, xla::XlaOp data_1d,
                     xla::XlaOp partitions_1d, const xla::Shape& data_1d_shape,
                     const xla::Shape& partition_1d_shape) {
    int64_t input_count = data_1d_shape.dimensions(0);
    std::vector<xla::XlaOp> to_sort = {partitions_1d, data_1d};
    std::vector<xla::PrimitiveType> types_to_sort = {
        partition_1d_shape.element_type(), data_1d_shape.element_type()};
    xla::XlaOp sorted = xla::Sort(
        to_sort, xla::CreateScalarLtComputation(types_to_sort, ctx->builder()),
        /*dimension=*/0,
        /*is_stable=*/true);
    xla::XlaOp sorted_partitions = xla::GetTupleElement(sorted, 0);
    xla::XlaOp sorted_data = xla::GetTupleElement(sorted, 1);

    // `partition_length[i]` is length of partition_i
    std::vector<xla::XlaOp> partition_length(num_partitions_);
    // `partition_start[i]` is sum(partition_start[0:i])
    std::vector<xla::XlaOp> partition_start(num_partitions_);
    xla::XlaOp count_so_far = xla::Zero(ctx->builder(), xla::S32);
    for (int64_t i = 0; i < num_partitions_; ++i) {
      xla::XlaOp count = CountS32(ctx, sorted_partitions, /*target=*/i);
      partition_length[i] = count;
      partition_start[i] = count_so_far;
      count_so_far = xla::Add(count_so_far, count);
    }

    // Pad input with `input_count` to avoid OOB -- dynamic slice with
    // OOB slice produces undefined result.
    xla::PaddingConfig padding_config;
    auto* dims = padding_config.add_dimensions();
    dims->set_edge_padding_low(0);
    dims->set_edge_padding_high(input_count);
    dims->set_interior_padding(0);
    auto padded_data =
        xla::Pad(sorted_data, xla::Zero(ctx->builder(), ctx->input_xla_type(0)),
                 padding_config);
    std::vector<xla::XlaOp> output(num_partitions_);
    for (int64_t i = 0; i < num_partitions_; ++i) {
      // Dynamic size will be set later after this function.
      padded_data = xla::RemoveDynamicDimension(padded_data, 0);
      // Slice full size out of the input starting from the offsets.
      auto sliced =
          xla::DynamicSlice(padded_data, {partition_start[i]}, {input_count});
      output[i] = sliced;
    }
    return {output, partition_length};
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSdynamic_partition_opDTcc mht_2(mht_2_v, 285, "", "./tensorflow/compiler/tf2xla/kernels/dynamic_partition_op.cc", "Compile");

    xla::Shape data_shape = ctx->InputXlaShape(0).ConsumeValueOrDie();
    xla::Shape partition_shape = ctx->InputXlaShape(1).ConsumeValueOrDie();
    xla::XlaOp data = ctx->Input(0);
    xla::XlaOp partitions = ctx->Input(1);
    std::vector<int64_t> partitions_static;
    bool partitions_are_static =
        ctx->ConstantInputReshapedToIntVector(1, &partitions_static).ok();
    // We know how to solve DynamicPartition on 1D inputs using
    // DynamicPartition1D. For other input, we do two things:
    //
    // 1. If partition_shape has lower rank than data_shape, we broadcast
    // partition_shape so it's the same as data_shape. This makes
    // partition_shape the same as data_shape.
    //
    // 2. If the data_shape has rank higher than 1, we reshape both data and
    // partition to R1. This reduces the problem to 1D, which we've already
    // solved using DynamicPartition1D.
    //
    // 3. We reshape the result of DynamicPartition1D back from 1D to output
    // shape.
    if (data_shape.rank() > partition_shape.rank()) {
      // Broadcast parititon_shape so that it can be the same as data_shape.
      std::vector<int64_t> broadcasted_dims;
      auto rank = partition_shape.rank();
      broadcasted_dims.reserve(rank);
      for (int64_t i = 0; i < rank; ++i) {
        broadcasted_dims.push_back(i);
      }
      partitions = xla::BroadcastInDim(partitions, data_shape.dimensions(),
                                       broadcasted_dims);
    }

    // Output shape bounded is calculated by
    // [count(partitions)] + data.shape[partitions.ndim:]
    // See also the output shape calculation at
    // https://www.tensorflow.org/api_docs/python/tf/dynamic_partition
    std::vector<int64_t> output_shape_bound_dims;
    output_shape_bound_dims.push_back(
        xla::ShapeUtil::ElementsIn(partition_shape));
    int64_t count_diff = 1;
    for (int64_t i = partition_shape.rank(); i < data_shape.rank(); ++i) {
      output_shape_bound_dims.push_back(data_shape.dimensions(i));
      count_diff *= data_shape.dimensions(i);
    }

    int64_t input_count = xla::ShapeUtil::ElementsIn(data_shape);
    auto data_1d = xla::Reshape(data, {input_count});
    auto partitions_1d = xla::Reshape(partitions, {input_count});
    xla::Shape data_1d_shape =
        xla::ShapeUtil::MakeShape(data_shape.element_type(), {input_count});

    xla::Shape partitions_1d_shape = xla::ShapeUtil::MakeShape(
        partition_shape.element_type(), {input_count});

    std::vector<xla::XlaOp> output, partition_length;
    std::tie(output, partition_length) = DynamicPartition1D(
        ctx, data_1d, partitions_1d, data_1d_shape, partitions_1d_shape);
    for (int64_t i = 0; i < num_partitions_; ++i) {
      auto reshape = xla::Reshape(output[i], output_shape_bound_dims);
      if (partitions_are_static) {
        int64_t size = absl::c_count(partitions_static, i);
        ctx->SetOutput(i, xla::SliceInDim(reshape, 0, size, 1, 0));
      } else {
        xla::XlaOp length;
        if (count_diff != 0) {
          length = xla::Div(partition_length[i],
                            xla::ConstantR0<int32>(ctx->builder(), count_diff));
        } else {
          length = CountS32(ctx, ctx->Input(1), /*target=*/i);
        }
        ctx->SetOutput(i, xla::SetDimensionSize(reshape, length, 0));
      }
    }
  }

 private:
  int64_t num_partitions_;
};

REGISTER_XLA_OP(Name("DynamicPartition"), DynamicPartitionOp);

}  // namespace
}  // namespace tensorflow
