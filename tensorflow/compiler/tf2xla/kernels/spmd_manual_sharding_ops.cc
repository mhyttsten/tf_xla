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
class MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSspmd_manual_sharding_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSspmd_manual_sharding_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSspmd_manual_sharding_opsDTcc() {
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

#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/tf2xla/xla_helpers.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/sharding_op_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {
namespace {

xla::OpSharding GetManualSharding(const xla::OpSharding& original,
                                  int64 single_dim) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSspmd_manual_sharding_opsDTcc mht_0(mht_0_v, 200, "", "./tensorflow/compiler/tf2xla/kernels/spmd_manual_sharding_ops.cc", "GetManualSharding");

  xla::OpSharding manual;
  if (single_dim < 0 || original.type() != xla::OpSharding::OTHER) {
    manual.set_type(xla::OpSharding::MANUAL);
    return manual;
  }
  manual.set_type(xla::OpSharding::OTHER);
  std::vector<int64_t> new_tile_shape(
      original.tile_assignment_dimensions().begin(),
      original.tile_assignment_dimensions().end());
  new_tile_shape.push_back(new_tile_shape[single_dim]);
  new_tile_shape[single_dim] = 1;
  xla::Array<int64_t> new_tile(new_tile_shape);
  new_tile.Each([&](absl::Span<const int64> indices, int64* v) {
    int64 src_index = 0;
    for (int64 i = 0; i < indices.size() - 1; ++i) {
      if (i > 0) {
        src_index *= new_tile_shape[i];
      }
      int64 index = indices[i];
      if (i == single_dim) {
        index = indices.back();
      }
      src_index += index;
    }
    *v = original.tile_assignment_devices(src_index);
  });
  for (int64 dim : new_tile_shape) {
    manual.add_tile_assignment_dimensions(dim);
  }
  for (int64 device : new_tile) {
    manual.add_tile_assignment_devices(device);
  }
  if (original.replicate_on_last_tile_dim()) {
    manual.add_last_tile_dims(xla::OpSharding::REPLICATED);
  }
  for (int64 type : original.last_tile_dims()) {
    manual.add_last_tile_dims(static_cast<xla::OpSharding::Type>(type));
  }
  manual.add_last_tile_dims(xla::OpSharding::MANUAL);
  return manual;
}

class XlaSpmdFullToShardShapeOp : public XlaOpKernel {
 public:
  explicit XlaSpmdFullToShardShapeOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSspmd_manual_sharding_opsDTcc mht_1(mht_1_v, 249, "", "./tensorflow/compiler/tf2xla/kernels/spmd_manual_sharding_ops.cc", "XlaSpmdFullToShardShapeOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("manual_sharding", &manual_sharding_str_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dim", &single_dim_));
    std::vector<int32_t> unspecified_dims;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("unspecified_dims", &unspecified_dims));
    for (int32_t i32 : unspecified_dims) {
      unspecified_dims_.push_back(i32);
    }
  }

  ~XlaSpmdFullToShardShapeOp() override = default;

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSspmd_manual_sharding_opsDTcc mht_2(mht_2_v, 264, "", "./tensorflow/compiler/tf2xla/kernels/spmd_manual_sharding_ops.cc", "Compile");

    xla::XlaOp input = ctx->Input(0);
    auto input_shape_or = ctx->InputXlaShape(0);
    OP_REQUIRES_OK(ctx, input_shape_or.status());
    xla::OpSharding sharding;
    if (!sharding.ParseFromString(manual_sharding_str_)) {
      OP_REQUIRES_OK(ctx,
                     xla::InvalidArgument("manual_sharding attribute was not a "
                                          "valid encoded xla::OpSharding "
                                          "proto."));
    }
    auto output_shape = input_shape_or.ValueOrDie();
    int64_t rank = output_shape.rank();
    if (sharding.type() == xla::OpSharding::OTHER) {
      for (int64_t i = 0; i < rank; ++i) {
        if (single_dim_ >= 0 && i != single_dim_) {
          continue;
        }
        int64_t partitions_i = sharding.tile_assignment_dimensions(i);
        if (partitions_i == 1) continue;
        int64_t dim_size =
            xla::CeilOfRatio(output_shape.dimensions(i), partitions_i);
        output_shape.set_dimensions(i, dim_size);
      }
    }
    xla::XlaOp input_annotation;
    {
      // Annotate the full-shape input with the sharding.
      xla::XlaScopedShardingAssignment assign_sharding(ctx->builder(),
                                                       sharding);
      input_annotation = xla::CustomCall(
          ctx->builder(), /*call_target_name=*/"Sharding", {input},
          input_shape_or.ValueOrDie(),
          /*opaque=*/
          xla::sharding_op_util::EncodeAttributes(unspecified_dims_));
    }

    {
      // Annotate the shard-shape output with manual sharding, so that the
      // partitioner will leave it as is.
      xla::OpSharding manual = GetManualSharding(sharding, single_dim_);
      xla::XlaScopedShardingAssignment assign_sharding(ctx->builder(), manual);
      auto output = xla::CustomCall(
          ctx->builder(),
          /*call_target_name=*/"SPMDFullToShardShape", {input_annotation},
          output_shape,
          /*opaque=*/
          xla::sharding_op_util::EncodeAttributes(unspecified_dims_));
      ctx->SetOutput(0, output);
    }
  }

 private:
  string manual_sharding_str_;
  int32 single_dim_;
  std::vector<int64_t> unspecified_dims_;
  TF_DISALLOW_COPY_AND_ASSIGN(XlaSpmdFullToShardShapeOp);
};

class XlaSpmdShardToFullShapeOp : public XlaOpKernel {
 public:
  explicit XlaSpmdShardToFullShapeOp(OpKernelConstruction* ctx)
      : XlaOpKernel(ctx) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSspmd_manual_sharding_opsDTcc mht_3(mht_3_v, 329, "", "./tensorflow/compiler/tf2xla/kernels/spmd_manual_sharding_ops.cc", "XlaSpmdShardToFullShapeOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("full_shape", &full_shape_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("manual_sharding", &manual_sharding_str_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dim", &single_dim_));
    std::vector<int32_t> unspecified_dims;
    OP_REQUIRES_OK(ctx, ctx->GetAttr("unspecified_dims", &unspecified_dims));
    for (int32_t i32 : unspecified_dims) {
      unspecified_dims_.push_back(i32);
    }
  }

  ~XlaSpmdShardToFullShapeOp() override = default;

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPStf2xlaPSkernelsPSspmd_manual_sharding_opsDTcc mht_4(mht_4_v, 345, "", "./tensorflow/compiler/tf2xla/kernels/spmd_manual_sharding_ops.cc", "Compile");

    xla::XlaOp input = ctx->Input(0);
    auto input_shape_or = ctx->InputXlaShape(0);
    OP_REQUIRES_OK(ctx, input_shape_or.status());
    auto output_shape = TensorShapeToXLAShape(
        input_shape_or.ValueOrDie().element_type(), full_shape_);

    xla::OpSharding sharding;
    if (!sharding.ParseFromString(manual_sharding_str_)) {
      OP_REQUIRES_OK(ctx,
                     xla::InvalidArgument("manual_sharding attribute was not a "
                                          "valid encoded xla::OpSharding "
                                          "proto."));
    }
    xla::XlaOp input_annotation;
    {
      // Annotate the shard-shape input with manual sharding, so that the
      // partitioner will leave it as is.
      xla::OpSharding manual = GetManualSharding(sharding, single_dim_);
      xla::XlaScopedShardingAssignment assign_sharding(ctx->builder(), manual);
      input_annotation = xla::CustomCall(
          ctx->builder(), /*call_target_name=*/"Sharding", {input},
          input_shape_or.ValueOrDie(),
          xla::sharding_op_util::EncodeAttributes(unspecified_dims_));
    }

    {
      // Annotate the full-shape output with the sharding.
      xla::XlaScopedShardingAssignment assign_sharding(ctx->builder(),
                                                       sharding);
      ctx->SetOutput(
          0, xla::CustomCall(
                 ctx->builder(),
                 /*call_target_name=*/"SPMDShardToFullShape",
                 {input_annotation}, output_shape,
                 xla::sharding_op_util::EncodeAttributes(unspecified_dims_)));
    }
  }

 private:
  TensorShape full_shape_;
  string manual_sharding_str_;
  int32 single_dim_;
  std::vector<int64_t> unspecified_dims_;
  TF_DISALLOW_COPY_AND_ASSIGN(XlaSpmdShardToFullShapeOp);
};

REGISTER_XLA_OP(Name("XlaSpmdFullToShardShape"), XlaSpmdFullToShardShapeOp);
REGISTER_XLA_OP(Name("XlaSpmdShardToFullShape"), XlaSpmdShardToFullShapeOp);

}  // namespace
}  // namespace tensorflow
