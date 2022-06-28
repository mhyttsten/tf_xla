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
class MHTracer_DTPStensorflowPScorePStpuPSkernelsPSxlaPSinfeed_opDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSxlaPSinfeed_opDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSkernelsPSxlaPSinfeed_opDTcc() {
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
#include "tensorflow/compiler/tf2xla/sharding_util.h"
#include "tensorflow/compiler/tf2xla/type_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/tpu/tpu_api.h"
#include "tensorflow/core/tpu/tpu_defs.h"
#include "tensorflow/stream_executor/tpu/c_api_conversions.h"
#include "tensorflow/stream_executor/tpu/c_api_decl.h"

namespace tensorflow {

namespace {

xla::Shape GetTPUInfeedLayout(const xla::Shape& shape) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSxlaPSinfeed_opDTcc mht_0(mht_0_v, 203, "", "./tensorflow/core/tpu/kernels/xla/infeed_op.cc", "GetTPUInfeedLayout");

  XLA_Shape c_shape;
  XLA_Shape c_infeed_shape;

  ApiConverter::ToC(shape, &c_shape);

  tpu::ExecutorApiFn()->TpuTransferManager_GetInfeedLayoutFn(&c_shape,
                                                             &c_infeed_shape);
  xla::Shape infeed_shape = ApiConverter::FromC(&c_infeed_shape);
  ApiConverter::Destroy(&c_shape);
  ApiConverter::Destroy(&c_infeed_shape);
  return infeed_shape;
}

// Updates the layout of the given infeed shape, optionally considering the
// sharding of the op. If the op has tile sharding, assign the layout based on
// the shard shape.
Status UpdateInfeedLayout(xla::Shape* shape,
                          absl::optional<xla::OpSharding> sharding) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSxlaPSinfeed_opDTcc mht_1(mht_1_v, 224, "", "./tensorflow/core/tpu/kernels/xla/infeed_op.cc", "UpdateInfeedLayout");

  if (sharding && sharding->type() == xla::OpSharding::OTHER) {
    TF_ASSIGN_OR_RETURN(auto hlo_sharding,
                        xla::HloSharding::FromProto(*sharding));
    for (int64_t i = 0; i < sharding->tile_assignment_devices_size(); ++i) {
      auto device = sharding->tile_assignment_devices(i);
      auto shard_shape =
          GetTPUInfeedLayout(hlo_sharding.TileShape(*shape, device));
      if (i == 0) {
        *shape->mutable_layout() = shard_shape.layout();
      }
      if (xla::ShapeUtil::ElementsIn(shard_shape) == 0) {
        // Shapes with 0 dimensions may be assigned with a different layout, but
        // it doesn't matter since we're not sending any data.
        continue;
      }
      if (!xla::LayoutUtil::Equal(shard_shape.layout(), shape->layout())) {
        return xla::Unimplemented(
            "Sharded infeed with non-uniform layouts is not supported. Try "
            "turning off the infeed layout optimization "
            "(--transpose_tpu_infeed=false) and report to XLA team.");
      }
    }
    return Status::OK();
  }
  *shape = GetTPUInfeedLayout(*shape);
  return Status::OK();
}

// TODO(pbar) Work out if we need to Infeed Tuples - if so then
// this op will need a way to provide a list of shapes
// since they can't be provided by the runtime JIT mechanism.
// (InfeedDequeue has no inputs!)
// Compare this op to tf.Queue operations which operate on N tensors.

// This TensorFlow op supports the XLA Infeed primitve.
class InfeedDequeueOp : public XlaOpKernel {
 public:
  explicit InfeedDequeueOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSxlaPSinfeed_opDTcc mht_2(mht_2_v, 265, "", "./tensorflow/core/tpu/kernels/xla/infeed_op.cc", "InfeedDequeueOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("shape", &shape_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtype", &dtype_));
    OP_REQUIRES_OK(ctx, TensorShapeToXLAShape(dtype_, shape_, &xla_shape_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSxlaPSinfeed_opDTcc mht_3(mht_3_v, 274, "", "./tensorflow/core/tpu/kernels/xla/infeed_op.cc", "Compile");

    xla::XlaBuilder* b = ctx->builder();
    OP_REQUIRES_OK(ctx, UpdateInfeedLayout(&xla_shape_, b->sharding()));
    ctx->SetOutput(0, xla::Infeed(b, xla_shape_));
  }

 private:
  TensorShape shape_;
  DataType dtype_;
  xla::Shape xla_shape_;

  TF_DISALLOW_COPY_AND_ASSIGN(InfeedDequeueOp);
};

REGISTER_XLA_OP(Name("InfeedDequeue"), InfeedDequeueOp);

// This TensorFlow op supports the XLA Infeed primitive for tuple types.
class InfeedDequeueTupleOp : public XlaOpKernel {
 public:
  explicit InfeedDequeueTupleOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSxlaPSinfeed_opDTcc mht_4(mht_4_v, 296, "", "./tensorflow/core/tpu/kernels/xla/infeed_op.cc", "InfeedDequeueTupleOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("shapes", &shapes_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("dtypes", &dtypes_));
    for (int i = 0; i < shapes_.size(); i++) {
      xla::Shape xla_shape;
      OP_REQUIRES_OK(ctx,
                     TensorShapeToXLAShape(dtypes_[i], shapes_[i], &xla_shape));
      xla_shapes_.push_back(xla_shape);
    }
  }

  ~InfeedDequeueTupleOp() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSxlaPSinfeed_opDTcc mht_5(mht_5_v, 310, "", "./tensorflow/core/tpu/kernels/xla/infeed_op.cc", "~InfeedDequeueTupleOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPSxlaPSinfeed_opDTcc mht_6(mht_6_v, 315, "", "./tensorflow/core/tpu/kernels/xla/infeed_op.cc", "Compile");

    xla::XlaBuilder* b = ctx->builder();
    for (int64_t i = 0; i < xla_shapes_.size(); ++i) {
      absl::optional<xla::OpSharding> sharding;
      if (b->sharding()) {
        sharding = b->sharding()->type() == xla::OpSharding::TUPLE
                       ? b->sharding()->tuple_shardings(i)
                       : b->sharding();
      }
      OP_REQUIRES_OK(ctx, UpdateInfeedLayout(&xla_shapes_[i], sharding));
    }
    tuple_shape_ = xla::ShapeUtil::MakeTupleShape(xla_shapes_);
    auto tuple = xla::Infeed(b, tuple_shape_);

    // Don't apply the infeed tuple sharding to the get-tuple-elements. They
    // need non-tuple shardings.
    xla::XlaScopedShardingAssignment clear_sharding(b, absl::nullopt);
    for (int i = 0; i < shapes_.size(); ++i) {
      ctx->SetOutput(i, xla::GetTupleElement(tuple, i));
    }
  }

 private:
  std::vector<TensorShape> shapes_;
  DataTypeVector dtypes_;
  std::vector<xla::Shape> xla_shapes_;
  xla::Shape tuple_shape_;

  TF_DISALLOW_COPY_AND_ASSIGN(InfeedDequeueTupleOp);
};

REGISTER_XLA_OP(Name("InfeedDequeueTuple"), InfeedDequeueTupleOp);

}  // anonymous namespace
}  // namespace tensorflow
