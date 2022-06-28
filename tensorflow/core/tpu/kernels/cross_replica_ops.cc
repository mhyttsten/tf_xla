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
class MHTracer_DTPStensorflowPScorePStpuPSkernelsPScross_replica_opsDTcc {
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
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPScross_replica_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStpuPSkernelsPScross_replica_opsDTcc() {
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
#include <utility>
#include <vector>

#include "tensorflow/compiler/tf2xla/xla_op_kernel.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

// Convert 1D group_assignment into 2D replica_groups.
std::vector<xla::ReplicaGroup> Convert(
    const std::vector<int64_t>& group_assignment,
    const TensorShape& group_assignment_shape) {
  VLOG(1) << "group_assignment size: " << group_assignment.size();
  VLOG(1) << "group_assignment_shape: " << group_assignment_shape.DebugString();

  std::vector<xla::ReplicaGroup> replica_groups;
  const int64_t num_groups = group_assignment_shape.dim_size(0);
  const int64_t num_replica_per_group = group_assignment_shape.dim_size(1);

  replica_groups.reserve(num_groups);
  for (int64_t g = 0; g < num_groups; ++g) {
    xla::ReplicaGroup replica_group;

    for (int64_t i = 0; i < num_replica_per_group; ++i) {
      int64_t replica = group_assignment[num_replica_per_group * g + i];
      replica_group.add_replica_ids(replica);
    }
    replica_groups.push_back(replica_group);
  }
  return replica_groups;
}

class CrossReplicaSumOp : public XlaOpKernel {
 public:
  explicit CrossReplicaSumOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPScross_replica_opsDTcc mht_0(mht_0_v, 222, "", "./tensorflow/core/tpu/kernels/cross_replica_ops.cc", "CrossReplicaSumOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPScross_replica_opsDTcc mht_1(mht_1_v, 227, "", "./tensorflow/core/tpu/kernels/cross_replica_ops.cc", "Compile");

    std::vector<int64_t> flattened_group_assignment;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputReshapedToIntVector(
                            1, &flattened_group_assignment));
    std::vector<xla::ReplicaGroup> replica_groups =
        Convert(flattened_group_assignment, ctx->InputShape(1));
    ctx->SetOutput(0, xla::CrossReplicaSum(ctx->Input(0), replica_groups));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CrossReplicaSumOp);
};

class AllToAllOp : public XlaOpKernel {
 public:
  explicit AllToAllOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPScross_replica_opsDTcc mht_2(mht_2_v, 245, "", "./tensorflow/core/tpu/kernels/cross_replica_ops.cc", "AllToAllOp");

    OP_REQUIRES_OK(ctx, ctx->GetAttr("split_dimension", &split_dimension_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("concat_dimension", &concat_dimension_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("split_count", &split_count_));
  }

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPScross_replica_opsDTcc mht_3(mht_3_v, 254, "", "./tensorflow/core/tpu/kernels/cross_replica_ops.cc", "Compile");

    std::vector<int64_t> flattened_group_assignment;
    OP_REQUIRES_OK(ctx, ctx->ConstantInputReshapedToIntVector(
                            1, &flattened_group_assignment));

    std::vector<xla::ReplicaGroup> replica_groups =
        Convert(flattened_group_assignment, ctx->InputShape(1));
    ctx->SetOutput(
        0, xla::AllToAll(ctx->Input(0), split_dimension_, concat_dimension_,
                         split_count_, replica_groups));
  }

 private:
  int64_t split_dimension_;
  int64_t concat_dimension_;
  int64_t split_count_;

  TF_DISALLOW_COPY_AND_ASSIGN(AllToAllOp);
};

class CollectivePermuteOp : public XlaOpKernel {
 public:
  explicit CollectivePermuteOp(OpKernelConstruction* ctx) : XlaOpKernel(ctx) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPScross_replica_opsDTcc mht_4(mht_4_v, 279, "", "./tensorflow/core/tpu/kernels/cross_replica_ops.cc", "CollectivePermuteOp");
}

  void Compile(XlaOpKernelContext* ctx) override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStpuPSkernelsPScross_replica_opsDTcc mht_5(mht_5_v, 284, "", "./tensorflow/core/tpu/kernels/cross_replica_ops.cc", "Compile");

    const TensorShape source_target_shape = ctx->InputShape(1);
    OP_REQUIRES(
        ctx,
        source_target_shape.dims() == 2 && source_target_shape.dim_size(1) == 2,
        errors::InvalidArgument(
            "CollectivePermuteOp requires source_target_pair's shape to"
            " [num_pairs, 2]. Get ",
            source_target_shape));

    xla::Literal source_target_literal;
    OP_REQUIRES_OK(ctx,
                   ctx->ConstantInputAsInt64Literal(1, &source_target_literal));
    const int num_pairs = source_target_shape.dim_size(0);
    std::vector<std::pair<int64_t, int64_t>> source_target_pairs(num_pairs);
    for (int i = 0; i < num_pairs; ++i) {
      source_target_pairs[i] = {source_target_literal.Get<int64_t>({i, 0}),
                                source_target_literal.Get<int64_t>({i, 1})};
    }
    ctx->SetOutput(0,
                   xla::CollectivePermute(ctx->Input(0), source_target_pairs));
  }

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(CollectivePermuteOp);
};

REGISTER_XLA_OP(Name("AllToAll").CompileTimeConstantInput("group_assignment"),
                AllToAllOp);
REGISTER_XLA_OP(Name("CollectivePermute")
                    .TypeConstraint("T", {DT_DOUBLE, DT_FLOAT, DT_BFLOAT16,
                                          DT_INT32, DT_COMPLEX64})
                    .CompileTimeConstantInput("source_target_pairs"),
                CollectivePermuteOp);
REGISTER_XLA_OP(
    Name("CrossReplicaSum").CompileTimeConstantInput("group_assignment"),
    CrossReplicaSumOp);

}  // anonymous namespace
}  // namespace tensorflow
