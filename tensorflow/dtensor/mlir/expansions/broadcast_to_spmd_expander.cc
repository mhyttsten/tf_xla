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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSbroadcast_to_spmd_expanderDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSbroadcast_to_spmd_expanderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSbroadcast_to_spmd_expanderDTcc() {
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

/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/dtensor/mlir/expansions/broadcast_to_spmd_expander.h"

#include <string>
#include <utility>

#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"
#include "tensorflow/dtensor/proto/layout.pb.h"

namespace tensorflow {
namespace dtensor {

StatusOr<mlir::Operation*> BroadcastToSPMDExpander::ExpandOp(
    mlir::Operation* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSbroadcast_to_spmd_expanderDTcc mht_0(mht_0_v, 207, "", "./tensorflow/dtensor/mlir/expansions/broadcast_to_spmd_expander.cc", "BroadcastToSPMDExpander::ExpandOp");

  auto broadcast_op = llvm::cast<mlir::TF::BroadcastToOp>(op);
  TF_ASSIGN_OR_RETURN(const Layout shape_layout,
                      ExtractRequiredLayoutFromOperand(broadcast_op.shape()));
  if (!shape_layout.IsFullyReplicated()) {
    return errors::InvalidArgument(
        "Error during BroadcastOp SPMD Expansion. Shape input of broadcast op "
        "must be fully replicated.");
  }

  TF_ASSIGN_OR_RETURN(const Layout input_layout,
                      ExtractRequiredLayoutFromOperand(broadcast_op.input()));
  TF_ASSIGN_OR_RETURN(const Layout output_layout,
                      ExtractRequiredSingleLayoutFromOp(broadcast_op));

  TF_ASSIGN_OR_RETURN(
      llvm::ArrayRef<int64_t> input_global_size,
      GetGlobalShapeOfValueFromDTensorLayout(broadcast_op.input()));

  llvm::SmallVector<int64_t, 4> broadcast_to_shape;
  TF_RETURN_IF_ERROR(ExtractConstVectorFromValue(
      GetForwardedDTensorLayoutInput(broadcast_op.shape()),
      &broadcast_to_shape));

  // Input to BroadcastTo op requires all to all if non-broadcasted-dimensions
  // are not same.
  const int broadcasted_dimensions = output_layout.rank() - input_layout.rank();
  bool requires_all_to_all = false;
  const auto output_num_shards = output_layout.num_shards();
  for (int i = 0; i < input_layout.rank(); ++i) {
    const int output_dim_index = i + broadcasted_dimensions;
    const std::string& output_layout_dim =
        output_layout.sharding_spec(output_dim_index);
    if (input_global_size[i] > 1 &&
        input_layout.sharding_spec(i) != output_layout_dim) {
      requires_all_to_all = true;
    }
    if (output_layout_dim != Layout::kUnshardedDim) {
      broadcast_to_shape[output_dim_index] /=
          output_num_shards[output_dim_index];
    }
  }

  // Insert all-to-all operations just before Broadcast op to ensure all inputs
  // in correct local values.
  mlir::OpBuilder builder(op);
  mlir::Value input_data = broadcast_op.input();
  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));
  const Layout all_to_all_input_layout =
      Layout::ReplicatedOnMesh(mesh, input_layout.rank());

  if (requires_all_to_all) {
    TF_ASSIGN_OR_RETURN(auto input_data,
                        EmitAllGather(builder, input_data, input_layout,
                                      all_to_all_input_layout));
    op->setOperand(0, input_data);
  } else {
    // When all-to-all is not needed, output of BroadcastTo operation may be
    // sharded. In that case, we must ensure that `shape` input of BroadcastTo
    // op has correct local sharded shape.
    // Note that we include the sharding on the first
    for (int i = 0; i < broadcasted_dimensions; ++i)
      if (output_layout.sharding_spec(i) != Layout::kUnshardedDim)
        broadcast_to_shape[i] /= output_num_shards[i];
    mlir::Value new_broadcast_to_shape =
        Int64Const(builder, op->getLoc(), broadcast_to_shape);
    op->setOperand(1, new_broadcast_to_shape);
  }

  op = InferSPMDExpandedLocalShape(op);
  if (!requires_all_to_all) return op;

  // If we all-to-all'ed, we may need to split after the local BroadcastTo op
  // has been created in graph.
  llvm::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;
  builder.setInsertionPointAfter(op);
  TF_ASSIGN_OR_RETURN(
      auto final_output,
      EmitAllScatter(builder, op->getOpResult(0),
                     all_to_all_input_layout.LeftPad(output_layout.rank()),
                     output_layout, &newly_created_ops));
  op->getOpResult(0).replaceAllUsesExcept(final_output, newly_created_ops);
  return final_output.getDefiningOp();
}

StatusOr<llvm::DenseMap<int, Layout>>
BroadcastToSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  // If we do not have an input layout then do not infer an output layout.
  if (input_layouts.find(0) == input_layouts.end())
    return llvm::DenseMap<int, Layout>();

  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));

  auto broadcast_op = llvm::cast<mlir::TF::BroadcastToOp>(op);
  TF_ASSIGN_OR_RETURN(
      const auto broadcasted_output_shape,
      GetShapeOfValue(broadcast_op.output(), /*fail_on_dynamic=*/true));
  TF_ASSIGN_OR_RETURN(
      const auto input_shape,
      GetShapeOfValue(broadcast_op.input(), /*fail_on_dynamic=*/true));

  // Broadcasting works from trailing dimensions and dimensions are broadcasted
  // in forward direction.
  const int output_shape_rank = broadcasted_output_shape.size();
  const int input_shape_rank = input_shape.size();
  const int broadcasted_dimensions = output_shape_rank - input_shape_rank;

  if (broadcasted_dimensions < 0)
    return errors::FailedPrecondition("Broadcasted dimension was less than 0.");

  Layout input_layout = input_layouts.lookup(0);

  std::vector<std::string> layout_sharding;
  for (int i = 0; i < output_shape_rank; ++i) {
    if (i < broadcasted_dimensions) {
      layout_sharding.push_back(Layout::kUnshardedDim);
    } else {
      layout_sharding.push_back(
          input_layout.sharding_spec(i - broadcasted_dimensions));
    }
  }
  TF_ASSIGN_OR_RETURN(Layout inferred_output_layout,
                      Layout::GetLayout(layout_sharding, mesh));
  return llvm::DenseMap<int, Layout>({{0, inferred_output_layout}});
}

StatusOr<llvm::DenseMap<int, Layout>>
BroadcastToSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));

  // If output layout is not set, then we can only infer the `shape` input
  // which should always be replicated.
  if (output_layouts.find(0) == output_layouts.end())
    return llvm::DenseMap<int, Layout>(
        {{1, Layout::ReplicatedOnMesh(mesh, ValueRank(op->getOperand(1)))}});

  auto output_layout = output_layouts.lookup(0);

  auto broadcast_op = llvm::cast<mlir::TF::BroadcastToOp>(op);
  TF_ASSIGN_OR_RETURN(
      const auto broadcasted_output_shape,
      GetShapeOfValue(broadcast_op.output(), /*fail_on_dynamic=*/true));
  TF_ASSIGN_OR_RETURN(
      const auto input_shape,
      GetShapeOfValue(broadcast_op.input(), /*fail_on_dynamic=*/true));

  // Broadcasting works from trailing dimensions and dimensions are broadcasted
  // in forward direction.
  const int output_shape_rank = broadcasted_output_shape.size();
  const int input_shape_rank = input_shape.size();
  const int broadcasted_dimensions = output_shape_rank - input_shape_rank;

  LayoutProto layout_proto;
  *layout_proto.mutable_mesh_config() = mesh.ToProto();
  for (int i = 0; i < input_shape_rank; ++i) {
    if (input_shape[i] == 1) {
      layout_proto.add_sharding_specs()->set_sharding_spec(
          Layout::kUnshardedDim);
    } else {
      layout_proto.add_sharding_specs()->set_sharding_spec(
          output_layout.sharding_spec(i + broadcasted_dimensions));
    }
  }
  TF_ASSIGN_OR_RETURN(Layout inferred_operand_layout,
                      Layout::FromProto(layout_proto));
  // `shape` input of BroadcastTo is always set as replicated.
  return llvm::DenseMap<int, Layout>(
      {{0, inferred_operand_layout},
       {1, Layout::ReplicatedOnMesh(mesh, ValueRank(op->getOperand(1)))}});
}

}  // namespace dtensor
}  // namespace tensorflow
