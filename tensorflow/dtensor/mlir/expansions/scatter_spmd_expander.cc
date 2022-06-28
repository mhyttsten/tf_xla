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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSscatter_spmd_expanderDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSscatter_spmd_expanderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSscatter_spmd_expanderDTcc() {
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

#include "tensorflow/dtensor/mlir/expansions/scatter_spmd_expander.h"

#include <string>

#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {
namespace {

StatusOr<Layout> GetOutputLayout(const absl::optional<Layout>& tensor_layout,
                                 int tensor_rank,
                                 const absl::optional<Layout>& updates_layout,
                                 int updates_rank, const Mesh& mesh) {
  // The first tensor_rank - update_rank dimensions of the output should be set
  // to replicated. The remainder are set from tensor_layout and updates_layout
  // with tensor_layout taking priority, as it is generally larger than updates
  // (as unsharding updates is faster).
  std::vector<ShardingSpec> output_specs(tensor_rank);

  // The number of dimensions at the start of the tensor input that are used
  // for the index, also the size of the second dimension of the indices tensor.
  const int index_dimensions = tensor_rank - (updates_rank - 1);

  for (int i = 0; i < tensor_rank; ++i)
    output_specs[i].set_sharding_spec(Layout::kUnshardedDim);

  absl::flat_hash_set<std::string> used_mesh_dims;

  if (tensor_layout) {
    for (int i = index_dimensions; i < tensor_rank; ++i) {
      output_specs[i] = tensor_layout->dim(i);
      if (Layout::IsShardedSpec(output_specs[i]))
        used_mesh_dims.emplace(output_specs[i].sharding_spec());
    }
  }

  if (updates_layout) {
    for (int i = index_dimensions; i < tensor_rank; ++i) {
      const ShardingSpec& update_spec =
          updates_layout->dim(i - index_dimensions + 1);

      if (Layout::IsUnshardedSpec(output_specs[i]) &&
          Layout::IsShardedSpec(update_spec) &&
          !used_mesh_dims.contains(update_spec.sharding_spec()))
        output_specs[i] = update_spec;
    }
  }

  return Layout::GetLayout(output_specs, mesh);
}

template <typename OpType>
StatusOr<mlir::Operation*> TensorScatterOpExpand(mlir::Operation* op) {
  auto scatter_op = llvm::cast<OpType>(op);
  TF_ASSIGN_OR_RETURN(auto tensor_layout,
                      ExtractLayoutFromOperand(scatter_op.tensor()));
  TF_ASSIGN_OR_RETURN(auto indices_layout,
                      ExtractLayoutFromOperand(scatter_op.indices()));
  TF_ASSIGN_OR_RETURN(auto updates_layout,
                      ExtractLayoutFromOperand(scatter_op.updates()));
  TF_ASSIGN_OR_RETURN(auto output_layout,
                      ExtractSingleLayoutFromOp(scatter_op));

  const int tensor_rank = ValueRank(scatter_op.tensor());
  const int updates_rank = ValueRank(scatter_op.updates());

  if (tensor_rank == -1 || updates_rank == -1)
    return errors::InvalidArgument("all inputs must have valid rank.");

  // Get the global shape of all inputs as we need them for the Relayout
  // operations.
  TF_ASSIGN_OR_RETURN(
      llvm::ArrayRef<int64_t> tensor_shape,
      GetGlobalShapeOfValueFromDTensorLayout(scatter_op.tensor()));
  TF_ASSIGN_OR_RETURN(
      llvm::ArrayRef<int64_t> indices_shape,
      GetGlobalShapeOfValueFromDTensorLayout(scatter_op.indices()));
  TF_ASSIGN_OR_RETURN(
      llvm::ArrayRef<int64_t> updates_shape,
      GetGlobalShapeOfValueFromDTensorLayout(scatter_op.updates()));

  // Start by relaying out the inputs. Indices should replicated.
  TF_ASSIGN_OR_RETURN(
      mlir::Value new_indices,
      EmitRelayout(scatter_op.indices(), *indices_layout,
                   Layout::ReplicatedOnMesh(indices_layout->mesh(),
                                            indices_shape.size())));

  // Create intermediate layouts for tensors and updates. Since the layout of
  // tensor and the output of the local tensor-scatter are the same we can reuse
  // GetOutputLayout.
  // If the true output layout is even more sharded, we could forward those
  // shardings here for even better performance.
  TF_ASSIGN_OR_RETURN(
      Layout pre_output_layout,
      GetOutputLayout(tensor_layout, tensor_rank, updates_layout, updates_rank,
                      tensor_layout->mesh()));

  std::vector<ShardingSpec> updates_specs(updates_rank);
  updates_specs[0].set_sharding_spec(Layout::kUnshardedDim);

  const int index_dimensions = tensor_rank - (updates_rank - 1);

  for (int i = 0; i < updates_rank - 1; ++i)
    updates_specs[i + 1] = pre_output_layout.dim(index_dimensions + i);

  TF_ASSIGN_OR_RETURN(Layout new_updates_layout,
                      Layout::GetLayout(updates_specs, updates_layout->mesh()));
  TF_ASSIGN_OR_RETURN(
      mlir::Value new_tensor,
      EmitRelayout(scatter_op.tensor(), *tensor_layout, pre_output_layout));
  TF_ASSIGN_OR_RETURN(
      mlir::Value new_updates,
      EmitRelayout(scatter_op.updates(), *updates_layout, new_updates_layout));

  mlir::OpBuilder builder(op);
  OpType new_scatter = builder.create<OpType>(
      op->getLoc(), new_tensor.getType(), new_tensor, new_indices, new_updates);

  TF_ASSIGN_OR_RETURN(
      mlir::Value new_output,
      EmitRelayout(new_scatter.output(), pre_output_layout, *output_layout));

  op->getResult(0).replaceAllUsesWith(new_output);
  op->erase();

  return new_output.getDefiningOp();
}

template <typename OpType>
StatusOr<llvm::DenseMap<int, Layout>> TensorScatterOpComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));
  auto scatter_op = llvm::cast<OpType>(op);

  const int tensor_rank = ValueRank(scatter_op.tensor());
  const int updates_rank = ValueRank(scatter_op.updates());
  if (tensor_rank == -1 || updates_rank == -1)
    return errors::InvalidArgument("all inputs must have valid rank.");

  absl::optional<Layout> tensor_layout;
  if (input_layouts.find(0) != input_layouts.end())
    tensor_layout.emplace(input_layouts.lookup(0));
  absl::optional<Layout> updates_layout;
  if (input_layouts.find(2) != input_layouts.end())
    updates_layout.emplace(input_layouts.lookup(2));

  if (tensor_layout || updates_layout) {
    TF_ASSIGN_OR_RETURN(const Layout output_layout,
                        GetOutputLayout(tensor_layout, tensor_rank,
                                        updates_layout, updates_rank, mesh));
    return llvm::DenseMap<int, Layout>({{0, output_layout}});
  }

  return llvm::DenseMap<int, Layout>();
}

template <typename OpType>
StatusOr<llvm::DenseMap<int, Layout>> TensorScatterOpComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));
  auto scatter_op = llvm::cast<OpType>(op);

  const int tensor_rank = ValueRank(scatter_op.tensor());
  const int indices_rank = ValueRank(scatter_op.indices());
  const int updates_rank = ValueRank(scatter_op.updates());
  if (tensor_rank == -1 || indices_rank == -1 || updates_rank == -1)
    return errors::InvalidArgument("all inputs must have valid rank.");

  // The number of dimensions at the start of the tensor input that are used
  // for the index, also the size of the second dimension of the indices tensor.
  const int index_dimensions = tensor_rank - (updates_rank - 1);

  llvm::DenseMap<int, Layout> input_layouts(scatter_op.getNumOperands());

  // Always set indices layout to replicated.
  const Layout indices_layout = Layout::ReplicatedOnMesh(mesh, indices_rank);
  input_layouts[1] = indices_layout;

  if (output_layouts.find(0) != output_layouts.end()) {
    const Layout output_layout = output_layouts.lookup(0);

    std::vector<std::string> tensor_sharding_specs(tensor_rank);
    std::vector<std::string> updates_sharding_specs(updates_rank);

    for (int i = 0; i < index_dimensions; ++i)
      tensor_sharding_specs[i] = Layout::kUnshardedDim;
    updates_sharding_specs[0] = Layout::kUnshardedDim;

    for (int i = index_dimensions; i < tensor_rank; ++i) {
      tensor_sharding_specs[i] = output_layout.sharding_spec(i);
      updates_sharding_specs[i - index_dimensions + 1] =
          output_layout.sharding_spec(i);
    }

    TF_ASSIGN_OR_RETURN(const Layout tensor_layout,
                        Layout::GetLayout(tensor_sharding_specs, mesh));
    TF_ASSIGN_OR_RETURN(const Layout updates_layout,
                        Layout::GetLayout(updates_sharding_specs, mesh));
    input_layouts[0] = tensor_layout;
    input_layouts[2] = updates_layout;
  }

  return input_layouts;
}

}  // namespace

StatusOr<mlir::Operation*> TensorScatterOpSPMDExpander::ExpandOp(
    mlir::Operation* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSscatter_spmd_expanderDTcc mht_0(mht_0_v, 402, "", "./tensorflow/dtensor/mlir/expansions/scatter_spmd_expander.cc", "TensorScatterOpSPMDExpander::ExpandOp");

  if (llvm::isa<mlir::TF::TensorScatterUpdateOp>(op)) {
    return TensorScatterOpExpand<mlir::TF::TensorScatterUpdateOp>(op);
  }
  if (llvm::isa<mlir::TF::TensorScatterAddOp>(op)) {
    return TensorScatterOpExpand<mlir::TF::TensorScatterAddOp>(op);
  }
  return errors::Unimplemented(absl::StrCat(
      "SPMD expansion for op : ", OpName(op), " is not implemented"));
}

StatusOr<llvm::DenseMap<int, Layout>>
TensorScatterOpSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  if (llvm::isa<mlir::TF::TensorScatterUpdateOp>(op)) {
    return TensorScatterOpComputeLayoutForward<mlir::TF::TensorScatterUpdateOp>(
        op, input_layouts);
  }
  if (llvm::isa<mlir::TF::TensorScatterAddOp>(op)) {
    return TensorScatterOpComputeLayoutForward<mlir::TF::TensorScatterAddOp>(
        op, input_layouts);
  }
  return errors::Unimplemented(absl::StrCat(
      "Layout propagation for op : ", OpName(op), " is not implemented"));
}

StatusOr<llvm::DenseMap<int, Layout>>
TensorScatterOpSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  if (llvm::isa<mlir::TF::TensorScatterUpdateOp>(op)) {
    return TensorScatterOpComputeLayoutBackward<
        mlir::TF::TensorScatterUpdateOp>(op, output_layouts);
  }
  if (llvm::isa<mlir::TF::TensorScatterAddOp>(op)) {
    return TensorScatterOpComputeLayoutBackward<mlir::TF::TensorScatterAddOp>(
        op, output_layouts);
  }
  return errors::Unimplemented(absl::StrCat(
      "Layout propagation for op : ", OpName(op), " is not implemented"));
}

}  // namespace dtensor
}  // namespace tensorflow
