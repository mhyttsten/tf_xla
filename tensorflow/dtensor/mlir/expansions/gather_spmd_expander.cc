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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSgather_spmd_expanderDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSgather_spmd_expanderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSgather_spmd_expanderDTcc() {
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

#include "tensorflow/dtensor/mlir/expansions/gather_spmd_expander.h"

#include <string>
#include <vector>

#include "absl/types/optional.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {

StatusOr<mlir::Operation*> GatherV2SPMDExpander::ExpandOp(mlir::Operation* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSgather_spmd_expanderDTcc mht_0(mht_0_v, 204, "", "./tensorflow/dtensor/mlir/expansions/gather_spmd_expander.cc", "GatherV2SPMDExpander::ExpandOp");

  auto gather_op = llvm::cast<mlir::TF::GatherV2Op>(op);
  TF_ASSIGN_OR_RETURN(int64_t axis, ExtractConstIntFromValue(gather_op.axis()));
  TF_ASSIGN_OR_RETURN(auto params_layout,
                      ExtractLayoutFromOperand(gather_op.params()));
  TF_ASSIGN_OR_RETURN(auto indices_layout,
                      ExtractLayoutFromOperand(gather_op.indices()));
  TF_ASSIGN_OR_RETURN(auto output_layout, ExtractSingleLayoutFromOp(gather_op));

  const auto params_rank = ValueRank(gather_op.params());
  const auto indices_rank = ValueRank(gather_op.indices());
  if (params_rank == -1)
    return errors::InvalidArgument("missing rank for params input");
  if (indices_rank == -1)
    return errors::InvalidArgument("missing rank for indices input");

  // Handle the case of negative axis.
  if (axis < 0) axis += params_rank;

  int batch_dims = gather_op.batch_dims();

  auto params = gather_op.params();
  auto indices = gather_op.indices();

  mlir::OpBuilder builder(op);

  // Step 1: If the params are sharded on dim axis, an unconditional all-concat
  // is generated. Alternatively, we could do: all-concating indices, followed
  // by tf.Gather + slicing with correct masks.
  //
  // Currently we only support the case that the output layout matching the
  // params layout for all non-axis dim. Other cases needs either a slicing or
  // all-concat, which can be added later.
  {
    LayoutProto tgt_params_layout;
    *tgt_params_layout.mutable_mesh_config() = params_layout->mesh().ToProto();
    // check the first half
    for (int i = 0; i < axis; ++i) {
      const auto dim_name = params_layout->sharding_spec(i);
      if (dim_name != output_layout->sharding_spec(i)) {
        return errors::InvalidArgument(
            llvm::formatv(
                "input and output layout do not agree on non-axis dim {0}. "
                "\n  params: {1}\n  output: {2}, axis: {3}",
                i, params_layout->ToString(), output_layout->ToString(), axis)
                .str());
      }
      tgt_params_layout.add_sharding_specs()->set_sharding_spec(dim_name);
    }
    // Set replicated for `axis` dim.
    tgt_params_layout.add_sharding_specs()->set_sharding_spec(
        Layout::kUnshardedDim);
    // Check the second half
    for (int i = axis + 1; i < params_rank; ++i) {
      auto dim_name = params_layout->sharding_spec(i);
      // To align the param dim with output, we can think we insert indices_rank
      // - batch_dims dims from indices and remove one from param (axis), so
      // the shifting is indices_rank - batch_dims - 1.
      if (dim_name !=
          output_layout->sharding_spec(i + indices_rank - batch_dims - 1)) {
        return errors::InvalidArgument(
            llvm::formatv(
                "input and output layout do not agree on non-axis dim {0}. "
                "\n  params: {1}\n  output: {2}, axis: {3}",
                i, params_layout->ToString(), output_layout->ToString(), axis)
                .str());
      }
      tgt_params_layout.add_sharding_specs()->set_sharding_spec(dim_name);
    }

    if (!Layout::IsUnshardedDimension(params_layout->sharding_spec(axis))) {
      TF_ASSIGN_OR_RETURN(
          params,
          EmitAllGather(builder, params, *params_layout,
                        Layout::FromProto(tgt_params_layout).ValueOrDie()));
    }
  }

  // Step 2: Check the output layout. If it requires all-relayouting indices.
  // Do it.
  //
  // Indices shape is not big typically. Relayouting is expected to be cheap.
  {
    bool indices_relayout_needed = false;
    LayoutProto tgt_indices_layout;
    *tgt_indices_layout.mutable_mesh_config() = output_layout->mesh().ToProto();
    for (int i = 0; i < indices_rank; ++i) {
      int index_in_output;
      int index_in_indices;
      if (i < batch_dims) {
        // For dim within batch_dims, indices dim is aligning at the same index
        // as output.
        index_in_output = i;
        index_in_indices = i;
      } else {
        // For dim after batch_dims, we can remove batch_dims from outputs and
        // indices first, i.e., (i - batch_dims), add axis back, i.e., axis -
        // batch_dims, and then put batch_dims back, so the target position in
        // output is
        //
        //   i - batch_dims + axis - batch_dims + batch_dims
        //
        // which is as follows:
        index_in_output = i + axis - batch_dims;
        index_in_indices = i;
      }
      tgt_indices_layout.add_sharding_specs()->set_sharding_spec(
          output_layout->sharding_spec(index_in_output));

      if (output_layout->sharding_spec(index_in_output) !=
          indices_layout->sharding_spec(index_in_indices)) {
        indices_relayout_needed = true;
      }
    }

    if (indices_relayout_needed) {
      TF_ASSIGN_OR_RETURN(
          indices,
          EmitRelayout(indices, *indices_layout,
                       Layout::FromProto(tgt_indices_layout).ValueOrDie()));
    }
  }

  auto new_gather = builder.create<mlir::TF::GatherV2Op>(
      gather_op.getLoc(), gather_op.getResult().getType(), params, indices,
      gather_op.axis(), gather_op.batch_dims());
  op->getResult(0).replaceAllUsesWith(new_gather.output());
  op->erase();

  return InferSPMDExpandedLocalShape(new_gather);
}

StatusOr<llvm::DenseMap<int, Layout>>
GatherV2SPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  auto gather_op = llvm::cast<mlir::TF::GatherV2Op>(op);
  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));

  TF_ASSIGN_OR_RETURN(int64_t axis, ExtractConstIntFromValue(gather_op.axis()));
  int batch_dims = gather_op.batch_dims();

  absl::optional<Layout> params_layout;
  if (input_layouts.find(0) != input_layouts.end())
    params_layout.emplace(input_layouts.lookup(0));
  absl::optional<Layout> indices_layout;
  if (input_layouts.find(1) != input_layouts.end())
    indices_layout.emplace(input_layouts.lookup(1));

  const int params_rank = ValueRank(gather_op.params());
  const int indices_rank = ValueRank(gather_op.indices());
  if (params_rank == -1)
    return errors::InvalidArgument("missing rank for params input");
  if (indices_rank == -1)
    return errors::InvalidArgument("missing rank for indices input");

  // Handle the case of negative axis.
  if (axis < 0) axis += params_rank;

  if (params_layout || indices_layout) {
    std::vector<std::string> output_layout_specs;

    // Get a list of mesh dims that params uses, other than the dim for axis.
    llvm::DenseSet<llvm::StringRef> params_mesh_dims;
    if (params_layout) {
      for (int i = 0; i < params_rank; ++i)
        if (i != axis &&
            !Layout::IsUnshardedDimension(params_layout->sharding_spec(i)))
          params_mesh_dims.insert(params_layout->sharding_spec(i));
    }

    auto add_mesh_dim_if = [&](const absl::optional<Layout>& input_layout,
                               int64 dim, bool indices = false) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSgather_spmd_expanderDTcc mht_1(mht_1_v, 378, "", "./tensorflow/dtensor/mlir/expansions/gather_spmd_expander.cc", "lambda");

      // Only add the mesh dimension to the output_layout if 1) the input layout
      // exists and 2) when the input is indices and the params dims don't
      // contain the mesh dim we are adding (to avoid two different tensor dims
      // being sharded over the same mesh dim).
      // Note that params->sharding_spec(axis) is specifically excluded from the
      // params_mesh_dims during construction above. This means that if we are
      // processing the indices layout and it contains
      // params->sharding_spec(axis) we will still add that sharding_spec to the
      // output layout.
      if (input_layout && (!indices || !params_mesh_dims.contains(
                                           input_layout->sharding_spec(dim))))
        output_layout_specs.push_back(input_layout->sharding_spec(dim));
      else
        output_layout_specs.push_back(Layout::kUnshardedDim);
    };

    for (int i = 0; i < axis; ++i) add_mesh_dim_if(params_layout, i);
    for (int i = batch_dims; i < indices_rank; ++i)
      add_mesh_dim_if(indices_layout, i, /*indices=*/true);
    for (int i = axis + 1; i < params_rank; ++i)
      add_mesh_dim_if(params_layout, i);

    TF_ASSIGN_OR_RETURN(const Layout output_layout,
                        Layout::GetLayout(output_layout_specs, mesh));
    return llvm::DenseMap<int, Layout>({{0, output_layout}});
  }

  return llvm::DenseMap<int, Layout>();
}

StatusOr<llvm::DenseMap<int, Layout>>
GatherV2SPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  auto gather_op = llvm::cast<mlir::TF::GatherV2Op>(op);
  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));

  llvm::DenseMap<int, Layout> input_layouts(gather_op.getNumOperands());
  // axis is const
  input_layouts[2] = Layout::ReplicatedOnMesh(mesh, /*rank=*/0);

  if (output_layouts.find(0) != output_layouts.end()) {
    // This will always exist since there is only one output.
    const Layout output_layout = output_layouts.lookup(0);

    TF_ASSIGN_OR_RETURN(int64_t axis,
                        ExtractConstIntFromValue(gather_op.axis()));
    int batch_dims = gather_op.batch_dims();

    const int params_rank = ValueRank(gather_op.params());
    const int indices_rank = ValueRank(gather_op.indices());
    if (params_rank == -1)
      return errors::InvalidArgument("missing rank for params input");
    if (indices_rank == -1)
      return errors::InvalidArgument("missing rank for indices input");

    // Handle the case of negative axis.
    if (axis < 0) axis += params_rank;

    std::vector<std::string> params_layout_specs;
    std::vector<std::string> indices_layout_specs;
    params_layout_specs.reserve(params_rank);
    indices_layout_specs.reserve(indices_rank);

    // Extract the params layout. We will request that axis is replicated as
    // that gives the least issues with spmd expansion.
    // E.g. If we had axis = 1 and parmas layout [p0 p1 p2 p3]
    // input layout [i0 i1] then the output layout would have been
    // [p0 i0 i1 p2 p3] so to go backwards, we extract the ranges [0, axis)
    // and [axis + indices.rank(), output.rank()) for params (with a replicated
    // dim for the missing dimension inbetween). Indices layout is based on the
    // range [axis, axis+indices)/
    for (int i = 0; i < axis; ++i)
      params_layout_specs.push_back(output_layout.sharding_spec(i));
    params_layout_specs.push_back(Layout::kUnshardedDim);
    for (int i = axis + indices_rank - batch_dims; i < output_layout.rank();
         ++i)
      params_layout_specs.push_back(output_layout.sharding_spec(i));

    // Extract the indices layout.
    for (int i = 0; i < batch_dims; ++i)
      indices_layout_specs.push_back(output_layout.sharding_spec(i));
    for (int i = axis; i < axis + indices_rank - batch_dims; ++i)
      indices_layout_specs.push_back(output_layout.sharding_spec(i));

    TF_ASSIGN_OR_RETURN(const Layout params_layout,
                        Layout::GetLayout(params_layout_specs, mesh));
    TF_ASSIGN_OR_RETURN(const Layout indices_layout,
                        Layout::GetLayout(indices_layout_specs, mesh));
    input_layouts[0] = params_layout;
    input_layouts[1] = indices_layout;
  }

  return input_layouts;
}

namespace {

StatusOr<Layout> GatherNdGetOutputLayoutFromInput(
    const absl::optional<Layout>& params_layout, int params_rank,
    const absl::optional<Layout>& indices_layout, int indices_rank,
    int index_dimensions, const Mesh& mesh) {
  // The layout of the output should be the layout of the first rank-1
  // dimensions of the indices plus the layout of the last params.rank -
  // indices.dims[-1] dimensions of params. If one of the two is missing we
  // replace them with replicated.
  // If sharding dimension is used by both params and indices, the params
  // layout will be respected as generally params is larger than indices.
  std::vector<ShardingSpec> output_specs(params_rank - index_dimensions +
                                         indices_rank - 1);
  absl::flat_hash_set<std::string> used_dimensions;
  const int params_offset = -index_dimensions + indices_rank - 1;

  for (int i = index_dimensions; i < params_rank; ++i) {
    if (params_layout && Layout::IsShardedSpec(params_layout->dim(i))) {
      const ShardingSpec& params_spec = params_layout->dim(i);
      output_specs[i + params_offset] = params_spec;
      used_dimensions.emplace(params_spec.sharding_spec());
    } else {
      output_specs[i + params_offset].set_sharding_spec(Layout::kUnshardedDim);
    }
  }
  for (int i = 0; i < indices_rank - 1; ++i) {
    if (indices_layout && Layout::IsShardedSpec(indices_layout->dim(i)) &&
        !used_dimensions.contains(indices_layout->sharding_spec(i)))
      output_specs[i] = indices_layout->dim(i);
    else
      output_specs[i].set_sharding_spec(Layout::kUnshardedDim);
  }
  return Layout::GetLayout(output_specs, mesh);
}

Status GatherNdGetInputLayoutFromOutput(const Layout& output_layout,
                                        Layout* params_layout, int params_rank,
                                        Layout* indices_layout,
                                        int indices_rank, int index_dimensions,
                                        const Mesh& mesh) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSgather_spmd_expanderDTcc mht_2(mht_2_v, 517, "", "./tensorflow/dtensor/mlir/expansions/gather_spmd_expander.cc", "GatherNdGetInputLayoutFromOutput");

  // We copy the first indices_rank - 1 dimensions of the output layout to
  // indices_layout (with the last dimensions replicated) and the remaining
  // dimensions to params_layout (with the first index_dimensions dimensions
  // replicated).
  std::vector<ShardingSpec> params_specs(params_rank);
  std::vector<ShardingSpec> indices_specs(indices_rank);

  for (int i = 0; i < index_dimensions; ++i)
    params_specs[i].set_sharding_spec(Layout::kUnshardedDim);

  const int params_offset = -index_dimensions + indices_rank - 1;
  for (int i = index_dimensions; i < params_rank; ++i)
    params_specs[i] = output_layout.dim(i + params_offset);

  for (int i = 0; i < indices_rank - 1; ++i)
    indices_specs[i] = output_layout.dim(i);

  indices_specs[indices_rank - 1].set_sharding_spec(Layout::kUnshardedDim);

  TF_ASSIGN_OR_RETURN(*params_layout, Layout::GetLayout(params_specs, mesh));
  TF_ASSIGN_OR_RETURN(*indices_layout, Layout::GetLayout(indices_specs, mesh));

  return Status::OK();
}

}  // namespace

StatusOr<mlir::Operation*> GatherNdSPMDExpander::ExpandOp(mlir::Operation* op) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSgather_spmd_expanderDTcc mht_3(mht_3_v, 548, "", "./tensorflow/dtensor/mlir/expansions/gather_spmd_expander.cc", "GatherNdSPMDExpander::ExpandOp");

  auto gather_op = llvm::cast<mlir::TF::GatherNdOp>(op);
  TF_ASSIGN_OR_RETURN(Layout params_layout,
                      ExtractRequiredLayoutFromOperand(gather_op.params()));
  TF_ASSIGN_OR_RETURN(Layout indices_layout,
                      ExtractRequiredLayoutFromOperand(gather_op.indices()));
  TF_ASSIGN_OR_RETURN(Layout output_layout,
                      ExtractRequiredSingleLayoutFromOp(gather_op));

  TF_ASSIGN_OR_RETURN(
      llvm::ArrayRef<int64_t> indices_shape,
      GetGlobalShapeOfValueFromDTensorLayout(gather_op.indices()));
  const int index_dimensions = indices_shape.back();

  const auto params_rank = ValueRank(gather_op.params());
  const auto indices_rank = ValueRank(gather_op.indices());
  if (params_rank == -1)
    return errors::InvalidArgument("missing rank for params input");
  if (indices_rank == -1)
    return errors::InvalidArgument("missing rank for indices input");

  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));

  // Note that there may be conflicts between the two input layouts.
  // To resolve these we:
  // 1) Compute output layout from the current input layouts. This ignores any
  //    sharding on the 'lookup' axis in params and the last axis in indices.
  //    If params and indices are sharded on same dimension, the indices will
  //    be unsharded in that dimension.
  // 2) Merge any new sharding from the final output layout into this layout.
  //    E.g. if the two inputs are unsharded, but the output is, this will be
  //    more efficient as the local computations will be smaller since we will
  //    more finely shard the inputs.
  // 3) Finally, we use this new output layout to compute what input layouts we
  //    need to get the given output.

  // Step 1)
  TF_ASSIGN_OR_RETURN(const Layout pre_output_layout,
                      GatherNdGetOutputLayoutFromInput(
                          params_layout, params_rank, indices_layout,
                          indices_rank, index_dimensions, mesh));

  // Step 2)
  llvm::DenseSet<llvm::StringRef> used_dimensions;
  for (const ShardingSpec& spec : pre_output_layout.sharding_specs())
    if (Layout::IsShardedSpec(spec))
      used_dimensions.insert(spec.sharding_spec());

  std::vector<ShardingSpec> sharding_specs(output_layout.rank());
  for (int i = 0; i < sharding_specs.size(); ++i) {
    if (Layout::IsShardedSpec(pre_output_layout.dim(i)))
      sharding_specs[i] = pre_output_layout.dim(i);
    // Merge in sharded dimensions from the output which aren't already used
    // by the pre_output_layout.
    else if (Layout::IsShardedSpec(output_layout.dim(i)) &&
             !used_dimensions.contains(output_layout.sharding_spec(i)))
      sharding_specs[i] = output_layout.dim(i);
    else
      sharding_specs[i].set_sharding_spec(Layout::kUnshardedDim);
  }

  // Step 3)
  TF_ASSIGN_OR_RETURN(
      const Layout merged_output_layout,
      Layout::GetLayout(sharding_specs, pre_output_layout.mesh()));

  Layout new_params_layout;
  Layout new_indices_layout;
  TF_RETURN_IF_ERROR(GatherNdGetInputLayoutFromOutput(
      merged_output_layout, &new_params_layout, params_rank,
      &new_indices_layout, indices_rank, index_dimensions, mesh));

  TF_ASSIGN_OR_RETURN(
      mlir::Value params,
      EmitRelayout(gather_op.params(), params_layout, new_params_layout));
  TF_ASSIGN_OR_RETURN(
      mlir::Value indices,
      EmitRelayout(gather_op.indices(), indices_layout, new_indices_layout));

  mlir::OpBuilder builder(op);

  // mlir::TF::GatherNdOp does not have a builder that does shape inference
  // so we have to provided the output type. This output type is incorrect and
  // we manually run shape inference after.
  mlir::TF::GatherNdOp new_gather = builder.create<mlir::TF::GatherNdOp>(
      op->getLoc(), gather_op.output().getType(), params, indices);
  InferSPMDExpandedLocalShape(new_gather);

  TF_ASSIGN_OR_RETURN(
      mlir::Value output,
      EmitRelayout(new_gather.output(), merged_output_layout, output_layout));

  gather_op.output().replaceAllUsesWith(output);
  gather_op.erase();
  return output.getDefiningOp();
}

StatusOr<llvm::DenseMap<int, Layout>>
GatherNdSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  auto gather_op = llvm::cast<mlir::TF::GatherNdOp>(op);
  TF_ASSIGN_OR_RETURN(Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));

  absl::optional<Layout> params_layout;
  if (input_layouts.find(0) != input_layouts.end())
    params_layout.emplace(input_layouts.lookup(0));
  absl::optional<Layout> indices_layout;
  if (input_layouts.find(1) != input_layouts.end())
    indices_layout.emplace(input_layouts.lookup(1));

  TF_ASSIGN_OR_RETURN(llvm::ArrayRef<int64_t> indices_shape,
                      ExtractGlobalInputShape(op->getOpOperand(1)));
  const int index_dimensions = indices_shape.back();
  if (index_dimensions < 0)
    return errors::Unimplemented(
        "dynamic last dimension for index is not supported");

  const int params_rank = ValueRank(gather_op.params());
  const int indices_rank = ValueRank(gather_op.indices());
  if (params_rank == -1)
    return errors::InvalidArgument("missing rank for params input");
  if (indices_rank == -1)
    return errors::InvalidArgument("missing rank for indices input");

  if (params_layout || indices_layout) {
    TF_ASSIGN_OR_RETURN(const Layout output_layout,
                        GatherNdGetOutputLayoutFromInput(
                            params_layout, params_rank, indices_layout,
                            indices_rank, index_dimensions, mesh));
    return llvm::DenseMap<int, Layout>({{0, output_layout}});
  }

  return llvm::DenseMap<int, Layout>();
}

StatusOr<llvm::DenseMap<int, Layout>>
GatherNdSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  // If there is no output layout present then do not infer any operand layouts.
  if (output_layouts.find(0) == output_layouts.end())
    return llvm::DenseMap<int, Layout>();

  auto gather_op = llvm::cast<mlir::TF::GatherNdOp>(op);
  TF_ASSIGN_OR_RETURN(Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));

  TF_ASSIGN_OR_RETURN(llvm::ArrayRef<int64_t> indices_shape,
                      ExtractGlobalInputShape(op->getOpOperand(1)));
  const int index_dimensions = indices_shape.back();
  if (index_dimensions < 0)
    return errors::Unimplemented(
        "dynamic last dimension for index is not supported");

  const int params_rank = ValueRank(gather_op.params());
  const int indices_rank = ValueRank(gather_op.indices());
  if (params_rank == -1)
    return errors::InvalidArgument("missing rank for params input");
  if (indices_rank == -1)
    return errors::InvalidArgument("missing rank for indices input");

  const Layout output_layout = output_layouts.lookup(0);
  Layout params_layout, indices_layout;
  TF_RETURN_IF_ERROR(GatherNdGetInputLayoutFromOutput(
      output_layout, &params_layout, params_rank, &indices_layout, indices_rank,
      index_dimensions, mesh));
  return llvm::DenseMap<int, Layout>({
      {0, params_layout},
      {1, indices_layout},
  });
}

}  // namespace dtensor
}  // namespace tensorflow
