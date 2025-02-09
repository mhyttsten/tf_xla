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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSmatmul_spmd_expanderDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSmatmul_spmd_expanderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSmatmul_spmd_expanderDTcc() {
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

#include "tensorflow/dtensor/mlir/expansions/matmul_spmd_expander.h"

#include <string>

#include "absl/container/flat_hash_set.h"
#include "llvm/ADT/SmallSet.h"
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {
namespace {

void GetTransposeSettings(mlir::Operation* op, bool* left_transposed,
                          bool* right_transposed) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSmatmul_spmd_expanderDTcc mht_0(mht_0_v, 209, "", "./tensorflow/dtensor/mlir/expansions/matmul_spmd_expander.cc", "GetTransposeSettings");

  if (mlir::isa<mlir::TF::BatchMatMulV2Op>(op)) {
    mlir::TF::BatchMatMulV2Op mm = mlir::cast<mlir::TF::BatchMatMulV2Op>(op);
    // Adjoint is just conjugate transpose.
    *left_transposed = mm.adj_x();
    *right_transposed = mm.adj_y();
  } else if (mlir::isa<mlir::TF::MatMulOp>(op)) {
    mlir::TF::MatMulOp mm = mlir::cast<mlir::TF::MatMulOp>(op);
    *left_transposed = mm.transpose_a();
    *right_transposed = mm.transpose_b();
  }
}

}  // namespace

StatusOr<mlir::Operation*> MatMulSPMDExpander::ExpandOp(mlir::Operation* op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSmatmul_spmd_expanderDTcc mht_1(mht_1_v, 227, "", "./tensorflow/dtensor/mlir/expansions/matmul_spmd_expander.cc", "MatMulSPMDExpander::ExpandOp");

  absl::flat_hash_set<std::string> reduced_dims;
  bool left_transposed;
  bool right_transposed;
  TF_ASSIGN_OR_RETURN(const Layout left_layout,
                      ExtractRequiredLayoutFromOperand(op->getOperand(0)));
  TF_ASSIGN_OR_RETURN(const Layout right_layout,
                      ExtractRequiredLayoutFromOperand(op->getOperand(1)));
  TF_ASSIGN_OR_RETURN(const Layout output_layout,
                      ExtractRequiredSingleLayoutFromOp(op));

  GetTransposeSettings(op, &left_transposed, &right_transposed);

  std::string reduce_dim;
  Layout layout_after_matmul;
  mlir::Value left, right;

  TF_RETURN_IF_ERROR(MaybeRelayoutInputs(
      op, left_layout, left_transposed, right_layout, right_transposed,
      output_layout, reduce_dim, layout_after_matmul, left, right));

  mlir::OpBuilder builder(op);

  mlir::BlockAndValueMapping mapping;
  mapping.map(op->getOperand(0), left);
  mapping.map(op->getOperand(1), right);
  mlir::Operation* new_op = builder.clone(*op, mapping);
  // Note that the output shape of new_op is cloned from op, so we need to
  // update to the local shape.
  new_op = InferSPMDExpandedLocalShape(new_op);

  if (Layout::IsShardedDimension(reduce_dim)) {
    TF_ASSIGN_OR_RETURN(
        new_op, EmitAllReduce(builder, layout_after_matmul, {reduce_dim},
                              new_op, kReduceOpAdd));
  }

  TF_ASSIGN_OR_RETURN(
      auto final_output,
      EmitRelayout(new_op->getOpResult(0), layout_after_matmul, output_layout));

  op->getOpResult(0).replaceAllUsesWith(final_output);
  op->erase();

  return final_output.getDefiningOp();
}

StatusOr<Layout> MatMulSPMDExpander::OutputLayoutAndReducedDims(
    bool allow_unknown_layouts, mlir::Operation* op,
    absl::flat_hash_set<std::string>* reduced_dims,
    absl::optional<Layout>* left, absl::optional<Layout>* right) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSmatmul_spmd_expanderDTcc mht_2(mht_2_v, 280, "", "./tensorflow/dtensor/mlir/expansions/matmul_spmd_expander.cc", "MatMulSPMDExpander::OutputLayoutAndReducedDims");

  // These layouts are 2d layouts for the non-batch dimensions.
  Layout left_layout;
  Layout right_layout;
  bool left_transposed;
  bool right_transposed;

  // This will hold the batch layout for the output.
  Layout batch_layout;

  if (!*left || !*right) {
    if (allow_unknown_layouts) return Status::OK();
    return errors::Unimplemented("failed to do SPMD expansion for ", OpName(op),
                                 " operand layouts "
                                 "unknown");
  }

  if (mlir::isa<mlir::TF::BatchMatMulV2Op>(op)) {
    mlir::TF::BatchMatMulV2Op mm = mlir::cast<mlir::TF::BatchMatMulV2Op>(op);

    // Note that it doesn't matter if we pass the global or local shape to
    // GetBroadcastLayoutForElementWise, it will return the same result.
    TF_ASSIGN_OR_RETURN(const auto left_shape, GetShapeOfValue(mm.x()));
    TF_ASSIGN_OR_RETURN(const auto right_shape, GetShapeOfValue(mm.y()));
    std::vector<std::string> left_splits;
    std::vector<std::string> right_splits;
    TF_ASSIGN_OR_RETURN(
        batch_layout,
        GetBroadcastLayoutForElementWise(
            left->value(), right->value(), left_shape, right_shape,
            /*dims_to_ignore=*/2, left_splits, right_splits));

    left_layout = (*left)->Truncate(left_shape.size() - 2, /*end=*/true);
    right_layout = (*right)->Truncate(right_shape.size() - 2, /*end=*/true);
  } else if (mlir::isa<mlir::TF::MatMulOp>(op)) {
    // There are no batch dims for MatMul op, so get an 'empty' layout that
    // we can concat later.
    batch_layout = (*left)->Truncate(/*split_point=*/0, /*end=*/false);
    left_layout = left->value();
    right_layout = right->value();
  } else {
    return errors::Internal("Unknown op ", OpName(op));
  }
  GetTransposeSettings(op, &left_transposed, &right_transposed);

  if (left_transposed) {
    TF_ASSIGN_OR_RETURN(left_layout, Layout::Transposed2D(left_layout));
  }
  if (right_transposed) {
    TF_ASSIGN_OR_RETURN(right_layout, Layout::Transposed2D(right_layout));
  }

  // Input layouts are [batch...],a,b;[batch...],b,c
  // Output layout is [batch...],a,c
  const auto& batch_sharding_specs = batch_layout.sharding_specs();
  std::vector<ShardingSpec> output_dims(batch_sharding_specs.begin(),
                                        batch_sharding_specs.end());
  if (Layout::IsShardedDimension(left_layout.sharding_spec(0)) &&
      left_layout.sharding_spec(0) == right_layout.sharding_spec(1)) {
    // If a and c above are the same and sharded, we should output a replicated
    // layout during propagation. This is so we don't create an illegal layout.
    output_dims.resize(output_dims.size() + 2);
    output_dims[output_dims.size() - 2].set_sharding_spec(
        Layout::kUnshardedDim);
    output_dims[output_dims.size() - 1].set_sharding_spec(
        Layout::kUnshardedDim);
  } else {
    output_dims.emplace_back(left_layout.dim(0));
    output_dims.emplace_back(right_layout.dim(1));
  }

  return Layout::GetLayout(output_dims, left_layout.mesh());
}

// This function will take the left and right input, possibly slice or add
// AllConcat along various mesh dimensions before the MatMul operation takes
// place. This also returns:
// * The mesh dimension, if any, that the output of the matmul should be
//   summed along.
// * The resulting layout of the matmul tensor, so we can insert an AllConcat/
//   split to make the output have the desired layout.
// * The left and right value for use as input to the matmul.
Status MatMulSPMDExpander::MaybeRelayoutInputs(
    mlir::Operation* op, const Layout& left_layout, bool left_transposed,
    const Layout& right_layout, bool right_transposed,
    const Layout& output_layout, std::string& reduced_dim,
    Layout& matmul_layout, mlir::Value& left, mlir::Value& right) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSmatmul_spmd_expanderDTcc mht_3(mht_3_v, 369, "", "./tensorflow/dtensor/mlir/expansions/matmul_spmd_expander.cc", "MatMulSPMDExpander::MaybeRelayoutInputs");

  // These two lists will contain the mesh dimensions desired for the left
  // and right inputs before the matmul. Since a Layout is generally immutable,
  // we use these vectors to store sharding for the layout and produce the
  // final layout at the end via Layout::GetLayout.
  std::vector<std::string> left_specs = left_layout.sharding_spec_strs();
  std::vector<std::string> right_specs = right_layout.sharding_spec_strs();
  // Specs for the layout of the matmul.
  std::vector<std::string> matmul_specs(output_layout.rank());

  TF_ASSIGN_OR_RETURN(const std::vector<int64_t> left_shape,
                      GetShapeOfValue(op->getOperand(0)));
  TF_ASSIGN_OR_RETURN(const std::vector<int64_t> right_shape,
                      GetShapeOfValue(op->getOperand(1)));

  const std::vector<int64_t> left_global_shape =
      left_layout.GlobalShapeFromLocalShape(left_shape);
  const std::vector<int64_t> right_global_shape =
      right_layout.GlobalShapeFromLocalShape(right_shape);

  // From this point on, we will use a short hand in the comments for the
  // last two dimensions of the inputs and output:
  // left: a,b right: c,d output: e,f after left and right are appropriately
  // transposed.

  std::string& a = left_specs[left_specs.size() - 2];
  std::string& b = left_specs[left_specs.size() - 1];
  std::string& c = right_specs[right_specs.size() - 2];
  std::string& d = right_specs[right_specs.size() - 1];
  const std::string& e = output_layout.sharding_spec(output_layout.rank() - 2);
  const std::string& f = output_layout.sharding_spec(output_layout.rank() - 1);

  if (left_transposed) std::swap(a, b);
  if (right_transposed) std::swap(c, d);

  // Set the mesh dimensions along the batch axis for the left and right side
  // from the output. This is relatively simple choice and there are cases that
  // we could improve:
  // - Left and right input are both sharded on a dimension and the output
  //   is not. With the current algorithm we will unshard the inputs. But it
  //   would be more efficient to leave the inputs sharded and unshard the
  //   output.
  llvm::SmallSet<std::string, 4> used_mesh_dimensions;
  used_mesh_dimensions.insert(a);
  used_mesh_dimensions.insert(b);
  used_mesh_dimensions.insert(c);
  used_mesh_dimensions.insert(d);
  for (int i = 0; i < matmul_specs.size() - 2; ++i) {
    matmul_specs[i] = output_layout.sharding_spec(i);
    if (used_mesh_dimensions.contains(matmul_specs[i]))
      matmul_specs[i] = Layout::kUnshardedDim;
    if (i >= matmul_specs.size() - left_specs.size()) {
      const int64_t left_pos = left_specs.size() - matmul_specs.size() + i;
      left_specs[left_pos] = matmul_specs[i];
      // If the left global shape is 1, its broadcasted so just set the
      // dimension to unsharded.
      if (left_global_shape[left_pos] == 1)
        left_specs[left_pos] = Layout::kUnshardedDim;
    }
    if (i >= matmul_specs.size() - right_specs.size()) {
      const int64_t right_pos = right_specs.size() - matmul_specs.size() + i;
      right_specs[right_pos] = matmul_specs[i];
      // If the right global shape is 1, its broadcasted so just set the
      // dimension to unsharded.
      if (right_global_shape[right_pos] == 1)
        right_specs[right_pos] = Layout::kUnshardedDim;
    }
  }

  // Reject the cases that we don't yet support, namely the contracting
  // dimensions are sharded not equal, or the input and output non-contracting
  // dimensions are equal and are sharded. These would require more extensive
  // relayout to solve.
  if (b != c && Layout::IsShardedDimension(b) && Layout::IsShardedDimension(c))
    return errors::InvalidArgument(
        "Contracting dimension for matmul has sharding dimension ", b,
        " for the left input and ", c,
        " for the right input which are not equal. This case is currently not "
        "supported.");

  if (a != e && Layout::IsShardedDimension(a) && Layout::IsShardedDimension(e))
    return errors::InvalidArgument(
        "Non-contracting dimension for left argument of matmul has sharding "
        "dimension ",
        a,
        " and the second to last dimension of the output has sharding "
        "dimension ",
        e, ", which are not equal. This case is currently not supported.");

  if (d != f && Layout::IsShardedDimension(d) && Layout::IsShardedDimension(f))
    return errors::InvalidArgument(
        "Non-contracting dimension for right argument of matmul has sharding "
        "dimension ",
        d, " and the last dimension of the output has sharding dimension ", f,
        ", which are not equal. This case is currently not supported.");

  // If the output is sharded and the corresponding non-contracting input is not
  // sharded, then shard the input on that dim, to reduce the amount of work
  // done. Note that this sharding spec can't be used anywhere in the batch
  // dimensions due the agreement between the sharding specs of the batch
  // dimensions for the input and output, so this is safe.
  // This handles the *,x . x,* -> *,y case.
  if (Layout::IsUnshardedDimension(a) && Layout::IsShardedDimension(e) &&
      e != b && e != c && e != d)
    a = e;
  if (Layout::IsUnshardedDimension(d) && Layout::IsShardedDimension(f) &&
      f != a && f != b && f != c)
    d = f;

  // Handle the case when the non-contracting dimensions have the same
  // sharding spec. This can't happen if either of the previous two cases are
  // true as it would imply that e and f have the same sharding spec. So, a and
  // d are sharded in the input and we need to AllConcat one of them.
  // This handles the y,x . x,y -> *,y case.
  if (Layout::IsShardedDimension(a) && a == d) {
    if (a == e)
      d = Layout::kUnshardedDim;
    else if (d == f)
      a = Layout::kUnshardedDim;
    else
      // TODO(bfontain): Update this to pick a or d to AllConcat based on shape.
      a = Layout::kUnshardedDim;
  }

  // Handle the case where a non-contracting and contracting dim have the same
  // sharding spec. For now we always unshard the contracting axis. Note that
  // this is safe since, e.g. if a = c and both are sharded, then b must be
  // unsharded.
  // This handles the case x,y . *,y -> x,y
  // Beware:
  // Consider *,y . *,y -> *,*, there are two choices, unshard b or unshard d.
  // If we unshard d, then we will need to shard c and all reduce. This maybe
  // a good idea for performance, but the current EmitAllGather cannot handle
  // the transformation of *,y to y,*. Thus it is safer to always unshard b in
  // this case.
  if (Layout::IsShardedDimension(a) && a == c) c = Layout::kUnshardedDim;
  if (Layout::IsShardedDimension(b) && b == d) b = Layout::kUnshardedDim;

  // Finally, we make both contracting axes agree on a sharding.
  // If b and c are sharded, we checked about that their sharding is equal.
  // If c is sharded then a != c (if a == c, then the above case would set it
  // to unsharded). And since c was never part of the batch dimensions (we
  // specifically excluded it earlier), the dimension c is not used in the left
  // input so it is always safe to set b = c.
  if (b != c) {
    if (Layout::IsShardedDimension(b))
      c = b;
    else
      b = c;
  }
  reduced_dim = b;

  // Generate the layout that will be the output of the matmul.
  // This may be different from the final output layout in the last two
  // dimensions.
  matmul_specs[output_layout.rank() - 2] = a;
  matmul_specs[output_layout.rank() - 1] = d;

  TF_ASSIGN_OR_RETURN(matmul_layout,
                      Layout::GetLayout(matmul_specs, output_layout.mesh()));
  if (left_transposed) std::swap(a, b);
  if (right_transposed) std::swap(c, d);

  TF_ASSIGN_OR_RETURN(auto new_left_layout,
                      Layout::GetLayout(left_specs, left_layout.mesh()));
  TF_ASSIGN_OR_RETURN(auto new_right_layout,
                      Layout::GetLayout(right_specs, right_layout.mesh()));

  TF_ASSIGN_OR_RETURN(
      left, EmitRelayout(op->getOperand(0), left_layout, new_left_layout));
  TF_ASSIGN_OR_RETURN(
      right, EmitRelayout(op->getOperand(1), right_layout, new_right_layout));

  return Status::OK();
}

StatusOr<llvm::DenseMap<int, Layout>> MatMulSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  if (input_layouts.empty()) return llvm::DenseMap<int, Layout>();

  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  absl::flat_hash_set<std::string> reduced_dims;
  TF_ASSIGN_OR_RETURN(const auto left_shape,
                      GetShapeOfValue(op->getOperand(0)));
  TF_ASSIGN_OR_RETURN(const auto right_shape,
                      GetShapeOfValue(op->getOperand(1)));

  // At least one input is set, calculate an output layout.
  absl::optional<Layout> left, right;
  if (input_layouts.find(0) != input_layouts.end())
    left.emplace(input_layouts.lookup(0));
  else
    left.emplace(Layout::ReplicatedOnMesh(mesh, left_shape.size()));
  if (input_layouts.find(1) != input_layouts.end())
    right.emplace(input_layouts.lookup(1));
  else
    right.emplace(Layout::ReplicatedOnMesh(mesh, right_shape.size()));

  TF_ASSIGN_OR_RETURN(
      const Layout output_layout,
      OutputLayoutAndReducedDims(
          /*allow_unknown_layouts=*/true, op, &reduced_dims, &left, &right));

  return llvm::DenseMap<int, Layout>({{0, output_layout}});
}

StatusOr<llvm::DenseMap<int, Layout>> MatMulSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  if (output_layouts.find(0) == output_layouts.end())
    return llvm::DenseMap<int, Layout>();

  const Layout output_layout = output_layouts.lookup(0);
  TF_ASSIGN_OR_RETURN(const auto left_shape,
                      GetShapeOfValue(op->getOperand(0)));
  TF_ASSIGN_OR_RETURN(const auto right_shape,
                      GetShapeOfValue(op->getOperand(1)));

  // We take output layout and 'copy' it to the two input layouts, but with
  // the contracting dimension set to replicated.
  // Note that some complication are introduced by the possibility of
  // broadcasting in the BatchMatMulV2 case.

  // Truncate layout in case of broadcast. Note that since
  // output->rank() == std::max(left_shape.size(), right_shape.size()) due to
  // broadcasting one of these truncations is just a copy of output and the
  // other may be shorter.
  Layout left = output_layout.Truncate(output_layout.rank() - left_shape.size(),
                                       /*end=*/true);
  Layout right = output_layout.Truncate(
      output_layout.rank() - right_shape.size(), /*end=*/true);

  // Make sure necessary dimensions are replicated.
  //
  // Due to broadcasting, each of the batch dimensions (i.e. from dimension 0
  // to dim - 2), one of the two inputs may have have dimension 1 while the
  // other has dimension > 1 and equal to the dim of the output. Since a
  // tensor with dimension 1 cannot be sharded, we set this to unsharded.
  auto specs_matmul_operands = [](const llvm::ArrayRef<int64>& tensor_shape,
                                  const Layout& layout,
                                  bool is_left_operand) -> StatusOr<Layout> {
    int contracting_dim =
        is_left_operand ? layout.rank() - 1 : layout.rank() - 2;
    // Assign "any" to the contracting dim and "unsharded" to any tensor dim
    // of length = 1.
    std::vector<std::string> sharding_specs = layout.sharding_spec_strs();
    for (size_t i = 0; i < layout.rank(); ++i)
      if (i == contracting_dim) {
        sharding_specs[i] = Layout::kAny;
      } else if (tensor_shape[i] == 1) {
        sharding_specs[i] = Layout::kUnshardedDim;
      }
    return Layout::GetLayout(sharding_specs, layout.mesh());
  };

  TF_ASSIGN_OR_RETURN(left, specs_matmul_operands(left_shape, left,
                                                  /*is_left_operand=*/true));
  TF_ASSIGN_OR_RETURN(right, specs_matmul_operands(right_shape, right,
                                                   /*is_left_operand=*/false));

  // Transpose the layouts if needed, as we just generated the non-transposed
  // layouts.
  bool left_transposed;
  bool right_transposed;
  GetTransposeSettings(op, &left_transposed, &right_transposed);
  if (left_transposed) {
    TF_ASSIGN_OR_RETURN(left, Layout::Transposed2D(left));
  }
  if (right_transposed) {
    TF_ASSIGN_OR_RETURN(right, Layout::Transposed2D(right));
  }

  return llvm::DenseMap<int, Layout>({{0, left}, {1, right}});
}

}  // namespace dtensor
}  // namespace tensorflow
