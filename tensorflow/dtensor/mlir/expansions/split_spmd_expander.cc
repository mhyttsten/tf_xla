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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSsplit_spmd_expanderDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSsplit_spmd_expanderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSsplit_spmd_expanderDTcc() {
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

#include "tensorflow/dtensor/mlir/expansions/split_spmd_expander.h"

#include <algorithm>
#include <cstdint>

#include "absl/types/optional.h"
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"
#include "tensorflow/dtensor/proto/layout.pb.h"

namespace tensorflow {
namespace dtensor {
namespace {

// Merges the output layouts to a common layout for Split/SplitV.
//
// Split has multiple outputs, so we first calculate a common output layout that
// we can use for passing backwards and then merge the remaining layouts.
StatusOr<Layout> MergeLayoutsForSplitOutput(
    int64_t split_dim, const llvm::DenseMap<int, Layout>& layouts) {
  assert(!layouts.empty());
  const Layout& first_layout = layouts.begin()->getSecond();
  std::vector<ShardingSpec> sharding_specs(
      first_layout.sharding_specs().begin(),
      first_layout.sharding_specs().end());

  // Merge remaining layouts. If there is a conflicting sharding, then set the
  // dim to replicated.
  for (auto it = layouts.begin(); it != layouts.end(); ++it) {
    const Layout& output_layout = it->getSecond();
    for (int dim = 0; dim < output_layout.rank(); ++dim) {
      if (Layout::IsShardedDimension(output_layout.dim(dim).sharding_spec()) &&
          Layout::IsShardedDimension(sharding_specs[dim].sharding_spec()) &&
          output_layout.dim(dim).sharding_spec() !=
              sharding_specs[dim].sharding_spec()) {
        sharding_specs[dim].set_sharding_spec(Layout::kUnshardedDim);
      }
    }
  }
  // Force the split_dim to be unsharded.
  sharding_specs[split_dim].set_sharding_spec(Layout::kUnshardedDim);
  return Layout::GetLayout(sharding_specs, first_layout.mesh());
}

// Retrieves the value of the split_dim operand adjusted based on the input
// rank. The split_dim operand's value can be [-rank(input), rank(input)), which
// is adjusted to a positive value.
StatusOr<int64_t> GetAdjustedSplitDim(mlir::Value split_dim_value,
                                      mlir::Value input_value) {
  TF_ASSIGN_OR_RETURN(int64_t split_dim,
                      ExtractConstIntFromValue(split_dim_value));
  if (split_dim < 0) {
    int rank = ValueRank(input_value);
    if (rank == -1) {
      return errors::InvalidArgument("Input operand has rank -1.");
    }
    split_dim += rank;
  }
  return split_dim;
}

}  // namespace

StatusOr<mlir::Operation*> SplitSPMDExpander::ExpandOp(mlir::Operation* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSsplit_spmd_expanderDTcc mht_0(mht_0_v, 253, "", "./tensorflow/dtensor/mlir/expansions/split_spmd_expander.cc", "SplitSPMDExpander::ExpandOp");

  auto split_op = mlir::cast<mlir::TF::SplitOp>(op);
  TF_ASSIGN_OR_RETURN(const Layout input_layout,
                      ExtractRequiredLayoutFromOperand(split_op.value()));
  TF_ASSIGN_OR_RETURN(
      const int64_t split_dim,
      GetAdjustedSplitDim(split_op.split_dim(), split_op.value()));

  if (Layout::IsShardedDimension(input_layout.dim(split_dim).sharding_spec())) {
    return errors::InvalidArgument(
        "Spliting over sharded dimension is not supported.");
  }

  return InferSPMDExpandedLocalShape(op);
}

StatusOr<llvm::DenseMap<int, Layout>> SplitSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  auto split_op = mlir::cast<mlir::TF::SplitOp>(op);
  llvm::DenseMap<int, Layout> output_layouts(split_op.getNumResults());
  if (input_layouts.find(1) != input_layouts.end()) {
    const Layout& suggested_layout = input_layouts.lookup(1);
    for (int i = 0; i < split_op.getNumResults(); ++i) {
      output_layouts[i] = suggested_layout;
    }
  }
  return output_layouts;
}

StatusOr<llvm::DenseMap<int, Layout>> SplitSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  auto split_op = mlir::cast<mlir::TF::SplitOp>(op);
  TF_ASSIGN_OR_RETURN(Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));

  llvm::DenseMap<int, Layout> input_layouts(split_op.getNumOperands());
  // axis
  input_layouts[0] = Layout::ReplicatedOnMesh(mesh, /*rank=*/0);

  if (!output_layouts.empty()) {
    // Split has multiple outputs, first calculate a common output layout that
    // we can use for passing backwards.
    TF_ASSIGN_OR_RETURN(
        const int64_t split_dim,
        GetAdjustedSplitDim(split_op.split_dim(), split_op.value()));
    TF_ASSIGN_OR_RETURN(const Layout common_output_layout,
                        MergeLayoutsForSplitOutput(split_dim, output_layouts));
    // value
    input_layouts[1] = common_output_layout;
  }

  return input_layouts;
}

StatusOr<mlir::Operation*> SplitVSPMDExpander::ExpandOp(mlir::Operation* op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSsplit_spmd_expanderDTcc mht_1(mht_1_v, 309, "", "./tensorflow/dtensor/mlir/expansions/split_spmd_expander.cc", "SplitVSPMDExpander::ExpandOp");

  auto split_v_op = mlir::cast<mlir::TF::SplitVOp>(op);
  TF_ASSIGN_OR_RETURN(const Layout input_layout,
                      ExtractRequiredLayoutFromOperand(split_v_op.value()));
  TF_ASSIGN_OR_RETURN(
      const int64_t split_dim,
      GetAdjustedSplitDim(split_v_op.split_dim(), split_v_op.value()));

  if (Layout::IsShardedDimension(input_layout.dim(split_dim).sharding_spec())) {
    return errors::InvalidArgument(
        "Spliting over sharded dimension is not supported.");
  }

  return InferSPMDExpandedLocalShape(op);
}

StatusOr<llvm::DenseMap<int, Layout>> SplitVSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  auto split_v_op = mlir::cast<mlir::TF::SplitVOp>(op);
  llvm::DenseMap<int, Layout> output_layouts(split_v_op.getNumResults());
  if (input_layouts.find(0) != input_layouts.end()) {
    const Layout& suggested_layout = input_layouts.lookup(0);
    for (int i = 0; i < split_v_op.getNumResults(); ++i) {
      output_layouts[i] = suggested_layout;
    }
  }
  return output_layouts;
}

StatusOr<llvm::DenseMap<int, Layout>> SplitVSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  auto split_v_op = mlir::cast<mlir::TF::SplitVOp>(op);
  TF_ASSIGN_OR_RETURN(Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));

  llvm::DenseMap<int, Layout> input_layouts(split_v_op.getNumOperands());
  // size_splits
  input_layouts[1] = Layout::ReplicatedOnMesh(mesh, /*rank=*/1);
  // axis
  input_layouts[2] = Layout::ReplicatedOnMesh(mesh, /*rank=*/0);

  if (!output_layouts.empty()) {
    // Split has multiple outputs, first calculate a common output layout that
    // we can use for passing backwards.
    TF_ASSIGN_OR_RETURN(
        const int64_t split_dim,
        GetAdjustedSplitDim(split_v_op.split_dim(), split_v_op.value()));
    TF_ASSIGN_OR_RETURN(const Layout common_output_layout,
                        MergeLayoutsForSplitOutput(split_dim, output_layouts));
    // value
    input_layouts[0] = common_output_layout;
  }

  return input_layouts;
}

}  // namespace dtensor
}  // namespace tensorflow
