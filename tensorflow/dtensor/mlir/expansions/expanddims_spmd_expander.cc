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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSexpanddims_spmd_expanderDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSexpanddims_spmd_expanderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSexpanddims_spmd_expanderDTcc() {
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

#include "tensorflow/dtensor/mlir/expansions/expanddims_spmd_expander.h"

#include <string>
#include <utility>

#include "absl/types/optional.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {

StatusOr<mlir::Operation*> ExpandDimsExpander::ExpandOp(mlir::Operation* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSexpanddims_spmd_expanderDTcc mht_0(mht_0_v, 199, "", "./tensorflow/dtensor/mlir/expansions/expanddims_spmd_expander.cc", "ExpandDimsExpander::ExpandOp");

  TF_ASSIGN_OR_RETURN(const absl::optional<Layout> output_layout,
                      ExtractSingleLayoutFromOp(op));
  TF_ASSIGN_OR_RETURN(const absl::optional<Layout> operand_layout,
                      ExtractLayoutFromOperand(op->getOperand(0)));
  mlir::TF::ExpandDimsOp expand_dims_op =
      mlir::cast<mlir::TF::ExpandDimsOp>(op);

  InferSPMDExpandedLocalShape(op);

  TF_ASSIGN_OR_RETURN(
      llvm::ArrayRef<int64_t> global_output_shape,
      GetGlobalShapeOfValueFromDTensorLayout(expand_dims_op.output()));

  // Compute current output layout (just input layout with unsharded on the
  // new dim);
  TF_ASSIGN_OR_RETURN(int64_t dim,
                      ExtractConstIntFromValue(expand_dims_op.dim()));

  if (dim < 0) dim += global_output_shape.size();
  std::vector<ShardingSpec> sharding_specs(global_output_shape.size());
  for (int i = 0; i < global_output_shape.size(); ++i) {
    if (i < dim)
      sharding_specs[i] = operand_layout->dim(i);
    else if (i == dim)
      sharding_specs[i].set_sharding_spec(Layout::kUnshardedDim);
    else
      sharding_specs[i] = operand_layout->dim(i - 1);
  }
  TF_ASSIGN_OR_RETURN(const Layout current_output_layout,
                      Layout::GetLayout(sharding_specs, output_layout->mesh()));

  llvm::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;

  TF_ASSIGN_OR_RETURN(
      mlir::Value output_value,
      EmitRelayout(expand_dims_op.output(), current_output_layout,
                   *output_layout, &newly_created_ops));

  expand_dims_op.output().replaceAllUsesExcept(output_value, newly_created_ops);

  return output_value.getDefiningOp();
}

StatusOr<llvm::DenseMap<int, Layout>> ExpandDimsExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));

  auto expand_dims_op = mlir::cast<mlir::TF::ExpandDimsOp>(op);

  TF_ASSIGN_OR_RETURN(int64_t dim,
                      ExtractConstIntFromValue(expand_dims_op.dim()));

  // Do not infer any output layout if no operand layout is present.
  if (input_layouts.find(0) == input_layouts.end())
    return llvm::DenseMap<int, Layout>();

  auto input_layout = input_layouts.lookup(0);

  if (dim < 0) dim += input_layout.rank() + 1;
  std::vector<std::string> layout_sharding;

  for (int i = 0; i <= input_layout.rank(); ++i) {
    if (i == dim) layout_sharding.push_back(Layout::kUnshardedDim);
    if (i < input_layout.rank())
      layout_sharding.push_back(input_layout.sharding_spec(i));
  }
  TF_ASSIGN_OR_RETURN(auto inferred_output_layout,
                      Layout::GetLayout(layout_sharding, mesh));
  return llvm::DenseMap<int, Layout>({{0, inferred_output_layout}});
}

StatusOr<llvm::DenseMap<int, Layout>> ExpandDimsExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  if (output_layouts.find(0) == output_layouts.end())
    return llvm::DenseMap<int, Layout>();
  auto output_layout = output_layouts.lookup(0);

  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));

  auto expand_dims_op = mlir::cast<mlir::TF::ExpandDimsOp>(op);

  TF_ASSIGN_OR_RETURN(int64_t dim,
                      ExtractConstIntFromValue(expand_dims_op.dim()));

  if (dim < 0) dim += output_layout.rank();

  std::vector<std::string> layout_sharding;

  for (int i = 0; i < output_layout.rank(); ++i) {
    if (i == dim) continue;
    layout_sharding.push_back(output_layout.sharding_spec(i));
  }

  TF_ASSIGN_OR_RETURN(auto inferred_input_layout,
                      Layout::GetLayout(layout_sharding, mesh));
  auto input_axis_rank = ValueRank(expand_dims_op->getOperand(1));

  return llvm::DenseMap<int, Layout>(
      {{0, inferred_input_layout},
       {1, Layout::ReplicatedOnMesh(mesh, /*rank=*/input_axis_rank)}});
}

}  // namespace dtensor
}  // namespace tensorflow
