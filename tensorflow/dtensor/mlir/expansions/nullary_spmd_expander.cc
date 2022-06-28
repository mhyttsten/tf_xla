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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSnullary_spmd_expanderDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSnullary_spmd_expanderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSnullary_spmd_expanderDTcc() {
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

#include "tensorflow/dtensor/mlir/expansions/nullary_spmd_expander.h"

#include "absl/strings/str_join.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/utils/array_container_utils.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"

namespace tensorflow {
namespace dtensor {

StatusOr<mlir::Operation*> NullarySPMDExpander::ExpandOp(mlir::Operation* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSnullary_spmd_expanderDTcc mht_0(mht_0_v, 209, "", "./tensorflow/dtensor/mlir/expansions/nullary_spmd_expander.cc", "NullarySPMDExpander::ExpandOp");

  if (op->getNumResults() == 0) return op;

  bool all_operands_fully_replicated = true;
  TF_ASSIGN_OR_RETURN(auto op_layouts, ExtractLayoutFromOp(op));
  for (const auto& op_layout : op_layouts) {
    if (!op_layout)
      return errors::InvalidArgument(
          "Nullary op layouts must be known before SPMD expansion.");
    all_operands_fully_replicated =
        all_operands_fully_replicated && op_layout->IsFullyReplicated();
  }

  if (all_operands_fully_replicated) return op;

  if (auto const_op = mlir::dyn_cast<mlir::TF::ConstOp>(op)) {
    if (auto dense = const_op.value().dyn_cast<mlir::DenseElementsAttr>()) {
      if (dense.isSplat()) {
        // A 'splat' value for a DenseElementsAttr, has a single value for
        // all its elements. For these inputs, we don't need to slice. We just
        // need to update the shape of the attribute given the requested
        // sharding.
        assert(dense.getType().getRank() == op_layouts[0]->rank());
        auto shape = dense.getType().getShape();
        std::vector<int64_t> new_shape(dense.getType().getRank());
        for (int i = 0; i < op_layouts[0]->rank(); ++i) {
          const int num_shards =
              op_layouts[0]->num_shards_for_dim(op_layouts[0]->dim(i));
          if (shape[i] % num_shards != 0)
            return errors::InvalidArgument(
                "has output dimension size ", shape[i],
                " which is not evenly divisible by the number of shards ",
                num_shards, " in the layout for that dimension.");
          new_shape[i] = shape[i] / num_shards;
        }
        const_op.valueAttr(mlir::DenseElementsAttr::get(
            mlir::RankedTensorType::get(new_shape,
                                        dense.getType().getElementType()),
            dense.getSplatValue<mlir::Attribute>()));
        return InferSPMDExpandedLocalShape(op);
      }
    }
  }

  llvm::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;
  llvm::SmallVector<mlir::Value, 4> generated_outputs;
  llvm::SmallVector<mlir::Type, 4> generated_types;

  mlir::OpBuilder builder(op);
  builder.setInsertionPointAfter(op);
  for (int i = 0; i < op_layouts.size(); ++i) {
    // Split each output to the correct layout by assuming the input is
    // replicated.
    TF_ASSIGN_OR_RETURN(
        const mlir::Value output,
        EmitAllScatter(builder, op->getOpResult(i),
                       Layout::ReplicatedOnMesh(op_layouts[i]->mesh(),
                                                op_layouts[i]->rank()),
                       *op_layouts[i], &newly_created_ops));
    generated_outputs.emplace_back(output);
    generated_types.emplace_back(output.getType());
  }

  auto identity_op = builder.create<mlir::TF::IdentityNOp>(
      op->getLoc(), generated_types, generated_outputs);

  for (int i = 0; i < op_layouts.size(); ++i)
    op->getOpResult(i).replaceAllUsesExcept(identity_op.getResult(i),
                                            newly_created_ops);

  return identity_op.getOperation();
}

StatusOr<llvm::DenseMap<int, Layout>> NullarySPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  auto enclosing_mesh = op->getParentOfType<mlir::tf_device::ClusterOp>();
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshFromOp(enclosing_mesh));
  if (!mesh.has_value())
    return errors::Internal("Failure in extracting mesh from Nullary Op.");
  llvm::DenseMap<int, Layout> output_layouts;
  // Nullary ops always output replicated layout for output values.
  for (auto i = 0; i < op->getNumResults(); ++i) {
    auto output_ranked_type =
        op->getResult(i).getType().dyn_cast<mlir::RankedTensorType>();
    if (!output_ranked_type) {
      return errors::InvalidArgument(
          llvm::formatv("requires output type to have statically known rank, "
                        "but got : {0}",
                        output_ranked_type)
              .str());
    }
    output_layouts[i] =
        Layout::ReplicatedOnMesh(*mesh, output_ranked_type.getRank());
  }
  return output_layouts;
}

StatusOr<llvm::DenseMap<int, Layout>>
NullarySPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  // No operand inputs.
  return llvm::DenseMap<int, Layout>();
}

}  // namespace dtensor
}  // namespace tensorflow
