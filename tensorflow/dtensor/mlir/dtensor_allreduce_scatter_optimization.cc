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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_allreduce_scatter_optimizationDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_allreduce_scatter_optimizationDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_allreduce_scatter_optimizationDTcc() {
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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/UseDefLists.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/dtensor/mlir/collectives_common.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes_classes.h"
#include "tensorflow/dtensor/mlir/group_assignment.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"

namespace tensorflow {
namespace dtensor {
namespace {

// Returns true if both group assignments are constant and equal.
bool same_group_assignments(mlir::DenseIntElementsAttr attr_a,
                            mlir::DenseIntElementsAttr attr_b) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_allreduce_scatter_optimizationDTcc mht_0(mht_0_v, 211, "", "./tensorflow/dtensor/mlir/dtensor_allreduce_scatter_optimization.cc", "same_group_assignments");

  if (attr_a.getType().getShape() != attr_b.getType().getShape()) {
    return false;
  }
  return std::equal(attr_a.begin(), attr_a.end(), attr_b.begin(), attr_b.end());
}

mlir::DenseIntElementsAttr GetScatterGroupAssignment(
    mlir::TF::DTensorAllScatterOp all_scatter, int scatter_dim) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_allreduce_scatter_optimizationDTcc mht_1(mht_1_v, 222, "", "./tensorflow/dtensor/mlir/dtensor_allreduce_scatter_optimization.cc", "GetScatterGroupAssignment");

  const Layout original_layout = all_scatter.input_layout();
  const Layout desired_layout = all_scatter.output_layout();
  absl::flat_hash_set<std::string> scattered_dims;
  scattered_dims.insert(desired_layout.sharding_spec(scatter_dim));

  auto partitions =
      GetAllReducePartitionsFromReducedDims(original_layout, scattered_dims)
          .ValueOrDie();
  const int32 num_partitions = partitions.size();

  // Construct a flattened list of scatter partitions.
  std::vector<int32> partitions_flat;
  for (auto& p : partitions) {
    partitions_flat.insert(partitions_flat.end(), p.second.begin(),
                           p.second.end());
  }

  int32 partition_size = partitions.begin()->second.size();
  mlir::OpBuilder builder(all_scatter);
  auto group_shaped_type = mlir::RankedTensorType::get(
      {num_partitions, partition_size},
      mlir::IntegerType::get(builder.getContext(), 32));

  return mlir::DenseIntElementsAttr::get(group_shaped_type, partitions_flat);
}

mlir::LogicalResult ApplyOptimization(mlir::func::FuncOp function) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_allreduce_scatter_optimizationDTcc mht_2(mht_2_v, 252, "", "./tensorflow/dtensor/mlir/dtensor_allreduce_scatter_optimization.cc", "ApplyOptimization");

  std::vector<mlir::Operation*> ops_to_delete;
  function.walk([&](mlir::TF::DTensorAllReduceOp all_reduce) {
    if (all_reduce->hasOneUse()) {
      if (auto all_scatter = mlir::dyn_cast<mlir::TF::DTensorAllScatterOp>(
              *all_reduce->getUsers().begin())) {
        VLOG(2) << "Found potential AllReduce+AllScatter to fuse.";
        if (VLOG_IS_ON(2)) all_reduce.dump();
        if (VLOG_IS_ON(2)) all_scatter.dump();

        const Layout original_layout = all_scatter.input_layout();
        const Layout desired_layout = all_scatter.output_layout();

        // Find all potential scatter dimensions.
        std::vector<int> scatter_dims;
        for (int i = 0; i < original_layout.rank(); ++i) {
          if (original_layout.sharding_spec(i) !=
              desired_layout.sharding_spec(i)) {
            scatter_dims.push_back(i);
          }
        }

        if (scatter_dims.empty()) return mlir::WalkResult::advance();
        if (scatter_dims.size() > 1) {
          VLOG(2) << "Multiple dimensions are scatter.  This is unsupported "
                     "for AllReduce+Scatter fusion.";
          return mlir::WalkResult::advance();
        }

        int scatter_dim = scatter_dims[0];
        VLOG(2) << "Scatter_dim: " << scatter_dim;

        // Check that the all-reduce and all-scatter group assignments are the
        // same.
        mlir::DenseIntElementsAttr all_reduce_group_assignment_attr;
        if (!matchPattern(all_reduce.group_assignment(),
                          m_Constant(&all_reduce_group_assignment_attr))) {
          all_reduce.emitOpError("group_assignment should be a constant");
          return mlir::WalkResult::interrupt();
        }

        mlir::DenseIntElementsAttr all_scatter_group_assignment_attr =
            GetScatterGroupAssignment(all_scatter, scatter_dim);

        VLOG(2) << "All scatter group assignment: ";
        if (VLOG_IS_ON(2)) all_scatter_group_assignment_attr.dump();

        bool same_group =
            same_group_assignments(all_reduce_group_assignment_attr,
                                   all_scatter_group_assignment_attr);

        if (!same_group) return mlir::WalkResult::advance();
        VLOG(2) << "Fuse reduce scatter with scatter_dim: " << scatter_dim;

        mlir::OpBuilder builder(all_reduce);
        auto scatter_dim_const_op = builder.create<mlir::TF::ConstOp>(
            all_reduce.getLoc(),
            mlir::DenseIntElementsAttr::get(
                mlir::RankedTensorType::get({}, builder.getI32Type()),
                {scatter_dim}));

        auto reduce_scatter = builder.create<mlir::TF::DTensorReduceScatterOp>(
            all_reduce.getLoc(), all_scatter->getResultTypes(),
            all_reduce.getOperand(0), all_reduce.group_assignment(),
            scatter_dim_const_op, all_reduce.reduce_op(),
            all_reduce.device_type());
        SetSingleLayoutOnOp(reduce_scatter, desired_layout);

        all_scatter->replaceAllUsesWith(reduce_scatter);

        ops_to_delete.push_back(all_scatter);
        ops_to_delete.push_back(all_reduce);
      }
    }
    return mlir::WalkResult::advance();
  });

  for (mlir::Operation* op : ops_to_delete) {
    op->erase();
  }
  return mlir::success();
}

// MLIR pass that combines AllReduce and AllScatter to ReduceScatter.
struct DTensorAllReduceScatterOptimization
    : public DTensorAllReduceScatterOptimizationBase<
          DTensorAllReduceScatterOptimization> {
  void runOnOperation() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_allreduce_scatter_optimizationDTcc mht_3(mht_3_v, 342, "", "./tensorflow/dtensor/mlir/dtensor_allreduce_scatter_optimization.cc", "runOnOperation");

    mlir::func::FuncOp function = getOperation();

    if (mlir::failed(ApplyOptimization(function))) return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorAllReduceScatterOptimization() {
  return std::make_unique<DTensorAllReduceScatterOptimization>();
}

}  // namespace dtensor
}  // namespace tensorflow
