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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_resource_partitioningDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_resource_partitioningDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_resource_partitioningDTcc() {
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

#include <memory>
#include <tuple>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"

namespace mlir {
namespace TFTPU {
namespace {

constexpr char kReplicateSharding[] = "";

struct TPUResourceReadsWritesPartitioningPass
    : public TF::TPUResourceReadsWritesPartitioningPassBase<
          TPUResourceReadsWritesPartitioningPass> {
  void runOnOperation() override;
};

bool AllResourceTypesHaveSubtypes(TypeRange resources) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_resource_partitioningDTcc mht_0(mht_0_v, 214, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_resource_partitioning.cc", "AllResourceTypesHaveSubtypes");

  for (Type resource : resources)
    if (!llvm::hasSingleElement(resource.cast<TensorType>()
                                    .getElementType()
                                    .cast<TF::ResourceType>()
                                    .getSubtypes()))
      return false;

  return true;
}

Type GetResourceSubtype(Type type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_resource_partitioningDTcc mht_1(mht_1_v, 228, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_resource_partitioning.cc", "GetResourceSubtype");

  return type.cast<TensorType>()
      .getElementType()
      .cast<TF::ResourceType>()
      .getSubtypes()
      .front();
}

Type GetResourceSubtype(Value resource) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_resource_partitioningDTcc mht_2(mht_2_v, 239, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_resource_partitioning.cc", "GetResourceSubtype");

  return GetResourceSubtype(resource.getType());
}

// Rewrites unpartitioned resource reads and writes to partitioned resource
// reads and writes. The TPU computation from the frontend is generated in such
// a way that resource operations operate on the unpartitioned resource handle
// (from a `tf.TPUReplicatedInput`). This results in resource reads and writes
// on the unpartitioned resource handle post resource op decomposition/lifting.
// Here the unpartitioned resource read and write is expanded to individual
// resource reads and writes per associated partitioned resource handle.
void PartitionResourceReadsWrites(tf_device::ClusterFuncOp cluster_func) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_resource_partitioningDTcc mht_3(mht_3_v, 253, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_resource_partitioning.cc", "PartitionResourceReadsWrites");

  bool use_spmd = false;
  if (auto use_spmd_attr = cluster_func->getAttrOfType<BoolAttr>(
          "use_spmd_for_xla_partitioning"))
    use_spmd = use_spmd_attr.getValue();

  if (!use_spmd) return;

  OpBuilder builder(cluster_func);
  // Rewrite results before rewriting operands as `tf.TPUPartitionedInput`
  // resource handle results is an indicator for a partitioned resource
  // variable. These `tf.TPUPartitionedInput` will be removed when rewriting
  // the operands.
  for (Value result : cluster_func.results()) {
    if (!result.hasOneUse()) continue;
    auto assign_var =
        llvm::dyn_cast<TF::AssignVariableOp>(*result.getUsers().begin());
    if (!assign_var || assign_var.value() != result) continue;
    auto partitioned_input = llvm::dyn_cast_or_null<TF::TPUPartitionedInputOp>(
        assign_var.resource().getDefiningOp());
    if (!partitioned_input ||
        !AllResourceTypesHaveSubtypes(partitioned_input.inputs().getTypes()))
      continue;

    builder.setInsertionPoint(assign_var);
    llvm::SmallVector<Type, 4> partitioned_output_types;
    partitioned_output_types.reserve(partitioned_input.N());
    for (Type input_type : partitioned_input.inputs().getTypes())
      partitioned_output_types.push_back(GetResourceSubtype(input_type));
    auto partitioned_output = builder.create<TF::TPUPartitionedOutputOp>(
        cluster_func->getLoc(), partitioned_output_types, result,
        partitioned_input.partition_dimAttr(),
        partitioned_input._XlaShardingAttr());
    for (auto resource_write :
         llvm::zip(partitioned_input.inputs(), partitioned_output.output()))
      builder.create<TF::AssignVariableOp>(
          assign_var->getLoc(), /*resource=*/std::get<0>(resource_write),
          /*value=*/std::get<1>(resource_write));
    assign_var.erase();
  }

  for (OpOperand& operand : cluster_func->getOpOperands()) {
    auto read_var = llvm::dyn_cast_or_null<TF::ReadVariableOp>(
        operand.get().getDefiningOp());
    if (!read_var || !read_var.value().hasOneUse()) continue;
    auto partitioned_input = llvm::dyn_cast_or_null<TF::TPUPartitionedInputOp>(
        read_var.resource().getDefiningOp());
    if (!partitioned_input ||
        !AllResourceTypesHaveSubtypes(partitioned_input.inputs().getTypes()))
      continue;

    builder.setInsertionPoint(partitioned_input);
    llvm::SmallVector<Value, 4> partitioned_reads;
    for (Value input : partitioned_input.inputs()) {
      auto partitioned_read = builder.create<TF::ReadVariableOp>(
          read_var->getLoc(), GetResourceSubtype(input), input);
      partitioned_reads.push_back(partitioned_read.value());
    }
    auto partitioned_read = builder.create<TF::TPUPartitionedInputOp>(
        partitioned_input->getLoc(), read_var.value().getType(),
        partitioned_reads, partitioned_input.partition_dimAttr(),
        partitioned_input._XlaShardingAttr());
    operand.set(partitioned_read);
    read_var->erase();
    if (partitioned_input->use_empty()) partitioned_input->erase();
  }
}

void TPUResourceReadsWritesPartitioningPass::runOnOperation() {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStpu_resource_partitioningDTcc mht_4(mht_4_v, 324, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tpu_resource_partitioning.cc", "TPUResourceReadsWritesPartitioningPass::runOnOperation");

  llvm::SmallVector<tf_device::ClusterFuncOp, 4> cluster_funcs;
  getOperation()->walk([&cluster_funcs](tf_device::ClusterFuncOp cluster_func) {
    cluster_funcs.push_back(cluster_func);
  });
  for (tf_device::ClusterFuncOp cluster_func : cluster_funcs)
    PartitionResourceReadsWrites(cluster_func);
}

}  // namespace

std::unique_ptr<OperationPass<FuncOp>>
CreateTPUResourceReadsWritesPartitioningPass() {
  return std::make_unique<TPUResourceReadsWritesPartitioningPass>();
}

}  // namespace TFTPU
}  // namespace mlir
