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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSfill_spmd_expanderDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSfill_spmd_expanderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSfill_spmd_expanderDTcc() {
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

#include "tensorflow/dtensor/mlir/expansions/fill_spmd_expander.h"

#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/utils/convert_op_folder.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/dtensor_location.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {

StatusOr<mlir::Operation*> FillSPMDExpander::ExpandOp(mlir::Operation* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSfill_spmd_expanderDTcc mht_0(mht_0_v, 201, "", "./tensorflow/dtensor/mlir/expansions/fill_spmd_expander.cc", "FillSPMDExpander::ExpandOp");

  auto original_fill = mlir::cast<mlir::TF::FillOp>(op);
  TF_ASSIGN_OR_RETURN(auto dims_layout,
                      ExtractLayoutFromOperand(original_fill.dims()));
  if (!dims_layout.has_value()) {
    return errors::InvalidArgument(
        "Failed during SPMD expansion of tf.FillOp. Layout of dimension "
        "input must be known.");
  }

  if (!dims_layout->IsFullyReplicated()) {
    return errors::InvalidArgument(
        "Expected the layout for fill's `dims` argument to be fully "
        "replicated.");
  }
  TF_ASSIGN_OR_RETURN(absl::optional<Layout> output_layout,
                      ExtractSingleLayoutFromOp(op));
  if (!output_layout.has_value())
    return errors::Internal(
        "FillOp doesn't have a layout after layout propagation");
  if (output_layout->IsFullyReplicated()) {
    // For fully replicated layouts the local shape on each device is the same
    // as the global shape.
    return InferSPMDExpandedLocalShape(op);
  }

  // For sharded outputs, the `dims` just needs to be translated from the
  // global to the local shape.
  mlir::OpBuilder builder(op->getBlock(), ++mlir::Block::iterator(op));

  // Create a tensor from the sharding spec, with the dtype of the original
  // attribute.
  auto shard_values = output_layout->num_shards();
  auto int_type = mlir::RankedTensorType::get(
      static_cast<int64>(shard_values.size()), builder.getIntegerType(32));
  auto int_attr = mlir::DenseIntElementsAttr::get(int_type, shard_values);
  auto target_type_attr = mlir::hlo::ConvertElementsAttr(
      int_attr,
      original_fill.dims().getType().cast<mlir::TensorType>().getElementType());

  auto location = DT_LOC(op);
  auto num_shards =
      builder.create<mlir::TF::ConstOp>(location, target_type_attr);
  // Divide the global shape by the sharding spec.
  auto div = builder.create<mlir::TF::DivOp>(location, original_fill.dims(),
                                             num_shards.getResult());
  // Copy over static shape information if available
  auto global_output_type =
      original_fill.getResult().getType().cast<mlir::TensorType>();
  TF_ASSIGN_OR_RETURN(
      auto local_type,
      LocalTypeFromGlobalType(output_layout.value(), global_output_type));

  auto new_fill = builder.create<mlir::TF::FillOp>(
      location, local_type, div.getResult(), original_fill.value());
  original_fill.getResult().replaceAllUsesWith(new_fill.output());
  original_fill.erase();
  return InferSPMDExpandedLocalShape(new_fill);
}

StatusOr<llvm::DenseMap<int, Layout>> FillSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  // Always set replicated layout for output.
  return llvm::DenseMap<int, Layout>(
      {{0, Layout::ReplicatedOnMesh(
               mesh, ValueRank(llvm::cast<mlir::TF::FillOp>(op).output()))}});
}

StatusOr<llvm::DenseMap<int, Layout>> FillSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  // Always set replicated layout for dims / value operand of Fill op.
  return llvm::DenseMap<int, Layout>({{0, Layout::ReplicatedOnMesh(mesh, 1)},
                                      {1, Layout::ReplicatedOnMesh(mesh, 0)}});
}

}  // namespace dtensor
}  // namespace tensorflow
