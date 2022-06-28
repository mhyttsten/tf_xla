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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSelementwise_spmd_expanderDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSelementwise_spmd_expanderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSelementwise_spmd_expanderDTcc() {
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

#include "tensorflow/dtensor/mlir/expansions/elementwise_spmd_expander.h"

#include <iterator>
#include <string>
#include <utility>

#include "absl/strings/str_join.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/UseDefLists.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/utils/array_container_utils.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {
namespace {

StatusOr<llvm::SmallVector<int64_t, 4>> GetShape(mlir::Value value) {
  auto type = value.getType().dyn_cast<mlir::RankedTensorType>();
  if (!type)
    return errors::InvalidArgument(
        "Rank of input values must be statically known.");

  const auto shape = type.getShape();
  return llvm::SmallVector<int64_t, 4>{shape.begin(), shape.end()};
}

}  // namespace

StatusOr<mlir::Operation*> ElementwiseSPMDExpander::ExpandOp(
    mlir::Operation* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSelementwise_spmd_expanderDTcc mht_0(mht_0_v, 232, "", "./tensorflow/dtensor/mlir/expansions/elementwise_spmd_expander.cc", "ElementwiseSPMDExpander::ExpandOp");

  TF_ASSIGN_OR_RETURN(auto output_layout, ExtractSingleLayoutFromOp(op));
  assert(output_layout);

  mlir::OpBuilder builder(op);

  for (auto& operand : op->getOpOperands()) {
    // Verify that all output dimensions (including the dimensions added by
    // broadcasting) is more sharded then the correspdonding layout
    // configuration of the same dimension of every operands.
    TF_ASSIGN_OR_RETURN(auto operand_layout,
                        ExtractLayoutFromOperand(operand.get()));
    if (!operand_layout)
      return errors::InvalidArgument(
          "input layout of elementwise op must be known before SPMD "
          "expansion.");

    // For scalar operands, splitting is not needed.
    if (operand_layout->rank() == 0) continue;

    llvm::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;

    // Note that due to broacasting, inputs and output tensors are aligned to
    // the right. Therefore, we truncate the output layout.
    const int rank_offset = output_layout->rank() - operand_layout->rank();

    // Get the desired layout for this operand.
    //
    // - Truncate: It should be the same layout as the output, truncated to
    //   take into account broadcasting on the rank and removing sharding in
    //   dimensions where the operand has size 1 (where we have broadcasting on
    //   the dimension size).
    //
    // - Relayout: If the output and operand have different sharding spec, we
    //   adjust the operands. For example, if operand is 'z,*' and output is
    //   '*.y', relayout operand to conform output. This means the SPMD safer
    //   and easeier. In future, we might do certain optimization to save FLops.
    //   For example, if all operands are 'x,y' and output is '*,*', relayouting
    //   output could be the choice (saving communications).
    auto truncated_layout = output_layout->Truncate(rank_offset, /*end=*/true);
    mlir::Value output;
    TF_ASSIGN_OR_RETURN(const auto& shape, ExtractGlobalInputShape(operand));
    absl::flat_hash_set<int> size_one_dims;
    for (int i = 0; i < shape.size(); ++i)
      if (shape[i] == 1) size_one_dims.emplace(i);
    truncated_layout = truncated_layout.GetLayoutWithReducedDims(
        size_one_dims, /*keep_dims=*/true);
    TF_ASSIGN_OR_RETURN(
        output, EmitRelayout(operand.get(), *operand_layout, truncated_layout));
    operand.set(output);
  }

  // For element-wise op SPMD expansion, given that operand layouts are
  // compatible to op's layout, op can simply be executed without any changes.
  return InferSPMDExpandedLocalShape(op);
}

// Computes output layouts of elementwise operation using broadcast logic for
// operands.
StatusOr<llvm::DenseMap<int, Layout>>
ElementwiseSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(absl::optional<Layout> merged_operand_layout,
                      GetMergedOperandLayout(input_layouts, op));

  if (merged_operand_layout) {
    const int output_rank = ValueRank(op->getOpResult(0));
    if (output_rank == -1)
      return errors::InvalidArgument("Output has unknown rank");

    // We assume that all elementwise operations output a single tensor.
    return llvm::DenseMap<int, Layout>(
        {{0, merged_operand_layout->LeftPad(output_rank)}});
  }
  return llvm::DenseMap<int, Layout>();
}

// Computes input layouts of elementwise operation using broadcast logic for
// operands.
StatusOr<llvm::DenseMap<int, Layout>>
ElementwiseSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  // Do not infer any operand layout if no output layout is present.
  if (output_layouts.find(0) == output_layouts.end())
    return llvm::DenseMap<int, Layout>();

  Layout output_layout = output_layouts.lookup(0);
  llvm::DenseMap<int, Layout> input_layouts;

  for (const auto& operand_and_index : llvm::enumerate(op->getOperands())) {
    const int operand_index = operand_and_index.index();
    auto operand = operand_and_index.value();

    TF_ASSIGN_OR_RETURN(auto operand_shape, GetShape(operand));
    auto inferred_operand_layout_strs =
        output_layout
            .Truncate(
                output_layout.sharding_specs().size() - operand_shape.size(),
                /*end=*/true)
            .sharding_spec_strs();

    if (inferred_operand_layout_strs.size() != operand_shape.size())
      return errors::FailedPrecondition(
          "Mismatch of operand shape size and layout size.");
    for (const auto& dim_shape_and_index : llvm::enumerate(operand_shape)) {
      const int dim_index = dim_shape_and_index.index();
      const int dim_shape = dim_shape_and_index.value();
      if (dim_shape <= 1) {
        inferred_operand_layout_strs[dim_index] = Layout::kUnshardedDim;
      }
    }
    TF_ASSIGN_OR_RETURN(
        auto inferred_operand_layout,
        Layout::GetLayout(inferred_operand_layout_strs, output_layout.mesh()));
    input_layouts[operand_index] = inferred_operand_layout;
  }
  return input_layouts;
}

}  // namespace dtensor
}  // namespace tensorflow
