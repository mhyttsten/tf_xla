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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSqr_spmd_expanderDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSqr_spmd_expanderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSqr_spmd_expanderDTcc() {
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

#include "tensorflow/dtensor/mlir/expansions/qr_spmd_expander.h"

#include <algorithm>

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"
#include "tensorflow/dtensor/proto/layout.pb.h"

namespace tensorflow {
namespace dtensor {

StatusOr<mlir::Operation*> QRSPMDExpander::ExpandOp(mlir::Operation* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSqr_spmd_expanderDTcc mht_0(mht_0_v, 211, "", "./tensorflow/dtensor/mlir/expansions/qr_spmd_expander.cc", "QRSPMDExpander::ExpandOp");

  TF_ASSIGN_OR_RETURN(const auto output_layouts,
                      ExtractRequiredLayoutFromOp(op));
  TF_ASSIGN_OR_RETURN(const auto operand_layouts,
                      ExtractRequiredLayoutFromOperands(op));

  // Relayout all layouts to the first output layout with the last two
  // dimensions replicated. We can do more optimization but this is fine
  Layout new_layout =
      output_layouts[0].GetLayoutWithReducedDims({-1, -2}, /*keep_dims=*/true);

  TF_ASSIGN_OR_RETURN(
      const auto new_operand,
      EmitRelayout(op->getOperand(0), operand_layouts[0], new_layout));
  op->setOperand(0, new_operand);

  mlir::OpBuilder builder(op);
  op = InferSPMDExpandedLocalShape(op);

  llvm::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;
  llvm::SmallVector<mlir::Value, 4> generated_outputs;
  llvm::SmallVector<mlir::Type, 4> generated_types;

  // Relayout outputs
  for (auto i = 0; i < output_layouts.size(); i++) {
    TF_ASSIGN_OR_RETURN(auto new_output,
                        EmitRelayout(op->getOpResult(i), new_layout,
                                     output_layouts[i], &newly_created_ops));
    generated_outputs.emplace_back(new_output);
    generated_types.emplace_back(new_output.getType());
  }
  if (generated_outputs[0].getDefiningOp()->isBeforeInBlock(
          generated_outputs[1].getDefiningOp()))
    builder.setInsertionPointAfterValue(generated_outputs[1]);
  else
    builder.setInsertionPointAfterValue(generated_outputs[0]);

  // Tie the two outputs together with an identity op
  auto identity_op = builder.create<mlir::TF::IdentityNOp>(
      op->getLoc(), generated_types, generated_outputs);
  newly_created_ops.insert(identity_op);
  for (int i = 0; i < output_layouts.size(); i++) {
    op->getOpResult(i).replaceAllUsesExcept(identity_op.getResult(i),
                                            newly_created_ops);
  }
  return identity_op.getOperation();
}

StatusOr<llvm::DenseMap<int, Layout>> QRSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  // If no input layout at index 0 is present then do not infer any output
  // layout.
  if (input_layouts.find(0) == input_layouts.end())
    return llvm::DenseMap<int, Layout>();

  // Set the output layouts as the copy of the input layouts with the last 2
  // dimensions replicated.
  Layout output_layout = input_layouts.lookup(0).GetLayoutWithReducedDims(
      {-1, -2}, /*keep_dims=*/true);
  return llvm::DenseMap<int, Layout>({{0, output_layout}, {1, output_layout}});
}

StatusOr<llvm::DenseMap<int, Layout>> QRSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  // If no output layout is present then do not infer any operand layouts.
  if (output_layouts.empty()) return llvm::DenseMap<int, Layout>();

  // Set the operand layout as the copy of the output layouts with the last 2
  // dimensions replicated.
  Layout layout = output_layouts.find(0) != output_layouts.end()
                      ? output_layouts.lookup(0)
                      : output_layouts.lookup(1);
  return llvm::DenseMap<int, Layout>(
      {{0, layout.GetLayoutWithReducedDims({-1, -2}, /*keep_dims=*/true)}});
}

}  // namespace dtensor
}  // namespace tensorflow
