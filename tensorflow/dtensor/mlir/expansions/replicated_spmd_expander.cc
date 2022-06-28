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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSreplicated_spmd_expanderDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSreplicated_spmd_expanderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSreplicated_spmd_expanderDTcc() {
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

#include "tensorflow/dtensor/mlir/expansions/replicated_spmd_expander.h"

#include <algorithm>

#include "llvm/Support/FormatVariadic.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {

namespace {

// Checks that all layouts are fully replicated
bool AllReplicated(const std::vector<Layout>& layouts) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSreplicated_spmd_expanderDTcc mht_0(mht_0_v, 202, "", "./tensorflow/dtensor/mlir/expansions/replicated_spmd_expander.cc", "AllReplicated");

  for (const auto& layout : layouts) {
    if (!layout.IsFullyReplicated()) return false;
  }
  return true;
}

}  // namespace

// Relayout all operands and outputs to replicated layout.
StatusOr<mlir::Operation*>
ReplicatedOpSPMDExpander::ReplicatedRelayoutOperandsAndOutputs(
    mlir::Operation* op, const std::vector<Layout>& operand_layouts,
    const std::vector<Layout>& output_layouts) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSreplicated_spmd_expanderDTcc mht_1(mht_1_v, 218, "", "./tensorflow/dtensor/mlir/expansions/replicated_spmd_expander.cc", "ReplicatedOpSPMDExpander::ReplicatedRelayoutOperandsAndOutputs");

  TF_ASSIGN_OR_RETURN(const auto mesh, ExtractDeviceMeshEnclosingCluster(op));

  // Relayout operands
  for (auto i = 0; i < operand_layouts.size(); ++i) {
    Layout new_layout =
        Layout::ReplicatedOnMesh(mesh, ValueRank(op->getOperand(i)));
    TF_ASSIGN_OR_RETURN(
        const auto new_operand,
        EmitRelayout(op->getOperand(i), operand_layouts[i], new_layout));
    op->setOperand(i, new_operand);
  }
  // Expand to local shape
  op = InferSPMDExpandedLocalShape(op);

  llvm::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;
  llvm::SmallVector<mlir::Value, 4> generated_outputs;
  llvm::SmallVector<mlir::Type, 4> generated_types;

  // Track the op that comes last after splitting.
  mlir::Operation* last_op_after_splitting = op;

  // Relayout outputs
  for (auto i = 0; i < output_layouts.size(); ++i) {
    Layout new_layout =
        Layout::ReplicatedOnMesh(mesh, ValueRank(op->getResult(i)));
    TF_ASSIGN_OR_RETURN(auto new_output,
                        EmitRelayout(op->getOpResult(i), new_layout,
                                     output_layouts[i], &newly_created_ops));
    generated_outputs.emplace_back(new_output);
    generated_types.emplace_back(new_output.getType());
    if (last_op_after_splitting->isBeforeInBlock(new_output.getDefiningOp())) {
      last_op_after_splitting = new_output.getDefiningOp();
    }
  }
  mlir::OpBuilder builder(op);
  builder.setInsertionPointAfter(last_op_after_splitting);

  // Tie all outputs together with identity_n
  auto identity_op = builder.create<mlir::TF::IdentityNOp>(
      op->getLoc(), generated_types, generated_outputs);
  newly_created_ops.insert(identity_op);
  for (int i = 0; i < output_layouts.size(); ++i) {
    op->getOpResult(i).replaceAllUsesExcept(identity_op.getResult(i),
                                            newly_created_ops);
  }

  return identity_op.getOperation();
}

StatusOr<mlir::Operation*> ReplicatedOpSPMDExpander::ExpandOp(
    mlir::Operation* op) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPSreplicated_spmd_expanderDTcc mht_2(mht_2_v, 272, "", "./tensorflow/dtensor/mlir/expansions/replicated_spmd_expander.cc", "ReplicatedOpSPMDExpander::ExpandOp");

  TF_ASSIGN_OR_RETURN(const auto output_layouts,
                      ExtractRequiredLayoutFromOp(op));
  TF_ASSIGN_OR_RETURN(const auto operand_layouts,
                      ExtractRequiredLayoutFromOperands(op));
  if (relayout_when_sharded_)
    return ReplicatedRelayoutOperandsAndOutputs(op, operand_layouts,
                                                output_layouts);
  if (!AllReplicated(output_layouts) || !AllReplicated(operand_layouts)) {
    return errors::InvalidArgument(
        llvm::formatv("Expecting {0} to have input and output layouts to be "
                      "fully replicated but was not. ",
                      OpName(op))
            .str());
  }
  return op;
}

// Always return a set of replicated layouts.
StatusOr<llvm::DenseMap<int, Layout>>
ReplicatedOpSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(const auto mesh, ExtractDeviceMeshEnclosingCluster(op));

  llvm::DenseMap<int, Layout> output_layouts(op->getNumResults());
  for (int i = 0; i < op->getNumResults(); ++i) {
    output_layouts[i] =
        Layout::ReplicatedOnMesh(mesh, ValueRank(op->getResult(i)));
  }
  return output_layouts;
}

// Always return a set of replicated layouts.
StatusOr<llvm::DenseMap<int, Layout>>
ReplicatedOpSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(const auto mesh, ExtractDeviceMeshEnclosingCluster(op));

  llvm::DenseMap<int, Layout> input_layouts(op->getNumOperands());
  for (int i = 0; i < op->getNumOperands(); ++i) {
    input_layouts[i] =
        Layout::ReplicatedOnMesh(mesh, ValueRank(op->getOperand(i)));
  }
  return input_layouts;
}

}  // namespace dtensor
}  // namespace tensorflow
