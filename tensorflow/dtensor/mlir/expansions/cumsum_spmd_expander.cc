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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPScumsum_spmd_expanderDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPScumsum_spmd_expanderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPScumsum_spmd_expanderDTcc() {
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

#include "tensorflow/dtensor/mlir/expansions/cumsum_spmd_expander.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

#include "llvm/Support/FormatVariadic.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/dtensor/cc/dstatus.h"
#include "tensorflow/dtensor/mlir/collectives.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/shape_utils.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {

namespace {

// Extract `axis` tensor from Cumsum op and return it's positive value, since
// it can be a negative index.
StatusOr<int64_t> GetAxisDimension(mlir::Operation* op) {
  auto cumsum = llvm::dyn_cast<mlir::TF::CumsumOp>(op);
  if (cumsum == nullptr) {
    return errors::Internal(
        absl::StrCat("Expected Cumsum op but got : ", OpName(op)).c_str());
  }
  TF_ASSIGN_OR_RETURN(int64_t axis_dim,
                      ExtractConstIntFromValue(cumsum.axis()));
  int64_t tensor_rank = ValueRank(cumsum.x());
  // Axis can be in range [-tensor_rank, tensor_rank), so we add tensor_rank
  // to wrap it around.
  if (axis_dim >= -tensor_rank && axis_dim < 0) {
    axis_dim += tensor_rank;
  } else if (axis_dim < -tensor_rank || axis_dim >= tensor_rank) {
    return errors::InvalidArgument(
        "Invalid axis; expected a value in [-tensor_rank, tensor_rank)");
  }
  return axis_dim;
}

}  // namespace

StatusOr<mlir::Operation*> CumsumSPMDExpander::ExpandOp(mlir::Operation* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPScumsum_spmd_expanderDTcc mht_0(mht_0_v, 230, "", "./tensorflow/dtensor/mlir/expansions/cumsum_spmd_expander.cc", "CumsumSPMDExpander::ExpandOp");

  StatusOr<int64_t> axis_dim = GetAxisDimension(op);
  if (!axis_dim.ok()) return axis_dim.status();

  TF_ASSIGN_OR_RETURN(auto output_layout, ExtractSingleLayoutFromOp(op));
  assert(output_layout);

  // Our intermediate computation layout is the output layout with
  // the axis dimension replicated. So set both the operand and output layout
  // to this intermediate layout.
  Layout intermediate_layout = output_layout->GetLayoutWithReducedDims(
      {axis_dim.ValueOrDie()}, /*keep_dims=*/true);

  // Relayout operand to intermediate layout.
  mlir::OpBuilder builder(op);
  const auto operand = op->getOperand(0);
  TF_ASSIGN_OR_RETURN(auto operand_layout, ExtractLayoutFromOperand(operand));
  if (!operand_layout)
    return errors::InvalidArgument(
        "input layout of Cumsum op must be known before SPMD "
        "expansion.");

  TF_ASSIGN_OR_RETURN(
      const auto new_operand,
      EmitRelayout(operand, operand_layout.value(), intermediate_layout));
  op->setOperand(0, new_operand);

  op = InferSPMDExpandedLocalShape(op);

  // Relayout output to intermediate layout.
  llvm::SmallPtrSet<mlir::Operation*, 4> newly_created_ops;
  builder.setInsertionPointAfter(op);
  TF_ASSIGN_OR_RETURN(auto final_output,
                      EmitRelayout(op->getOpResult(0), intermediate_layout,
                                   output_layout.value(), &newly_created_ops));
  op->getOpResult(0).replaceAllUsesExcept(final_output, newly_created_ops);
  return final_output.getDefiningOp();
}

StatusOr<llvm::DenseMap<int, Layout>> CumsumSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  TF_ASSIGN_OR_RETURN(const auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  TF_ASSIGN_OR_RETURN(int64_t axis_dim, GetAxisDimension(op));

  if (input_layouts.find(0) == input_layouts.end())
    return llvm::DenseMap<int, Layout>();

  auto input_layout = input_layouts.lookup(0);
  return llvm::DenseMap<int, Layout>(
      {{0, input_layout.GetLayoutWithReducedDims({axis_dim},
                                                 /*keep_dims=*/true)}});
}

StatusOr<llvm::DenseMap<int, Layout>> CumsumSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  TF_ASSIGN_OR_RETURN(const auto mesh, ExtractDeviceMeshEnclosingCluster(op));
  TF_ASSIGN_OR_RETURN(int64_t axis_dim, GetAxisDimension(op));

  if (output_layouts.find(0) == output_layouts.end())
    return llvm::DenseMap<int, Layout>();
  auto output_layout = output_layouts.lookup(0);
  return llvm::DenseMap<int, Layout>(
      {{0, output_layout.GetLayoutWithReducedDims({axis_dim},
                                                  /*keep_dims=*/true)}});
}

}  // namespace dtensor
}  // namespace tensorflow
