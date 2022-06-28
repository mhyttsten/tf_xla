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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPScontrol_flow_spmd_expanderDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPScontrol_flow_spmd_expanderDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPScontrol_flow_spmd_expanderDTcc() {
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

#include "tensorflow/dtensor/mlir/expansions/control_flow_spmd_expander.h"

namespace tensorflow {
namespace dtensor {

StatusOr<mlir::Operation*> WhileRegionSPMDExpander::ExpandOp(
    mlir::Operation* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPScontrol_flow_spmd_expanderDTcc mht_0(mht_0_v, 191, "", "./tensorflow/dtensor/mlir/expansions/control_flow_spmd_expander.cc", "WhileRegionSPMDExpander::ExpandOp");

  assert(op->getNumOperands() == op->getNumResults());
  // Set the type for the results of the WhileRegion explicitly.
  //
  // Normally we would use InferSPMDExpandedLocalShape for this, but that
  // function requires the op to either have a type inference interface
  // (which WhileRegion does not) or a TensorFlow ShapeFn (WhileRegion is not
  // a TensorFlow op). During the standard MLIR shape inference pass this op
  // is handled by a special case in InferShapeForSingleOperation.
  for (int i = 0; i < op->getNumOperands(); ++i)
    op->getResult(i).setType(op->getOperand(i).getType());

  auto while_op = llvm::cast<mlir::TF::WhileRegionOp>(op);
  for (const auto& data :
       llvm::enumerate(llvm::zip(while_op.cond().front().getArguments(),
                                 while_op.body().front().getArguments()))) {
    const int index = data.index();
    mlir::BlockArgument cond_arg = std::get<0>(data.value());
    mlir::BlockArgument body_arg = std::get<1>(data.value());
    cond_arg.setType(while_op.getOperand(index).getType());
    body_arg.setType(while_op.getOperand(index).getType());
  }

  return op;
}

StatusOr<llvm::DenseMap<int, Layout>>
WhileRegionSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  return errors::Unimplemented(
      "WhileRegion does not support compute layouts. This should not be "
      "called.");
}

StatusOr<llvm::DenseMap<int, Layout>>
WhileRegionSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  return errors::Unimplemented(
      "WhileRegion does not support compute layouts. This should not be "
      "called.");
}

StatusOr<mlir::Operation*> IfRegionSPMDExpander::ExpandOp(mlir::Operation* op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSexpansionsPScontrol_flow_spmd_expanderDTcc mht_1(mht_1_v, 236, "", "./tensorflow/dtensor/mlir/expansions/control_flow_spmd_expander.cc", "IfRegionSPMDExpander::ExpandOp");

  auto if_op = llvm::cast<mlir::TF::IfRegionOp>(op);
  for (mlir::Value result : if_op->getResults()) {
    auto result_layout_op = llvm::dyn_cast_or_null<mlir::TF::DTensorLayout>(
        *result.getUsers().begin());
    if (!result_layout_op)
      return errors::InvalidArgument(
          "Missing layout of If op result during SPMD expansion.");

    const Layout layout = result_layout_op.layout();
    if (!layout.IsFullyReplicated()) {
      const auto global_shape = result_layout_op.global_shape();
      if (!global_shape)
        return errors::InvalidArgument(
            "Shape of If op must be statically known for SPMD expansion.");

      result.setType(mlir::RankedTensorType::get(
          layout.LocalShapeFromGlobalShape(*global_shape),
          result.getType().cast<mlir::TensorType>().getElementType()));
    }
  }
  return op;
}

StatusOr<llvm::DenseMap<int, Layout>>
IfRegionSPMDExpander::ComputeLayoutForward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& input_layouts) {
  // No-op for forward propagation.
  return llvm::DenseMap<int, Layout>();
}

StatusOr<llvm::DenseMap<int, Layout>>
IfRegionSPMDExpander::ComputeLayoutBackward(
    mlir::Operation* op, const llvm::DenseMap<int, Layout>& output_layouts) {
  // Layout propagation for for TF::IfRegion op is no-op. Actual layout
  // propagation logic depends on layout propgation of ops inside the
  // then/else regions of the IfRegion op.
  TF_ASSIGN_OR_RETURN(const Mesh mesh, ExtractDeviceMeshEnclosingCluster(op));
  return llvm::DenseMap<int, Layout>(
      {{0, Layout::ReplicatedOnMesh(mesh, ValueRank(op->getOperand(0)))}});
}

}  // namespace dtensor
}  // namespace tensorflow
