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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_mixed_precision_reduceDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_mixed_precision_reduceDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_mixed_precision_reduceDTcc() {
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

#include "absl/strings/string_view.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/dtensor/cc/dtensor_utils.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes_classes.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"

namespace tensorflow {
namespace dtensor {
namespace {

// Extracts the reduction group size from the group_assignment operand of the
// reduce op. group_assignment is a 2-dimensional array where each element is
// the list of devices that are a part of the same reduction group.
template <class ReduceOpType>
mlir::LogicalResult GetAllReduceGroupSize(ReduceOpType reduce_op,
                                          int32* group_size) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_mixed_precision_reduceDTcc mht_0(mht_0_v, 210, "", "./tensorflow/dtensor/mlir/dtensor_mixed_precision_reduce.cc", "GetAllReduceGroupSize");

  mlir::DenseIntElementsAttr group_assignment_attr;
  if (!matchPattern(reduce_op.group_assignment(),
                    m_Constant(&group_assignment_attr)))
    return mlir::emitError(reduce_op.getLoc(),
                           "group_assigment must be a constant.");
  if (group_assignment_attr.getType().getRank() != 2)
    return mlir::emitError(reduce_op.getLoc(),
                           "group_assignment should have two dimensions.");

  *group_size = group_assignment_attr.getType().getShape()[1];
  return mlir::success();
}

// For large enough reduction groups, we compute reductions in a higher
// precision type to ensure accuracy is not lost with sequential addition
// of large numbers in a lower precision type. If the given reduce op meets the
// following criteria:
//   - the tensors being reduced are of type bfloat16,
//   - the reduction group is at least as large as the configurable env var
//     DTENSOR_REDUCE_IN_BFLOAT16_MAX_GROUP_SIZE,
// then the tensors are upcasted to float32 for the reduction before being
// downcasted again.
template <class ReduceOpType>
mlir::LogicalResult MaybeUpcastForReduction(ReduceOpType reduce_op,
                                            bool* changed) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_mixed_precision_reduceDTcc mht_1(mht_1_v, 238, "", "./tensorflow/dtensor/mlir/dtensor_mixed_precision_reduce.cc", "MaybeUpcastForReduction");

  const mlir::RankedTensorType& input_type =
      reduce_op.input().getType().template dyn_cast<mlir::RankedTensorType>();
  if (!input_type.getElementType().isBF16()) {
    // Upcast only applies for bfloat16 input.
    return mlir::success();
  }

  mlir::OpBuilder builder(reduce_op);
  const mlir::Location loc = reduce_op.getLoc();

  int32 group_size;
  if (mlir::failed(GetAllReduceGroupSize(reduce_op, &group_size)))
    return mlir::failure();
  if (group_size <= ReduceInBfloat16MaxGroupSize())
    // Reduce group size is not sufficient, so we do not modify the ops.
    return mlir::success();

  const auto reduce_layout = ExtractRequiredSingleLayoutFromOp(reduce_op);
  if (!reduce_layout.ok())
    return reduce_op.emitOpError(llvm::formatv(
        "Malformed layout specification for DTensor reduce op found: {0}",
        reduce_layout.status().error_message()));

  // The original output tensor type that would have been used by all users of
  // the reduce op.
  const mlir::RankedTensorType& output_type =
      reduce_op.output().getType().template dyn_cast<mlir::RankedTensorType>();

  mlir::TF::CastOp upcast = builder.create<mlir::TF::CastOp>(
      loc,
      mlir::RankedTensorType::get(input_type.getShape(), builder.getF32Type()),
      reduce_op.input());
  reduce_op->setOperand(0, upcast.y());
  reduce_op.output().setType(upcast.y().getType());

  builder.setInsertionPointAfter(reduce_op);
  mlir::TF::CastOp downcast = builder.create<mlir::TF::CastOp>(
      loc,
      mlir::RankedTensorType::get(output_type.getShape(),
                                  output_type.getElementType()),
      reduce_op);
  // Match the layout of the downcast with the reduce op, this is required for
  // the later passes.
  SetSingleLayoutOnOp(downcast, *reduce_layout);
  reduce_op.output().replaceAllUsesExcept(downcast.y(), downcast);

  *changed = true;
  return mlir::success();
}

template <class ReduceOpType>
mlir::LogicalResult TryMixedPrecisionReduce(mlir::func::FuncOp function,
                                            absl::string_view opName) {
   std::vector<std::string> mht_2_v;
   mht_2_v.push_back("opName: \"" + std::string(opName.data(), opName.size()) + "\"");
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_mixed_precision_reduceDTcc mht_2(mht_2_v, 295, "", "./tensorflow/dtensor/mlir/dtensor_mixed_precision_reduce.cc", "TryMixedPrecisionReduce");

  int32_t reduceOpsCounter = 0;
  int32_t changedReduceOpsCounter = 0;

  mlir::WalkResult walk_result = function.walk([&](ReduceOpType reduce_op) {
    if (reduce_op.reduce_op().str() == kReduceOpAdd) {
      reduceOpsCounter += 1;
      bool changed = false;
      if (mlir::failed(MaybeUpcastForReduction(reduce_op, &changed)))
        return mlir::WalkResult::interrupt();
      if (changed) changedReduceOpsCounter += 1;
    }
    return mlir::WalkResult::advance();
  });
  if (walk_result.wasInterrupted()) return mlir::failure();

  VLOG(2) << "Applied mixed precision to " << changedReduceOpsCounter << " of "
          << reduceOpsCounter << " Add " << opName << " ops.";

  return mlir::success();
}

// MLIR pass that enables tensor upcasting within mixed-precision reduction.
struct DTensorMixedPrecisionReducePass
    : public DTensorMixedPrecisionReduceBase<DTensorMixedPrecisionReducePass> {
  void runOnOperation() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSdtensor_mixed_precision_reduceDTcc mht_3(mht_3_v, 323, "", "./tensorflow/dtensor/mlir/dtensor_mixed_precision_reduce.cc", "runOnOperation");

    mlir::func::FuncOp function = getOperation();

    if (mlir::failed(TryMixedPrecisionReduce<mlir::TF::DTensorAllReduceOp>(
            function, "DTensorAllReduce")))
      return signalPassFailure();
    if (mlir::failed(TryMixedPrecisionReduce<mlir::TF::DTensorReduceScatterOp>(
            function, "DTensorReduceScatter")))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorMixedPrecisionReducePass() {
  return std::make_unique<DTensorMixedPrecisionReducePass>();
}

}  // namespace dtensor
}  // namespace tensorflow
