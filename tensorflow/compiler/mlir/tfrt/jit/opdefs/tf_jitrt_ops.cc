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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPSopdefsPStf_jitrt_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPSopdefsPStf_jitrt_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPSopdefsPStf_jitrt_opsDTcc() {
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

/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tfrt/jit/opdefs/tf_jitrt_ops.h"

#include <algorithm>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Transforms/InliningUtils.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"
#include "tfrt/basic_kernels/opdefs/types.h"  // from @tf_runtime

namespace mlir {
namespace tf_jitrt {

//===----------------------------------------------------------------------===//
// JitRuntimeDialect Interfaces
//===----------------------------------------------------------------------===//

namespace {
// Operations in the `tf_jitrt` dialect are always safe to inline because they
// are pure compute operations.
struct JitRuntimeInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  bool isLegalToInline(Operation*, Operation*, bool) const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPSopdefsPStf_jitrt_opsDTcc mht_0(mht_0_v, 214, "", "./tensorflow/compiler/mlir/tfrt/jit/opdefs/tf_jitrt_ops.cc", "isLegalToInline");

    assert(false && "tf_jitrt doesn't have callable operations");
    return true;
  }

  bool isLegalToInline(Region*, Region*, bool,
                       BlockAndValueMapping&) const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPSopdefsPStf_jitrt_opsDTcc mht_1(mht_1_v, 223, "", "./tensorflow/compiler/mlir/tfrt/jit/opdefs/tf_jitrt_ops.cc", "isLegalToInline");

    return true;
  }

  bool isLegalToInline(Operation*, Region*, bool,
                       BlockAndValueMapping&) const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPSopdefsPStf_jitrt_opsDTcc mht_2(mht_2_v, 231, "", "./tensorflow/compiler/mlir/tfrt/jit/opdefs/tf_jitrt_ops.cc", "isLegalToInline");

    return true;
  }
};
}  // namespace

//===----------------------------------------------------------------------===//
// JitRuntimeDialect Dialect
//===----------------------------------------------------------------------===//

JitRuntimeDialect::JitRuntimeDialect(mlir::MLIRContext* context)
    : Dialect(/*name*/ "tf_jitrt", context,
              mlir::TypeID::get<JitRuntimeDialect>()) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPSopdefsPStf_jitrt_opsDTcc mht_3(mht_3_v, 246, "", "./tensorflow/compiler/mlir/tfrt/jit/opdefs/tf_jitrt_ops.cc", "JitRuntimeDialect::JitRuntimeDialect");

  addInterfaces<JitRuntimeInlinerInterface>();
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/tfrt/tf_jitrt_ops.cc.inc"
      >();
}

// Computes the number of elements in the tensor type. Optimistically use `1` as
// a size of all unknown dimensions. These heuristics match cost estimates of
// the fallback_async::ExecuteOp operations.
static int64_t GetRankedTensorSize(TensorType tensor) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPSopdefsPStf_jitrt_opsDTcc mht_4(mht_4_v, 260, "", "./tensorflow/compiler/mlir/tfrt/jit/opdefs/tf_jitrt_ops.cc", "GetRankedTensorSize");

  assert(tensor.hasRank() && "shape must be ranked");
  if (!tensor.hasRank()) return 0;

  int64_t size = 1;  // scalars (rank 0) have size 1
  for (int64_t dim : tensor.getShape()) size *= std::max<int64_t>(1, dim);
  return size;
}

int64_t GetMaxArgSize(mlir::func::FuncOp func) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPSopdefsPStf_jitrt_opsDTcc mht_5(mht_5_v, 272, "", "./tensorflow/compiler/mlir/tfrt/jit/opdefs/tf_jitrt_ops.cc", "GetMaxArgSize");

  int64_t max_arg_size = 1;
  for (BlockArgument& arg : func.getArguments()) {
    auto type = arg.getType().cast<mlir::TensorType>();
    if (type.hasRank())
      max_arg_size = std::max(max_arg_size, GetRankedTensorSize(type));
  }
  return max_arg_size;
}

int64_t FallbackExecuteOp::cost() {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPSjitPSopdefsPStf_jitrt_opsDTcc mht_6(mht_6_v, 285, "", "./tensorflow/compiler/mlir/tfrt/jit/opdefs/tf_jitrt_ops.cc", "FallbackExecuteOp::cost");

  Operation* self = getOperation();

  // Find the referenced kernel function.
  auto kernel_fn = SymbolTable::lookupNearestSymbolFrom<FuncOp>(self, kernel());
  if (!kernel_fn) return 1;

  int64_t cost = 0;

  // Compute the max argument size, which we will assign to unranked inputs
  // just like TFRT's cost model does.
  int64_t max_arg_size = GetMaxArgSize(kernel_fn);

  // Maybe override max argument size with explicit value passed via attribute.
  auto module = kernel_fn->getParentOfType<mlir::ModuleOp>();
  if (auto attr = module->getAttrOfType<IntegerAttr>("tfrt.max-arg-size"))
    max_arg_size = attr.getValue().getSExtValue();

  // Get the sum of sizes of all ranked inputs for all operations in the
  // function body. This approach approximates the cost analysis in the
  // tfrt_compiler::CostAnalysis, because initially we want to get identical
  // stream assignments, however long term we want to use more precise cost
  // estimation, together with a more precise stream assignment.
  //
  // TODO(ezhulenev): Once we have a proper cost model for MLIR operations,
  // use it to compute a more precise cost estimation.
  for (mlir::Operation& op : kernel_fn.getBody().getOps()) {
    // Skip return operation.
    if (mlir::isa<mlir::func::ReturnOp>(op)) continue;

    // These ops are cheap regardless of their input sizes.
    if (mlir::isa<mlir::TF::ShapeOp, mlir::TF::StridedSliceOp,
                  mlir::TF::ReshapeOp, mlir::TF::ExpandDimsOp>(op)) {
      cost += 1;
      continue;
    }

    // Set initial op cost to 1, just like TFRT's cost model does.
    cost += 1;
    for (Type type : op.getOperandTypes()) {
      if (auto tensor = type.dyn_cast<RankedTensorType>()) {
        cost += GetRankedTensorSize(tensor);
      } else {
        cost += max_arg_size;
      }
    }
  }

  return std::max<int64_t>(1, cost);
}

}  // namespace tf_jitrt
}  // end namespace mlir

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tfrt/tf_jitrt_ops.cc.inc"
