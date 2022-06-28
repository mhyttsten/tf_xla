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
class MHTracer_DTPStensorflowPSdtensorPSmlirPScluster_function_conversionDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPScluster_function_conversionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPScluster_function_conversionDTcc() {
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

#include <utility>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes_classes.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"

namespace tensorflow {
namespace dtensor {
namespace {

// Attach layouts for all the returned values so that custom device could get
// layouts for the handles.
mlir::LogicalResult AttachRetvalLayouts(
    mlir::OpBuilder* builder, mlir::TF::StatefulPartitionedCallOp sp_call_op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPScluster_function_conversionDTcc mht_0(mht_0_v, 219, "", "./tensorflow/dtensor/mlir/cluster_function_conversion.cc", "AttachRetvalLayouts");

  // Find the FuncOp that the StatefulPartitionedCallOp is invoking.
  mlir::SymbolRefAttr sym =
      sp_call_op.getCallableForCallee().dyn_cast<mlir::SymbolRefAttr>();
  if (!sym)
    return sp_call_op.emitOpError(
        "has no symbolRef for given StatefulPartitionedCallOp");

  auto func = mlir::dyn_cast<mlir::func::FuncOp>(
      mlir::SymbolTable::lookupNearestSymbolFrom(sp_call_op, sym));
  if (!func)
    return sp_call_op.emitOpError() << "found no FuncOp for symbol " << sym;

  llvm::SmallVector<absl::optional<Layout>, 8> retvals_layouts;
  retvals_layouts.reserve(func.getNumResults());
  for (auto operand : func.front().getTerminator()->getOperands()) {
    auto result_layout_or_status = ExtractLayoutFromOperand(operand);
    if (!result_layout_or_status.ok()) {
      return func.emitOpError("error while parsing result layout for function");
    }

    auto result_layout = result_layout_or_status.ValueOrDie();

    // When function returns its arguments directly, layout information for the
    // return value of `func` may be only obtainable by looking at it's callsite
    // operations. In that case, query the input layouts for function callsite
    // operations for layout information.
    if (!result_layout) {
      if (auto block_arg = operand.dyn_cast<mlir::BlockArgument>()) {
        auto layout_or_status = ExtractLayoutFromOperand(
            sp_call_op.getOperand(block_arg.getArgNumber()));
        if (!layout_or_status.ok())
          return func.emitOpError(
              "error while parsing result layout for function");
        result_layout = std::move(layout_or_status.ValueOrDie());
      }

      if (!result_layout)
        return func.emitOpError(
            llvm::formatv("missing result layout attribute for function. All "
                          "DTensor functions "
                          "must have layouts for its results."));
    }
    retvals_layouts.emplace_back(result_layout.value());
  }

  // Note that we set this unconditionally - retvals_layout could be empty, but
  // that is fine and we will have an empty _layout for the
  // StatefulPartitionedCallOp. This is fine as for op without return values,
  // all we need is a placeholder layout so that no special case is needed in
  // dtensor_device.
  SetLayoutOnOp(sp_call_op,
                absl::Span<const absl::optional<Layout>>(
                    retvals_layouts.data(), retvals_layouts.size()));

  return mlir::success();
}

// Add an anotation to skip xla compilation for VarHandleOp and
// DestroyResourceOp.
void MaybeSkipXlaCompilation(mlir::OpBuilder* builder,
                             mlir::Operation* call_op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPScluster_function_conversionDTcc mht_1(mht_1_v, 283, "", "./tensorflow/dtensor/mlir/cluster_function_conversion.cc", "MaybeSkipXlaCompilation");

  auto function = MaybeFindFunction(call_op);
  const auto& body_ops = function->getBody().front().without_terminator();
  // VarHandleOp and DestroyResourceOp run on op-by-op mode, so there is only
  // one op in the function body.
  if (std::distance(std::begin(body_ops), std::end(body_ops)) == 1 &&
      llvm::isa<mlir::TF::VarHandleOp, mlir::TF::DestroyResourceOp>(
          body_ops.begin())) {
    call_op->setAttr(kSkipXlaCompilation, builder->getBoolAttr(true));
  }
}

mlir::LogicalResult ReplaceClusterWithPartitionCallOp(
    mlir::OpBuilder* builder, mlir::tf_device::ClusterFuncOp cluster_func) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPScluster_function_conversionDTcc mht_2(mht_2_v, 299, "", "./tensorflow/dtensor/mlir/cluster_function_conversion.cc", "ReplaceClusterWithPartitionCallOp");

  auto mesh_attr = cluster_func->getAttrOfType<mlir::StringAttr>(kMeshAttr);
  if (!mesh_attr)
    return cluster_func.emitOpError()
           << "requires " << llvm::StringRef(kMeshAttr) << " attribute";

  llvm::SmallVector<mlir::Type, 8> output_types{
      cluster_func.getResultTypes().begin(),
      cluster_func.getResultTypes().end()};

  auto function_name = cluster_func.funcAttr();

  builder->setInsertionPoint(cluster_func);
  auto call_op = builder->create<mlir::TF::StatefulPartitionedCallOp>(
      cluster_func.getLoc(), output_types, cluster_func.getOperands(),
      function_name, mesh_attr, /*config_proto=*/builder->getStringAttr(""),
      /*executor_type=*/builder->getStringAttr(""));

  MaybeSkipXlaCompilation(builder, call_op);

  if (mlir::failed(ValidateMetadataAttributes(cluster_func)))
    return mlir::failure();

  // All attributes beginning with `_` is validate, perform copy.
  mlir::TF::CopyUnderscoredAttributes(cluster_func, call_op);

  cluster_func.replaceAllUsesWith(call_op.getResults());
  cluster_func.erase();

  return AttachRetvalLayouts(builder, call_op);
}

// MLIR pass that converts tf_device.cluster_func to TF partitioned call
// op with device mesh config added to `config` attribute.
struct DTensorClusterFunctionConversion
    : public DTensorClusterFunctionConversionBase<
          DTensorClusterFunctionConversion> {
  void runOnOperation() override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPScluster_function_conversionDTcc mht_3(mht_3_v, 339, "", "./tensorflow/dtensor/mlir/cluster_function_conversion.cc", "runOnOperation");

    mlir::MLIRContext& context = getContext();

    // Find all tf_device.ClusterFunc ops and visit them in post order. This
    // order guarantees that ops in function definition is visited before
    // function call site operations. When python graph includes tf.functions
    // this leads to nested tf_device.ClusterFunc ops. As we infer the layout
    // of function call operations with layout attached to return values in the
    // function definition, ClusterFunc op in nested/inner functions must be
    // visited before ClusterFunc op in outer functions.
    llvm::SmallVector<mlir::tf_device::ClusterFuncOp, 8> clusters;
    getOperation().walk([&](mlir::tf_device::ClusterFuncOp cluster_func) {
      clusters.emplace_back(cluster_func);
    });

    mlir::OpBuilder op_builder(&context);
    for (auto cluster_func : llvm::reverse(clusters)) {
      if (mlir::failed(
              ReplaceClusterWithPartitionCallOp(&op_builder, cluster_func))) {
        return signalPassFailure();
      }
    }
  };
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorClusterFunctionConversion() {
  return std::make_unique<DTensorClusterFunctionConversion>();
}

}  // namespace dtensor
}  // namespace tensorflow
