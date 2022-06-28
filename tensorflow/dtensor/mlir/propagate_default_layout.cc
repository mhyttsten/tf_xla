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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSpropagate_default_layoutDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSpropagate_default_layoutDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSpropagate_default_layoutDTcc() {
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

#include <string>

#include "absl/types/optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/dialect.h"
#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/dtensor_attributes.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes_classes.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {
namespace {

// Creates tf.DTensorLayout op that forwards `input` value.
void CreateDTensorLayoutOp(const Layout& layout, mlir::Value input,
                           mlir::TensorType& type, mlir::Location loc,
                           mlir::OpBuilder* builder,
                           mlir::MLIRContext* context) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSpropagate_default_layoutDTcc mht_0(mht_0_v, 217, "", "./tensorflow/dtensor/mlir/propagate_default_layout.cc", "CreateDTensorLayoutOp");

  if (layout.IsEmpty()) return;

  auto layout_op = builder->create<mlir::TF::DTensorLayout>(
      loc, input, mlir::dtensor::LayoutAttr::get(context, layout),
      mlir::TF::ShapeAttr::get(context, type));
  llvm::SmallPtrSet<mlir::Operation*, 4> exception{layout_op};
  input.replaceAllUsesExcept(layout_op.output(), exception);
}

// Adds DTensorLayout op following each Relayout operation to ensure that
// tensor from `relayout` has fixed layout.
mlir::LogicalResult PropagateDTensorLayoutForRelayout(
    mlir::MLIRContext& c, mlir::TF::RelayoutOp relayout) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSpropagate_default_layoutDTcc mht_1(mht_1_v, 233, "", "./tensorflow/dtensor/mlir/propagate_default_layout.cc", "PropagateDTensorLayoutForRelayout");

  const std::string layout_str = relayout.layout().str();
  auto layout_or_status = Layout::FromString(layout_str);
  if (!layout_or_status.ok()) {
    return relayout.emitOpError(
        llvm::formatv("found Relayout op with incorrect/unparsable layout. "
                      "Found layout: {0} ",
                      layout_str));
  }
  const Layout& layout = layout_or_status.ValueOrDie();

  // Skip adding a DTensorLayout if Relayout is 'dynamic'. Any dimension with
  // MATCH for the layout will have its layout preserved in layout propagation.
  for (const std::string& sharding_spec : layout.sharding_spec_strs())
    if (sharding_spec == Layout::kMatch) return mlir::success();

  mlir::OpBuilder builder(relayout->getBlock(),
                          ++mlir::Block::iterator(relayout));
  mlir::TensorType type = relayout.getType().dyn_cast<mlir::TensorType>();
  if (!type) return relayout.emitOpError("type required for Relayout op");

  CreateDTensorLayoutOp(layout, relayout.output(), type, relayout.getLoc(),
                        &builder, &c);
  return mlir::success();
}

// Creates tf.DTensorLayout that is connected to each function argument if
// function arg contains layout attribute.
mlir::LogicalResult PropagateFunctionArgAttrToLayoutOp(
    mlir::MLIRContext& c, mlir::func::FuncOp function) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSpropagate_default_layoutDTcc mht_2(mht_2_v, 265, "", "./tensorflow/dtensor/mlir/propagate_default_layout.cc", "PropagateFunctionArgAttrToLayoutOp");

  for (int arg_index = 0; arg_index < function.getNumArguments(); ++arg_index) {
    auto layout_attr = function.getArgAttrOfType<mlir::StringAttr>(
        arg_index, kCustomDeviceAttr);
    if (!layout_attr) continue;
    const auto layout_str = layout_attr.getValue().str();
    auto layout_or_status = Layout::FromString(layout_str);
    if (!layout_or_status.ok())
      return function.emitOpError(llvm::formatv(
          "function includes attribute {0} for {1}-th arg that cannot be "
          "serialized to correct layout format. Found attribute {3}",
          kCustomDeviceAttr, arg_index, layout_str));

    mlir::OpBuilder builder(function.getBody());
    auto arg = function.getArgument(arg_index);
    mlir::Type tensor_type = GetSubtypeOrSelf(arg);
    if (auto type = tensor_type.dyn_cast<mlir::TensorType>()) {
      CreateDTensorLayoutOp(layout_or_status.ValueOrDie(), arg, type,
                            function.getLoc(), &builder, &c);
    } else {
      return function.emitOpError()
             << "is missing tensor type for argument " << arg_index;
    }
  }

  return mlir::success();
}

// Creates tf.DTensorLayout that is connected to terminator op of function if
// function contains default layout attribute that represents layout of function
// outputs.
mlir::LogicalResult PropagateFunctionDefaultLayoutAttrToLayoutOp(
    mlir::MLIRContext& c, mlir::func::FuncOp function) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSpropagate_default_layoutDTcc mht_3(mht_3_v, 300, "", "./tensorflow/dtensor/mlir/propagate_default_layout.cc", "PropagateFunctionDefaultLayoutAttrToLayoutOp");

  for (int ret_index = 0; ret_index < function.getNumResults(); ++ret_index) {
    auto layout_attr_from_func_result =
        function.getResultAttrOfType<mlir::StringAttr>(
            ret_index, kCustomDefaultLayoutAttr);
    if (!layout_attr_from_func_result) continue;

    const std::string layout_string =
        layout_attr_from_func_result.getValue().str();
    auto result_layout_or_status = Layout::FromString(layout_string);
    if (!result_layout_or_status.ok())
      return function.emitOpError(
          llvm::formatv("function includes default layout attribute {0} for "
                        "{1}-th output that cannot be serialized to correct "
                        "layout format. Found attribute {3}",
                        kCustomDefaultLayoutAttr, ret_index, layout_string));

    auto function_terminator = function.getBody().front().getTerminator();
    mlir::OpBuilder builder(function_terminator);
    auto return_value = function_terminator->getOperand(ret_index);

    if (auto type = return_value.getType().dyn_cast<mlir::TensorType>())
      CreateDTensorLayoutOp(result_layout_or_status.ValueOrDie(), return_value,
                            type, function.getLoc(), &builder, &c);
    else
      return function.emitOpError()
             << "is missing tensor type for result " << ret_index;
  }

  return mlir::success();
}

// MLIR pass that removes trivially unused operations in graph.
struct DTensorPropagateDefaultLayout
    : public DTensorPropagateDefaultLayoutBase<DTensorPropagateDefaultLayout> {
  void getDependentDialects(mlir::DialectRegistry& registry) const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSpropagate_default_layoutDTcc mht_4(mht_4_v, 338, "", "./tensorflow/dtensor/mlir/propagate_default_layout.cc", "getDependentDialects");

    registry.insert<mlir::dtensor::DTensorDialect>();
  }

  void runOnOperation() override {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSpropagate_default_layoutDTcc mht_5(mht_5_v, 345, "", "./tensorflow/dtensor/mlir/propagate_default_layout.cc", "runOnOperation");

    mlir::MLIRContext& context = getContext();
    mlir::OpBuilder builder(&context);

    auto function = getOperation();

    auto walk_result =
        getOperation().walk([&](mlir::Operation* op) -> mlir::WalkResult {
          if (auto relayout = llvm::dyn_cast<mlir::TF::RelayoutOp>(op)) {
            (void)PropagateDTensorLayoutForRelayout(context, relayout);
            return mlir::WalkResult::advance();
          }

          // Set user annotated layout on operations.
          auto layout_or_status = ExtractLayoutFromOp(op);
          if (!layout_or_status.ok()) {
            op->emitOpError(llvm::formatv(
                "op has layout attribute {0} that cannot be deserizlied.",
                kLayoutAttr));
            return mlir::WalkResult::interrupt();
          }

          mlir::OpBuilder builder(&context);
          builder.setInsertionPointAfter(op);
          const auto layouts = layout_or_status.ValueOrDie();
          for (const auto& layout_and_index : llvm::enumerate(layouts)) {
            const int index = layout_and_index.index();
            const auto& layout = layout_and_index.value();
            if (!layout || layout->IsEmpty()) continue;

            auto op_output = op->getResult(index);
            if (auto type = op_output.getType().dyn_cast<mlir::TensorType>()) {
              auto layout_op = builder.create<mlir::TF::DTensorLayout>(
                  function.getLoc(), op_output,
                  mlir::dtensor::LayoutAttr::get(&context, *layout),
                  mlir::TF::ShapeAttr::get(&context, type));
              llvm::SmallPtrSet<mlir::Operation*, 4> exception{layout_op};
              op_output.replaceAllUsesExcept(layout_op.output(), exception);
            } else {
              return op->emitOpError()
                     << "type for output " << index << " is not a TensorType";
            }
          }

          return mlir::WalkResult::advance();
        });

    if (walk_result.wasInterrupted()) return signalPassFailure();

    // Set user annotated layout on function arguments.
    if (mlir::failed(PropagateFunctionArgAttrToLayoutOp(context, function)))
      return signalPassFailure();

    // Set user annotated layout on function outputs.
    if (mlir::failed(
            PropagateFunctionDefaultLayoutAttrToLayoutOp(context, function)))
      return signalPassFailure();
  }
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
CreateDTensorPropagateDefaultLayout() {
  return std::make_unique<DTensorPropagateDefaultLayout>();
}

}  // namespace dtensor
}  // namespace tensorflow
