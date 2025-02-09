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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSspmd_expansionDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSspmd_expansionDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSspmd_expansionDTcc() {
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

#include "absl/types/optional.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Dialect.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/UseDefLists.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/Passes.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/cc/tensor_layout.h"
#include "tensorflow/dtensor/mlir/dtensor_dialect/ir/dialect.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes.h"
#include "tensorflow/dtensor/mlir/dtensor_mlir_passes_classes.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/layout_parsing.h"
#include "tensorflow/dtensor/mlir/op_utils.h"
#include "tensorflow/dtensor/mlir/spmd_expander.h"
#include "tensorflow/dtensor/mlir/spmd_expander_common.h"

namespace tensorflow {
namespace dtensor {
namespace {

constexpr char kMainFunctionName[] = "main";

// Updates `function` input signature operand at `argument_index` with
// `new_shape`.
void UpdateFunctionInputShape(const int argument_index,
                              mlir::RankedTensorType new_arg_type,
                              mlir::func::FuncOp function) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSspmd_expansionDTcc mht_0(mht_0_v, 233, "", "./tensorflow/dtensor/mlir/spmd_expansion.cc", "UpdateFunctionInputShape");

  auto func_type = function.getFunctionType();
  auto input_types = llvm::to_vector<8>(func_type.getInputs());
  input_types[argument_index] = new_arg_type;
  auto new_func_type = mlir::FunctionType::get(
      function.getContext(), input_types, func_type.getResults());
  function.setType(new_func_type);
  function.getBody()
      .getArgument(argument_index)
      .setType(function.getFunctionType().getInput(argument_index));
}

// If `op` is a TF operation, return itself. If it is an DTensorLayout op,
// return it's consumer TF operation.
mlir::Operation* NextTFOp(mlir::Operation* op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSspmd_expansionDTcc mht_1(mht_1_v, 250, "", "./tensorflow/dtensor/mlir/spmd_expansion.cc", "NextTFOp");

  while (auto layout = llvm::dyn_cast<mlir::TF::DTensorLayout>(op)) {
    if (op->getUsers().empty()) return nullptr;
    op = *(op->getUsers().begin());
  }
  return op;
}

// Updates the shape of resource argument if argument has `tf._layout`
// attribute.
// For example:
// main(%arg0: tensor<!tf_type.resource<tensor<4x4xf32>>
//                  {tf._layout = "mesh:TPU,x=2,y=2 layout:x,not_sharded"})
//
// will be converted to:
//
// main(%arg0: tensor<!tf_type.resource<tensor<2x4xf32>>
//                   {tf._layout = "mesh:TPU,x=2,y=2 layout:x,not_sharded"})
//
// Note that resource argument type is still a resource type. But it's subtype
// has been changed to reflect local shape.
// If resource argument does not have subtype or subtype does not have static
// shapes or if resource argument does not have corresponding layout attribute,
// this function is an no-op.
mlir::LogicalResult UpdateResourceArgumentType(
    const int arg_index, mlir::func::FuncOp function,
    absl::optional<mlir::RankedTensorType> new_subtype = absl::nullopt) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSspmd_expansionDTcc mht_2(mht_2_v, 279, "", "./tensorflow/dtensor/mlir/spmd_expansion.cc", "UpdateResourceArgumentType");

  auto resource_arg = function.getArgument(arg_index);
  if (new_subtype) {
    auto new_var_type = mlir::RankedTensorType::get(
        {}, mlir::TF::ResourceType::get(
                mlir::ArrayRef<mlir::TensorType>{*new_subtype},
                function.getContext()));
    UpdateFunctionInputShape(arg_index, new_var_type, function);
    function.setArgAttr(arg_index, kAssignedResourceLocalShape,
                        ConvertTypeToTensorShapeAttr(*new_subtype));
    return mlir::success();
  }

  auto resource_type = resource_arg.getType()
                           .cast<mlir::TensorType>()
                           .getElementType()
                           .dyn_cast<mlir::TF::ResourceType>();
  if (!resource_type) return mlir::success();

  auto sub_types = resource_type.getSubtypes();
  if (sub_types.size() != 1) return mlir::success();

  auto resource_arg_sub_type = sub_types.front();
  if (!resource_arg_sub_type.hasStaticShape()) return mlir::success();

  // The local shape that is to be assigned to this resource argument type. We
  // will either pull it from the assigned local shape attribute or compute it
  // based on the layout.
  // TODO(srujun): use the attribute value only to check the computed shape.
  // This is currently blocked by an "empty_layout" set on the resource
  // arguments, meaning it is not possible to compute local layout.
  llvm::SmallVector<int64_t, 4> local_arg_shape;
  auto assigned_resource_local_shape_attr =
      function.getArgAttrOfType<mlir::TF::ShapeAttr>(
          arg_index, kAssignedResourceLocalShape);
  if (assigned_resource_local_shape_attr) {
    local_arg_shape.append(
        assigned_resource_local_shape_attr.getShape().begin(),
        assigned_resource_local_shape_attr.getShape().end());
  } else {
    auto layout_or_status = ExtractLayoutFromOperand(resource_arg);
    if (!layout_or_status.ok())
      return function.emitOpError(layout_or_status.status().error_message());

    const auto& layout = layout_or_status.ValueOrDie();
    if (!layout) return mlir::success();

    std::vector<int64_t> local_arg_shape_vec =
        layout->LocalShapeFromGlobalShape(resource_arg_sub_type.getShape());
    local_arg_shape.append(local_arg_shape_vec.begin(),
                           local_arg_shape_vec.end());
  }

  auto local_variable_subtype = mlir::RankedTensorType::get(
      local_arg_shape, resource_arg_sub_type.getElementType());
  auto new_var_type = mlir::RankedTensorType::get(
      {}, mlir::TF::ResourceType::get(
              mlir::ArrayRef<mlir::TensorType>{local_variable_subtype},
              function.getContext()));

  UpdateFunctionInputShape(arg_index, new_var_type, function);
  function.setArgAttr(
      arg_index, kAssignedResourceLocalShape,
      mlir::TF::ShapeAttr::get(local_variable_subtype.getContext(),
                               mlir::ArrayRef<int64_t>(local_arg_shape)));

  return mlir::success();
}

// Returns whether `value` is used by AssignVariable op, skipping DTensorLayout
// op.
bool IsValueUsedByAssignVariableOp(
    mlir::Value value, int* resource_argument_index_for_assign_variable) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSspmd_expansionDTcc mht_3(mht_3_v, 354, "", "./tensorflow/dtensor/mlir/spmd_expansion.cc", "IsValueUsedByAssignVariableOp");

  for (auto user : value.getUsers()) {
    if (auto assign_variable_op =
            llvm::dyn_cast_or_null<mlir::TF::AssignVariableOp>(
                NextTFOp(user))) {
      *resource_argument_index_for_assign_variable =
          GetForwardedDTensorLayoutInput(assign_variable_op.resource())
              .cast<mlir::BlockArgument>()
              .getArgNumber();
      return true;
    }
  }
  return false;
}

// Updates argument shapes of `function` based on `tf._layout` attribute.
mlir::LogicalResult UpdateFunctionArgsUsingLayout(mlir::func::FuncOp function) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSspmd_expansionDTcc mht_4(mht_4_v, 373, "", "./tensorflow/dtensor/mlir/spmd_expansion.cc", "UpdateFunctionArgsUsingLayout");

  for (int argument_index = 0; argument_index < function.getNumArguments();
       ++argument_index) {
    auto arg_layout_attr = function.getArgAttrOfType<mlir::StringAttr>(
        argument_index, kCustomDeviceAttr);
    if (!arg_layout_attr) continue;

    auto arg_layout = Layout::FromString(arg_layout_attr.getValue().str());
    if (!arg_layout.ok())
      return function.emitOpError(llvm::formatv(
          "Invalid layout attribute found during SPMD expansion: {0}",
          arg_layout.status().error_message()));

    mlir::Type arg_type = mlir::getElementTypeOrSelf(
        function.getFunctionType().getInput(argument_index));

    // If argument is a resource type update the subtype shape information
    // to reflect local shape of resources.
    if (arg_type.isa<mlir::TF::ResourceType>()) {
      if (mlir::failed(UpdateResourceArgumentType(argument_index, function)))
        return mlir::failure();
      continue;
    }

    mlir::RankedTensorType ranked_type =
        function.getFunctionType()
            .getInput(argument_index)
            .dyn_cast<mlir::RankedTensorType>();
    if (!ranked_type) continue;

    // If input value is non-resource type, then update the value to reflect
    // local shape.
    llvm::ArrayRef<int64_t> arg_shape = ranked_type.getShape();
    const std::vector<int64_t> arg_local_shape =
        arg_layout->LocalShapeFromGlobalShape(arg_shape);
    mlir::RankedTensorType new_arg_type = mlir::RankedTensorType::get(
        arg_local_shape, ranked_type.getElementType());
    UpdateFunctionInputShape(argument_index, new_arg_type, function);

    // If non-resource value was used for AssignVariable op, then ensure that
    // resource shape of updated/assigned resource is consistent with the
    // local shape of assigned value.
    int assigned_resource_argument_index = -1;
    if (IsValueUsedByAssignVariableOp(function.getArgument(argument_index),
                                      &assigned_resource_argument_index)) {
      (void)UpdateResourceArgumentType(assigned_resource_argument_index,
                                       function, new_arg_type);
    }
  }
  return mlir::success();
}

// Given SPMD expanded `function_operands` to `function`, update the function
// signature to reflect the local shape of `function_operands`.
mlir::LogicalResult UpdateFunctionWithLocalInputShapes(
    mlir::MutableArrayRef<mlir::OpOperand> function_operands,
    mlir::func::FuncOp function) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSspmd_expansionDTcc mht_5(mht_5_v, 432, "", "./tensorflow/dtensor/mlir/spmd_expansion.cc", "UpdateFunctionWithLocalInputShapes");

  for (auto& operand : function_operands) {
    const int index = operand.getOperandNumber();
    auto arg_type = operand.get().getType().dyn_cast<mlir::RankedTensorType>();
    if (!arg_type) continue;

    auto arg_local_shape = arg_type.getShape();
    auto new_arg_type =
        mlir::RankedTensorType::get(arg_local_shape, arg_type.getElementType());
    UpdateFunctionInputShape(index, new_arg_type, function);
  }
  return mlir::success();
}

// Updates output shapes of enclosing op or function containing `terminator_op`
// to local shapes.
mlir::LogicalResult UpdateReturnValueShapes(mlir::ModuleOp module,
                                            mlir::Operation* terminator_op) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSspmd_expansionDTcc mht_6(mht_6_v, 452, "", "./tensorflow/dtensor/mlir/spmd_expansion.cc", "UpdateReturnValueShapes");

  auto parent_op = terminator_op->getBlock()->getParentOp();
  if (!parent_op) return mlir::success();

  auto output_types = llvm::to_vector<8>(terminator_op->getOperandTypes());
  if (auto function = llvm::dyn_cast<mlir::func::FuncOp>(parent_op)) {
    // Update function output type to have local shape.
    auto new_func_type = mlir::FunctionType::get(
        function.getContext(), function.getFunctionType().getInputs(),
        output_types);
    function.setType(new_func_type);

    // Update function callsite operations to reflect local output shapes.
    auto function_uses =
        mlir::SymbolTable::getSymbolUses(function, &module.getBodyRegion());
    if (!function_uses) return mlir::success();

    // Update function callsite operations to reflect local output shapes.
    for (auto function_use : *function_uses) {
      auto callsite_op = function_use.getUser();
      if (!callsite_op) continue;

      for (auto& output_type_and_index : llvm::enumerate(output_types)) {
        int index = output_type_and_index.index();
        const auto& type = output_type_and_index.value();
        callsite_op->getResult(index).setType(type);
      }
    }
  } else {
    for (auto& output_type_and_index : llvm::enumerate(output_types)) {
      int index = output_type_and_index.index();
      const auto& type = output_type_and_index.value();
      parent_op->getResult(index).setType(type);
    }
  }

  return mlir::success();
}

// Conducts SPMD expansion for all ops in `module`. If function call operation
// exists, walk the function in topological order to update inputs/outputs of
// functions before SPMD expansion of callsite operations is done.
// Note that the iteration won't work with recursive function calls.
mlir::LogicalResult ConductSPMDExpansion(mlir::ModuleOp module) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSspmd_expansionDTcc mht_7(mht_7_v, 498, "", "./tensorflow/dtensor/mlir/spmd_expansion.cc", "ConductSPMDExpansion");

  auto main_func = module.lookupSymbol<mlir::func::FuncOp>(kMainFunctionName);
  if (!main_func)
    return module.emitOpError(
        "could not find `main` function in module for SPMD expansion.");

  if (mlir::failed(UpdateFunctionArgsUsingLayout(main_func)))
    return mlir::failure();

  TopologicalIterator iterator(main_func);
  while (iterator.hasNext()) {
    mlir::Operation* op = iterator.next();
    absl::optional<mlir::func::FuncOp> func = MaybeFindFunction(op);
    if (func.has_value()) {
      if (mlir::failed(
              UpdateFunctionWithLocalInputShapes(op->getOpOperands(), *func)))
        return mlir::failure();
    }

    const bool is_terminator_op =
        llvm::isa<mlir::func::ReturnOp, mlir::tf_device::ReturnOp>(op);
    if (auto layout_op = llvm::dyn_cast<mlir::TF::DTensorLayout>(op))
      layout_op.output().setType(layout_op.input().getType());

    mlir::Operation* expanded_op = nullptr;
    auto status = RunSPMDExpansion(op, &expanded_op);
    if (!status.ok() || expanded_op == nullptr) {
      // Sometimes op may been erased and expanded_op set.
      // In this case we should emit the error on the expanded op.
      mlir::Operation* emit_op = op;
      if (expanded_op != nullptr) emit_op = expanded_op;
      return emit_op->emitError(WithContext(status, __FILE__, __LINE__,
                                            "While computing SPMD expansion")
                                    .error_message());
    }

    // If expanded op is terminator of tf_device.Cluster or a function, then
    // make sure to update the function return value as well as the shape of
    // it's callsite operation.
    if (is_terminator_op)
      if (mlir::failed(UpdateReturnValueShapes(module, expanded_op)))
        return mlir::failure();
  }
  return mlir::success();
}

void RemoveDTensorLayoutOps(mlir::ModuleOp module) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSspmd_expansionDTcc mht_8(mht_8_v, 547, "", "./tensorflow/dtensor/mlir/spmd_expansion.cc", "RemoveDTensorLayoutOps");

  llvm::SmallVector<mlir::TF::DTensorLayout, 4> layout_ops;
  module.walk(
      [&](mlir::TF::DTensorLayout layout) { layout_ops.emplace_back(layout); });

  for (auto layout_op : layout_ops) RemoveDTensorLayoutOp(layout_op);
}

// MLIR pass that converts graph in global view into a local view which can be
// invoked in parallel on distributed set of devices. This pass also removes
// all DTensorLayout ops after the expansion is done.
struct DTensorSPMDExpansion
    : public DTensorSPMDExpansionBase<DTensorSPMDExpansion> {
  void getDependentDialects(mlir::DialectRegistry& registry) const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSspmd_expansionDTcc mht_9(mht_9_v, 563, "", "./tensorflow/dtensor/mlir/spmd_expansion.cc", "getDependentDialects");

    registry.insert<mlir::dtensor::DTensorDialect>();
  }

  void runOnOperation() override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSspmd_expansionDTcc mht_10(mht_10_v, 570, "", "./tensorflow/dtensor/mlir/spmd_expansion.cc", "runOnOperation");

    auto module = getOperation();
    if (failed(ConductSPMDExpansion(module))) return signalPassFailure();

    // DTensorLayout only conveys layout information of tensors which is no
    // longer needed after SPMD expansion. As so, remove all layouts from
    // graph.
    RemoveDTensorLayoutOps(module);
  };
};

}  // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
CreateDTensorSPMDExpansion() {
  return std::make_unique<DTensorSPMDExpansion>();
}

}  // namespace dtensor
}  // namespace tensorflow
