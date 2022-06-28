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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_opsDTcc() {
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

/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "absl/strings/str_cat.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Traits.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Interfaces/DecodeAttributesInterfaces.h"  // from @llvm-project
#include "mlir/Interfaces/FoldInterfaces.h"  // from @llvm-project
#include "mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/InliningUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_side_effects.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_structs.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/core/common_runtime/inline_function_utils.h"
#include "tensorflow/core/common_runtime/lower_function_call_inline_policy.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tensorflow/core/util/tensor_format.h"

namespace mlir {
namespace TF {

//===----------------------------------------------------------------------===//
// TF Dialect Interfaces
//===----------------------------------------------------------------------===//

namespace {

struct TFConstantFoldInterface : public DialectFoldInterface {
  TFConstantFoldInterface(Dialect *dialect) : DialectFoldInterface(dialect) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_opsDTcc mht_0(mht_0_v, 258, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_ops.cc", "TFConstantFoldInterface");
}
  LogicalResult fold(Operation *op, ArrayRef<Attribute> operands,
                     SmallVectorImpl<OpFoldResult> &results) const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_opsDTcc mht_1(mht_1_v, 263, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_ops.cc", "fold");

    return TensorFlowDialect::constantFold(op, operands, results);
  }
};

struct TFDecodeAttributesInterface : public DialectDecodeAttributesInterface {
  TFDecodeAttributesInterface(Dialect *dialect)
      : DialectDecodeAttributesInterface(dialect) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_opsDTcc mht_2(mht_2_v, 273, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_ops.cc", "TFDecodeAttributesInterface");
}
  LogicalResult decode(OpaqueElementsAttr input,
                       ElementsAttr &output) const override {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_opsDTcc mht_3(mht_3_v, 278, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_ops.cc", "decode");

    return TensorFlowDialect::decode(input, output);
  }
};

// Helper function that implements the multi-device inlining policy behavior
// for the inliner hook. In particular, for all function body nodes set unset
// placement attributes to match the function call node.
void MultiDeviceProcessInlinedCallBlocks(
    Operation *call, iterator_range<Region::iterator> inlinedBlocks) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_opsDTcc mht_4(mht_4_v, 290, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_ops.cc", "MultiDeviceProcessInlinedCallBlocks");

  using DeviceNameUtils = tensorflow::DeviceNameUtils;

  // Duplicate of the logic in MultiDeviceFunctionBodyPlacer::BodyNodeDevice
  // LINT.IfChange
  auto device_id = StringAttr::get(call->getContext(), "device");
  auto caller_device = call->getAttrOfType<StringAttr>(device_id);
  if (!caller_device) return;

  DeviceNameUtils::ParsedName caller_parsed_device;
  if (!DeviceNameUtils::ParseFullName(caller_device.getValue().str(),
                                      &caller_parsed_device))
    return;

  MLIRContext *context = call->getContext();
  auto node_device = [&](Operation *n) -> StringAttr {
    auto device = n->getAttrOfType<StringAttr>(device_id);
    if (!device || device.getValue().empty()) return caller_device;

    DeviceNameUtils::ParsedName ndef_parsed_device;
    if (!DeviceNameUtils::ParseFullName(device.getValue().str(),
                                        &ndef_parsed_device))
      return device;
    DeviceNameUtils::MergeUnsetDevNames(&ndef_parsed_device,
                                        caller_parsed_device);
    return StringAttr::get(
        context, DeviceNameUtils::ParsedNameToString(ndef_parsed_device));
  };
  // LINT.ThenChange(../../../../core/common_runtime/inline_function_utils.cc)

  for (Block &block : inlinedBlocks) {
    block.walk([&](Operation *op) {
      if (op->getDialect() == call->getDialect())
        op->setAttr(device_id, node_device(op));
    });
  }
}

struct TFInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

  //===--------------------------------------------------------------------===//
  // Analysis Hooks
  //===--------------------------------------------------------------------===//

  // Returns if it's legal to inline 'callable' into the 'call', where 'call' is
  // a TF operation.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_opsDTcc mht_5(mht_5_v, 341, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_ops.cc", "isLegalToInline");

    // Skip inlining for TPUPartitionedCalls.
    if (isa<TPUPartitionedCallOp>(call)) return false;
    // Maintain inlining for  `tf.function`s with jit_compile option.
    if (callable->hasAttr("tf._XlaMustCompile")) return true;
    auto noinline_attr_name = absl::StrCat("tf.", tensorflow::kNoInlineAttr);
    if (auto noinline_attr =
            callable->getAttrOfType<BoolAttr>(noinline_attr_name))
      return !noinline_attr.getValue();
    return true;
  }

  // Returns if its legal to inline 'src' region into the 'dest' region
  // attached to a TF operation.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       BlockAndValueMapping &valueMapping) const final {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_opsDTcc mht_6(mht_6_v, 359, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_ops.cc", "isLegalToInline");

    // Allow inlining in regions attached to region based control flow
    // operations only if the src region is a single block region
    return isa<IfRegionOp, WhileRegionOp>(dest->getParentOp()) &&
           llvm::hasSingleElement(*src);
  }

  // Returns true if its legal to inline a TF operation `op` into the `dest`
  // region.
  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       BlockAndValueMapping &) const final {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_opsDTcc mht_7(mht_7_v, 372, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_ops.cc", "isLegalToInline");

    // An op is legal to inline if either of the following conditions is true:
    // (a) Its legal to duplicate the Op.
    // (b) The Op is inside a single use function. If that function is inlined,
    //     post inlining, the function will be dead and eliminated from the IR.
    //     So there won't be any code duplication.
    // plus the function caller op can be replaced by inlined ops.
    return !wouldBeCloned || TensorFlowDialect::CanDuplicate(op);
  }

  //===--------------------------------------------------------------------===//
  // Transformation Hooks
  //===--------------------------------------------------------------------===//

  // Attempts to materialize a conversion for a type mismatch between a call
  // from this dialect, and a callable region. This method should generate an
  // operation that takes 'input' as the only operand, and produces a single
  // result of 'resultType'. If a conversion can not be generated, nullptr
  // should be returned.
  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type result_type,
                                       Location conversion_loc) const final {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_opsDTcc mht_8(mht_8_v, 396, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_ops.cc", "materializeCallConversion");

    if (!result_type.isa<TensorType>() || !input.getType().isa<TensorType>())
      return nullptr;
    return builder.create<TF::CastOp>(conversion_loc, result_type, input,
                                      /*truncate=*/builder.getBoolAttr(false));
  }

  void processInlinedCallBlocks(
      Operation *call,
      iterator_range<Region::iterator> inlinedBlocks) const final {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_opsDTcc mht_9(mht_9_v, 408, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_ops.cc", "processInlinedCallBlocks");

    bool has_lower_as_multi_device_function_attr = false;
    if (auto lower = call->getAttrOfType<BoolAttr>(
            tensorflow::LowerFunctionalOpsConstants::
                kLowerAsMultiDeviceFunctionAttr))
      has_lower_as_multi_device_function_attr = lower.getValue();
    tensorflow::FunctionCallInlinePolicy policy =
        tensorflow::GetFunctionCallInlinePolicy(
            isa<PartitionedCallOp, StatefulPartitionedCallOp>(call),
            has_lower_as_multi_device_function_attr);

    if (policy == tensorflow::FunctionCallInlinePolicy::kMultiDevicePlacer)
      return MultiDeviceProcessInlinedCallBlocks(call, inlinedBlocks);
  }
};
}  // end anonymous namespace

//===----------------------------------------------------------------------===//
// TF Dialect
//===----------------------------------------------------------------------===//

// Returns true if the op can be duplicated.
bool TensorFlowDialect::CanDuplicate(Operation *op) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_opsDTcc mht_10(mht_10_v, 433, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_ops.cc", "TensorFlowDialect::CanDuplicate");

  // If the op is marked with the cannot duplicate trait, it cannot be
  // duplicated.
  if (op->hasTrait<OpTrait::TF::CannotDuplicate>()) return false;

  // If the op has no memory side effects, it can be duplicated.
  if (MemoryEffectOpInterface::hasNoEffect(op)) return true;

  // If the op is marked stateless using the `is_stateless` attribute, that
  // attribute determines if the op can be duplicated.
  if (auto is_stateless = op->getAttrOfType<BoolAttr>("is_stateless"))
    return is_stateless.getValue();

  // Assume ops can be duplicated if modelled.
  return op->isRegistered();
}

// TF dialect fallback for MemoryEffectOpInterface. The filtering for returning
// the interface is done in the return below and here it is empty as it is only
// returned for known not-stateful and unmodelled ops.
struct TensorFlowRegistryEffectInterfaceFallback
    : public MemoryEffectOpInterface::FallbackModel<
          TensorFlowRegistryEffectInterfaceFallback> {
  static bool classof(Operation *op) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_opsDTcc mht_11(mht_11_v, 459, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_ops.cc", "classof");
 return true; }
  void getEffects(
      Operation *op,
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
          &effects) const {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_opsDTcc mht_12(mht_12_v, 466, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_ops.cc", "getEffects");
}
};

void *TensorFlowDialect::getRegisteredInterfaceForOp(
    mlir::TypeID interface, mlir::OperationName opName) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_opsDTcc mht_13(mht_13_v, 473, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_ops.cc", "TensorFlowDialect::getRegisteredInterfaceForOp");

  if (interface == TypeID::get<mlir::MemoryEffectOpInterface>()) {
    // Don't use fallback for modelled ops.
    if (opName.isRegistered()) return nullptr;

    // Only use fallback interface for known not-stateful ops.
    const tensorflow::OpRegistrationData *op_reg_data = nullptr;
    tensorflow::Status s = tensorflow::OpRegistry::Global()->LookUp(
        opName.stripDialect().str(), &op_reg_data);
    return (s.ok() && !op_reg_data->op_def.is_stateful())
               ? fallback_effect_op_interface_
               : nullptr;
  }

  return nullptr;
}

// Returns true if the op can have side effects.
bool TensorFlowDialect::CanHaveSideEffects(Operation *op) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_opsDTcc mht_14(mht_14_v, 494, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_ops.cc", "TensorFlowDialect::CanHaveSideEffects");

  // If the op has no memory side effects, it has no side effects
  if (MemoryEffectOpInterface::hasNoEffect(op)) return false;

  // If the op is marked stateless using the `is_stateless` attribute, then
  // it has no side effects.
  if (auto is_stateless = op->getAttrOfType<BoolAttr>("is_stateless"))
    return !is_stateless.getValue();

  // Terminators defined in the TF dialect do not have side effects.
  if (op->hasTrait<OpTrait::IsTerminator>()) return false;

  // Otherwise assume that the op can have side effects.
  return true;
}

// Hook functions which may add additional operations to the dialect.
// These are invoked at construction time.
static DenseMap<TypeID, TensorFlowDialect::AdditionalOpFunction>
    &GetAdditionalOperationHooks() {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_opsDTcc mht_15(mht_15_v, 516, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_ops.cc", "GetAdditionalOperationHooks");

  static auto *additional_operation_hooks =
      new DenseMap<TypeID, TensorFlowDialect::AdditionalOpFunction>();
  return *additional_operation_hooks;
}

void TensorFlowDialect::RegisterAdditionalOperationHook(
    TypeID id, AdditionalOpFunction fn) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_opsDTcc mht_16(mht_16_v, 526, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_ops.cc", "TensorFlowDialect::RegisterAdditionalOperationHook");

  GetAdditionalOperationHooks().try_emplace(id, std::move(fn));
}

TensorFlowDialect::ConstantFoldHook TensorFlowDialect::constant_fold_hook_;
TensorFlowDialect::DecodeConstantHook TensorFlowDialect::decode_constant_hook_;

TensorFlowDialect::TensorFlowDialect(MLIRContext *context)
    : Dialect(/*name=*/"tf", context, TypeID::get<TensorFlowDialect>()) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_opsDTcc mht_17(mht_17_v, 537, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_ops.cc", "TensorFlowDialect::TensorFlowDialect");

  context->getOrLoadDialect<::mlir::tf_type::TFTypeDialect>();
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_all_ops.cc.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/tensorflow/ir/tfrt_ops.cc.inc"
      >();
  addInterfaces<TFInlinerInterface, TFDecodeAttributesInterface,
                TFConstantFoldInterface>();
  fallback_effect_op_interface_ =
      new TensorFlowRegistryEffectInterfaceFallback();

  // Support unknown operations because not all TensorFlow operations are
  // registered.
  allowUnknownOperations();

  for (auto &hook : GetAdditionalOperationHooks()) {
    hook.second(*this);
  }
}

TensorFlowDialect::~TensorFlowDialect() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_opsDTcc mht_18(mht_18_v, 564, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_ops.cc", "TensorFlowDialect::~TensorFlowDialect");

  delete fallback_effect_op_interface_;
}

Type TensorFlowDialect::parseType(DialectAsmParser &parser) const {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_opsDTcc mht_19(mht_19_v, 571, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_ops.cc", "TensorFlowDialect::parseType");

  StringRef spec = parser.getFullSymbolSpec();
  llvm::SMLoc loc = parser.getCurrentLocation();
  parser.emitError(
      loc, "tf dialect has no types, potentially meant !tf_type." + spec);
  return nullptr;
}

Attribute TensorFlowDialect::parseAttribute(DialectAsmParser &parser,
                                            Type type) const {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_opsDTcc mht_20(mht_20_v, 583, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_ops.cc", "TensorFlowDialect::parseAttribute");

  StringRef spec = parser.getFullSymbolSpec();
  llvm::SMLoc loc = parser.getCurrentLocation();
  parser.emitError(
      loc, "tf dialect has no attributes, potentially meant #tf_type." + spec);
  return nullptr;
}

Operation *TensorFlowDialect::materializeConstant(OpBuilder &builder,
                                                  Attribute value, Type type,
                                                  Location loc) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPSirPStf_opsDTcc mht_21(mht_21_v, 596, "", "./tensorflow/compiler/mlir/tensorflow/ir/tf_ops.cc", "TensorFlowDialect::materializeConstant");

  return builder.create<ConstOp>(loc, type, value);
}

}  // namespace TF
}  // namespace mlir
