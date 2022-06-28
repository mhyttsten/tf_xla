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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScorert_converterDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScorert_converterDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScorert_converterDTcc() {
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

#include "tensorflow/compiler/mlir/tfrt/transforms/corert_converter.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/core/util/device_name_utils.h"
#include "tfrt/basic_kernels/opdefs/basic_kernels.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/attributes.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/core_runtime.h"  // from @tf_runtime
#include "tfrt/core_runtime/opdefs/types.h"  // from @tf_runtime
#include "tfrt/distributed_runtime/opdefs/kernels.h"  // from @tf_runtime

namespace tensorflow {

CoreRTConverter::CoreRTConverter(
    mlir::MLIRContext *context,
    const mlir::TF::SideEffectAnalysis::Info *side_effect_analysis)
    : builder_(context), side_effect_analysis_(*side_effect_analysis) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScorert_converterDTcc mht_0(mht_0_v, 209, "", "./tensorflow/compiler/mlir/tfrt/transforms/corert_converter.cc", "CoreRTConverter::CoreRTConverter");

  addConversion([](tfrt::compiler::ChainType type) { return type; });
  addConversion([](tfrt::corert::OpHandlerType type) { return type; });
  addConversion([](tfrt::dist::DistributedContextType type) { return type; });
  addConversion([](tfrt::corert::TensorHandleType type) { return type; });
  addConversion([=](mlir::TensorType type) -> llvm::Optional<mlir::Type> {
    // Ref types are not supported in both compiler and runtime.
    if (type.getElementType().isa<mlir::TF::TensorFlowRefType>())
      return llvm::None;
    return tensor_handle_type();
  });
  addConversion([=](mlir::Type type) -> llvm::Optional<mlir::Type> {
    if (type == builder_.getI1Type()) return type;
    return llvm::None;
  });
}

void CoreRTConverter::MaterializeDerivedAttributes(mlir::Operation *op) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScorert_converterDTcc mht_1(mht_1_v, 229, "", "./tensorflow/compiler/mlir/tfrt/transforms/corert_converter.cc", "CoreRTConverter::MaterializeDerivedAttributes");

  if (auto interface = llvm::dyn_cast<mlir::DerivedAttributeOpInterface>(op)) {
    auto derived_attrs = interface.materializeDerivedAttributes();
    for (auto named_attr : derived_attrs) {
      op->setAttr(named_attr.getName(), named_attr.getValue());
    }
  }
}

bool CoreRTConverter::IsSupportedNumericDType(mlir::Type type) const {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScorert_converterDTcc mht_2(mht_2_v, 241, "", "./tensorflow/compiler/mlir/tfrt/transforms/corert_converter.cc", "CoreRTConverter::IsSupportedNumericDType");

  // Most of the tensorflow data types (eg. f32, i64) are supported and they
  // are standard MLIR types that need no conversion here.
  if (type.isBF16() || type.isF16() || type.isF32() || type.isF64() ||
      type.isInteger(1) || type.isInteger(8) || type.isInteger(16) ||
      type.isInteger(32) || type.isInteger(64) || type.isUnsignedInteger(8) ||
      type.isUnsignedInteger(16) || type.isUnsignedInteger(32) ||
      type.isUnsignedInteger(64))
    return true;

  if (auto complex_type = type.dyn_cast<mlir::ComplexType>()) {
    auto element_type = complex_type.getElementType();
    if (element_type.isF32() || element_type.isF64()) return true;
  }

  return false;
}

mlir::ArrayAttr CoreRTConverter::CreateOpAttrs(ArrayRef<NamedAttribute> attrs) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScorert_converterDTcc mht_3(mht_3_v, 262, "", "./tensorflow/compiler/mlir/tfrt/transforms/corert_converter.cc", "CoreRTConverter::CreateOpAttrs");

  llvm::SmallVector<mlir::Attribute, 4> attr_array;
  for (auto key_and_value : attrs) {
    if (!IsUnusedAttribute(key_and_value.getName())) {
      auto converted = ConvertAttribute(key_and_value.getValue());
      if (!converted) return {};

      mlir::StringAttr key =
          builder_.getStringAttr(key_and_value.getName().strref());
      attr_array.push_back(builder_.getArrayAttr({key, converted}));
    }
  }
  return builder_.getArrayAttr(attr_array);
}

mlir::ArrayAttr CoreRTConverter::CreateOpFuncAttrs(
    ArrayRef<NamedAttribute> attrs,
    llvm::SmallVector<mlir::StringAttr, 4> *func_attr_keys) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScorert_converterDTcc mht_4(mht_4_v, 282, "", "./tensorflow/compiler/mlir/tfrt/transforms/corert_converter.cc", "CoreRTConverter::CreateOpFuncAttrs");

  llvm::SmallVector<mlir::Attribute, 4> attr_array;
  for (auto key_and_value : attrs) {
    auto attr_key = key_and_value.getName();
    auto attr_value = key_and_value.getValue();
    if (!IsUnusedAttribute(attr_key) &&
        attr_value.isa<mlir::FlatSymbolRefAttr, mlir::SymbolRefAttr>()) {
      auto func_attr = attr_value.dyn_cast<mlir::FlatSymbolRefAttr>();
      auto converted = ConvertSymbolAttrToStringAttr(func_attr);
      mlir::StringAttr key = builder_.getStringAttr(attr_key.strref());
      attr_array.push_back(builder_.getArrayAttr({key, converted}));

      // Remove the attribute to avoid being converted again.
      func_attr_keys->push_back(attr_key);
    }
  }
  return builder_.getArrayAttr(attr_array);
}

// TODO(chky): Add support for multiple device instances.
llvm::Optional<ParseDeviceNameResult> CoreRTConverter::ParseDeviceName(
    llvm::StringRef device_name) const {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScorert_converterDTcc mht_5(mht_5_v, 306, "", "./tensorflow/compiler/mlir/tfrt/transforms/corert_converter.cc", "CoreRTConverter::ParseDeviceName");

  std::string tf_device_name = device_name.str();

  if (tf_device_name.empty()) {
    return llvm::None;
  }

  ParseDeviceNameResult result;
  result.device_name = tf_device_name;

  // Parse the device name in format of the current tensorflow.
  DeviceNameUtils::ParsedName parsed_name;
  if (!DeviceNameUtils::ParseFullName(result.device_name, &parsed_name)) {
    return llvm::None;
  }
  if (!parsed_name.has_type) {
    return llvm::None;
  }
  result.device_type = parsed_name.type;

  result.op_handler_name = tf_device_name;

  return result;
}

llvm::Optional<ParseDeviceNameResult> CoreRTConverter::ParseDeviceName(
    mlir::Operation *op) const {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScorert_converterDTcc mht_6(mht_6_v, 335, "", "./tensorflow/compiler/mlir/tfrt/transforms/corert_converter.cc", "CoreRTConverter::ParseDeviceName");

  auto device_attr = op->getAttr("device");
  if (!device_attr) {
    return llvm::None;
  }

  auto parsed_device_name =
      ParseDeviceName(device_attr.cast<mlir::StringAttr>().getValue());
  if (!parsed_device_name) op->emitWarning("failed to parse device name.");
  return parsed_device_name;
}

mlir::Value CoreRTConverter::ConvertOpHandler(
    mlir::Operation *op, llvm::StringRef op_handler_name,
    ConversionPatternRewriter *rewriter) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScorert_converterDTcc mht_7(mht_7_v, 352, "", "./tensorflow/compiler/mlir/tfrt/transforms/corert_converter.cc", "CoreRTConverter::ConvertOpHandler");

  auto iter = op_handler_by_name_.find(op_handler_name);
  if (iter != op_handler_by_name_.end()) return iter->second;

  mlir::Block *block = op->getBlock();
  ConversionPatternRewriter::InsertionGuard insertion_guard(*rewriter);
  rewriter->setInsertionPointToStart(block);

  FuncOp func_op = op->getParentOfType<mlir::func::FuncOp>();
  mlir::Value in_chain = func_op.getArgument(0);
  auto get_op_handler_op = rewriter->create<tfrt::corert::GetOpHandler>(
      block->getParent()->getLoc(), op_handler_type(), in_chain,
      op_handler_name);
  op_handler_by_name_[op_handler_name] = get_op_handler_op.getResult();
  return get_op_handler_op.getResult();
}

mlir::Value CoreRTConverter::GetDistributedContext(
    mlir::Operation *op, mlir::ConversionPatternRewriter *rewriter) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScorert_converterDTcc mht_8(mht_8_v, 373, "", "./tensorflow/compiler/mlir/tfrt/transforms/corert_converter.cc", "CoreRTConverter::GetDistributedContext");

  mlir::func::FuncOp func_op = op->getParentOfType<mlir::func::FuncOp>();
  auto iter = distributed_context_by_func_.find(func_op.getOperation());
  if (iter != distributed_context_by_func_.end()) {
    return iter->second;
  }
  ConversionPatternRewriter::InsertionGuard insertion_guard(*rewriter);
  rewriter->setInsertionPoint(op);
  auto get_dist_ctx_op = rewriter->create<tfrt::dist::GetDistributedContextOp>(
      op->getLoc(), distributed_context_type());

  mlir::Value result = get_dist_ctx_op.result();
  distributed_context_by_func_[func_op.getOperation()] = result;
  return result;
}

mlir::Value CoreRTConverter::GetRemoteChainManager(
    mlir::Operation *op, mlir::ConversionPatternRewriter *rewriter) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScorert_converterDTcc mht_9(mht_9_v, 393, "", "./tensorflow/compiler/mlir/tfrt/transforms/corert_converter.cc", "CoreRTConverter::GetRemoteChainManager");

  mlir::func::FuncOp func_op = op->getParentOfType<mlir::func::FuncOp>();
  auto iter = remote_chain_mgr_by_func_.find(func_op.getOperation());
  if (iter != remote_chain_mgr_by_func_.end()) {
    return iter->second;
  }
  ConversionPatternRewriter::InsertionGuard insertion_guard(*rewriter);
  rewriter->setInsertionPoint(op);

  mlir::Type remote_chain_mgr_type =
      builder_.getType<::tfrt::dist::RemoteChainManagerType>();
  mlir::Value dist_ctx = GetDistributedContext(op, rewriter);
  auto create_mgr_op = rewriter->create<tfrt::dist::CreateRemoteChainManager>(
      op->getLoc(), remote_chain_mgr_type, dist_ctx);

  mlir::Value result = create_mgr_op.result();
  remote_chain_mgr_by_func_[func_op.getOperation()] = result;
  return result;
}

mlir::Value CoreRTConverter::GetLocalSideEffectChain(
    mlir::Operation *op, mlir::ConversionPatternRewriter *rewriter) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScorert_converterDTcc mht_10(mht_10_v, 417, "", "./tensorflow/compiler/mlir/tfrt/transforms/corert_converter.cc", "CoreRTConverter::GetLocalSideEffectChain");

  auto func_op = op->getParentOfType<mlir::func::FuncOp>();

  llvm::SmallVector<mlir::Operation *, 4> predecessors;
  if (llvm::isa<mlir::func::ReturnOp>(op)) {
    auto sinks = side_effect_analysis_.ControlSinks();
    predecessors.assign(sinks.begin(), sinks.end());
  } else {
    predecessors = side_effect_analysis_.DirectControlPredecessors(op);
  }

  llvm::SmallVector<mlir::Value, 2> chains;
  for (auto *pred : predecessors) {
    // TODO(chky): ReadVariableOp is removed in the pass and not converted.
    // Ideally, every side-effecting op should be converted to a
    // tfrt_fallback.executeop.seq op. The special rewrite logic of
    // ReadVariableOp should be done in a previous pass.
    if (auto chain = local_side_effect_chains_.lookup(pred))
      chains.push_back(chain);
  }

  // If there is no side-effect predecessor, then the input side-effect chain
  // is used.
  if (chains.empty()) return func_op.getArgument(0);

  if (chains.size() == 1) return chains[0];

  // If there are multiple side-effect predecessors, insert a merge_chains
  // kernel and return the merged chain.
  ConversionPatternRewriter::InsertionGuard insertion_guard(*rewriter);
  rewriter->setInsertionPoint(op);
  return rewriter->create<tfrt::compiler::MergeChainsOp>(op->getLoc(),
                                                         chain_type(), chains);
}

mlir::Value CoreRTConverter::GetTaskHandle(
    mlir::Operation *op, StringRef task_name,
    mlir::ConversionPatternRewriter *rewriter) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScorert_converterDTcc mht_11(mht_11_v, 457, "", "./tensorflow/compiler/mlir/tfrt/transforms/corert_converter.cc", "CoreRTConverter::GetTaskHandle");

  mlir::func::FuncOp func_op = op->getParentOfType<mlir::func::FuncOp>();
  llvm::StringMap<mlir::Value> &task_handle_by_name =
      task_handles_by_func_[func_op.getOperation()];
  auto iter = task_handle_by_name.find(task_name);
  if (iter != task_handle_by_name.end()) {
    return iter->second;
  }

  mlir::Value distributed_context = GetDistributedContext(op, rewriter);
  auto task_handle_op = rewriter->create<tfrt::dist::GetTaskHandleOp>(
      op->getLoc(), rewriter->getType<tfrt::dist::TaskHandleType>(),
      distributed_context, task_name);

  task_handle_by_name[task_name] = task_handle_op.getResult();
  return task_handle_op.getResult();
}

mlir::Value CoreRTConverter::GetRemoteSideEffectChain(
    mlir::Operation *op, StringRef remote_host,
    mlir::ConversionPatternRewriter *rewriter) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScorert_converterDTcc mht_12(mht_12_v, 480, "", "./tensorflow/compiler/mlir/tfrt/transforms/corert_converter.cc", "CoreRTConverter::GetRemoteSideEffectChain");

  mlir::Value remote_chain_mgr = GetRemoteChainManager(op, rewriter);
  mlir::Value local_chain = GetLocalSideEffectChain(op, rewriter);
  mlir::Value task_handle = GetTaskHandle(op, remote_host, rewriter);
  mlir::Type remote_obj_id_ty =
      rewriter->getType<tfrt::dist::RemoteObjectIdType>();

  // Get the remote chain using the tfrt_dist.get_chain_for_task_handle op.
  auto get_chain_op = rewriter->create<tfrt::dist::GetChainForTaskHandleOp>(
      op->getLoc(), remote_obj_id_ty, local_chain, remote_chain_mgr,
      task_handle);
  return get_chain_op.getResult();
}

mlir::Attribute CoreRTConverter::ConvertAttribute(mlir::Attribute attr) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScorert_converterDTcc mht_13(mht_13_v, 497, "", "./tensorflow/compiler/mlir/tfrt/transforms/corert_converter.cc", "CoreRTConverter::ConvertAttribute");

  // The supported attributes here should be kept consistent with
  // //third_party/tf_runtime/include/tfrt/core_runtime/op_attr_type.h
  //
  // Currently, not all tensorflow data types are supported. Unranked shape
  // attributes are not supported yet.

  // Return directly if the attribute is already supported.
  if (attr.isa<mlir::IntegerAttr, mlir::FloatAttr, mlir::BoolAttr,
               mlir::StringAttr, mlir::DenseIntOrFPElementsAttr>())
    return attr;

  // For type attributes, we convert non-standard MLIR types to corresponding
  // corert types.
  if (auto type_attr = attr.dyn_cast<mlir::TypeAttr>()) {
    if (auto shape_type = type_attr.getValue().dyn_cast<mlir::TensorType>()) {
      if (!shape_type.hasRank())
        return tfrt::corert::ShapeAttr::get(builder_.getContext());

      return tfrt::corert::ShapeAttr::get(builder_.getContext(),
                                          shape_type.getShape());
    }

    return ConvertTypeAttribute(type_attr);
  }

  // Convert the attribute to the corresponding format in TFRT dialect if
  // needed.
  if (auto shape_attr = attr.dyn_cast<mlir::TF::ShapeAttr>()) {
    if (!shape_attr.hasRank())
      return tfrt::corert::ShapeAttr::get(builder_.getContext());
    return tfrt::corert::ShapeAttr::get(builder_.getContext(),
                                        shape_attr.getShape());
  }

  // For arrays, we recursively convert the elements.
  if (auto array_attr = attr.dyn_cast<mlir::ArrayAttr>()) {
    llvm::SmallVector<mlir::Attribute, 8> attrs;
    attrs.reserve(array_attr.size());
    for (auto attr : array_attr) {
      auto converted = ConvertAttribute(attr);
      if (!converted) return {};
      attrs.push_back(converted);
    }
    return builder_.getArrayAttr(attrs);
  }

  return {};
}

mlir::StringAttr CoreRTConverter::ConvertSymbolAttrToStringAttr(
    mlir::FlatSymbolRefAttr symbol_attr) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScorert_converterDTcc mht_14(mht_14_v, 551, "", "./tensorflow/compiler/mlir/tfrt/transforms/corert_converter.cc", "CoreRTConverter::ConvertSymbolAttrToStringAttr");

  // Currently in TF graph to MLIR importing, a "0" is appended to the original
  // function name, so we pop it here. The renaming is for TF/XLA v1 bridge
  // use cases. Refer to b/142268695, b/141617294 for more context.
  //
  // In TFRT use cases, in almost every case "0" is the only literal
  // appended since TF Graph already guarantee function name uniqueness.
  // TODO(b/172092902): Investigate a better way to make the tf_func_name to
  // mlir_tf_func_name conversion reversible.
  auto func_name = symbol_attr.getValue().drop_back().str();

  return mlir::StringAttr::get(builder_.getContext(), func_name);
}

mlir::TypeAttr CoreRTConverter::ConvertTypeAttribute(mlir::TypeAttr type_attr) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrtPStransformsPScorert_converterDTcc mht_15(mht_15_v, 568, "", "./tensorflow/compiler/mlir/tfrt/transforms/corert_converter.cc", "CoreRTConverter::ConvertTypeAttribute");

  auto type = type_attr.getValue();

  if (IsSupportedNumericDType(type)) return type_attr;

  // For TF custom types, we convert it to custom corert types.
  if (type.isa<mlir::TF::StringType>())
    return mlir::TypeAttr::get(
        tfrt::corert::StringType::get(builder_.getContext()));

  if (type.isa<mlir::TF::ResourceType>())
    return mlir::TypeAttr::get(
        tfrt::corert::ResourceType::get(builder_.getContext()));

  if (type.isa<mlir::TF::VariantType>())
    return mlir::TypeAttr::get(
        tfrt::corert::VariantType::get(builder_.getContext()));

  if (type.isa<mlir::TF::Quint8Type>()) {
    return mlir::TypeAttr::get(
        tfrt::corert::Quint8Type::get(builder_.getContext()));
  }

  if (type.isa<mlir::TF::Quint16Type>()) {
    return mlir::TypeAttr::get(
        tfrt::corert::Quint16Type::get(builder_.getContext()));
  }

  if (type.isa<mlir::TF::Qint8Type>()) {
    return mlir::TypeAttr::get(
        tfrt::corert::Qint8Type::get(builder_.getContext()));
  }

  if (type.isa<mlir::TF::Qint16Type>()) {
    return mlir::TypeAttr::get(
        tfrt::corert::Qint16Type::get(builder_.getContext()));
  }

  if (type.isa<mlir::TF::Qint32Type>()) {
    return mlir::TypeAttr::get(
        tfrt::corert::Qint32Type::get(builder_.getContext()));
  }

  // Return invalid results to emit error for unsupported types.
  return {};
}

}  // namespace tensorflow
