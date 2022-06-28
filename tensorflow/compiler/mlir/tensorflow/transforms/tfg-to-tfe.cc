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
class MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStfgSLtoSLtfeDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStfgSLtoSLtfeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStfgSLtoSLtfeDTcc() {
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

#include "absl/strings/match.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/BlockAndValueMapping.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/Visitors.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/dialect_registration.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/passes_detail.h"
#include "tensorflow/core/transforms/toposort/toposort_pass.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace mlir {
namespace TF {
namespace {

// FIXME: This should be consistent with
// tensorflow::kImportModelDefaultGraphFuncName
static const char kImportModelDefaultGraphFuncName[] = "main";

// Please refer to the TFG dialect description for the list of used attributes.
// Belows are the attributes in TFE.
// TFE Arguments and Results (Got from "_Arg",
// "_Retval", .etc)
//  NodeDef.device <-> "tf.device"
//  NodeDef.attr <-> "tf."
//
// TFE general operations
//  NodeDef.device <-> "device"
//
// The following two functions are only used for mapping/excluding attributes
// which are inconsistent between TFG and TFE.
//
static mlir::LogicalResult FilterTfgSpecificArgResultAttributes(
    mlir::MLIRContext *context, mlir::ArrayRef<Type> types,
    mlir::ArrayAttr array_attr, llvm::SmallVector<mlir::Type> &output_types,
    llvm::SmallVector<mlir::DictionaryAttr> &output_attrs) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStfgSLtoSLtfeDTcc mht_0(mht_0_v, 228, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tfg-to-tfe.cc", "FilterTfgSpecificArgResultAttributes");

  for (auto it : llvm::zip(
           types, array_attr.template getAsRange<mlir::DictionaryAttr>())) {
    if (std::get<0>(it).isa<tfg::ControlType>()) continue;
    output_types.push_back(std::get<0>(it));

    mlir::NamedAttrList list;
    for (mlir::NamedAttribute attr : std::get<1>(it).getValue()) {
      // Skip if the attribute has "tfg" prefix.
      if (attr.getName().getValue().startswith("tfg")) continue;
      list.append(attr);
    }
    output_attrs.push_back(list.getDictionary(context));
  }
  return mlir::success();
}

static mlir::LogicalResult ReformatOpAttributes(
    mlir::MLIRContext *context, llvm::ArrayRef<mlir::NamedAttribute> attrs,
    llvm::SmallVectorImpl<mlir::NamedAttribute> &output) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStfgSLtoSLtfeDTcc mht_1(mht_1_v, 250, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tfg-to-tfe.cc", "ReformatOpAttributes");

  for (mlir::NamedAttribute attr : attrs) {
    if (attr.getName().strref().contains(
            mlir::tfg::TFGraphDialect::getDeviceAttrKey())) {
      tensorflow::DeviceNameUtils::ParsedName parsed_name;
      if (!tensorflow::DeviceNameUtils::ParseFullName(
              attr.getValue().cast<mlir::StringAttr>().getValue().str(),
              &parsed_name))
        return mlir::failure();
      if (!parsed_name.has_type) {
        parsed_name.type = "CPU";
        parsed_name.has_type = true;
      }
      if (!parsed_name.has_id) {
        parsed_name.id = 0;
        parsed_name.has_id = true;
      }
      output.push_back(mlir::NamedAttribute(
          mlir::StringAttr::get(context, "device"),
          mlir::StringAttr::get(
              context,
              tensorflow::DeviceNameUtils::ParsedNameToString(parsed_name))));
    } else {
      output.push_back(attr);
    }
  }
  return mlir::success();
}

static void FilterOutBlockArgControlDep(
    ValueRange operands, llvm::SmallVectorImpl<Value> &filtered) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStfgSLtoSLtfeDTcc mht_2(mht_2_v, 283, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tfg-to-tfe.cc", "FilterOutBlockArgControlDep");

  for (Value value : operands)
    if (!value.isa<mlir::BlockArgument>()) filtered.push_back(value);
}

// Split the tfg.NextIteration into tf_executor::NextIterationSourceOp and
// tf_executor::NextIterationSinkOp to break the cycle introduced by itself.
static void SplitNextIteration(Block &block) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStfgSLtoSLtfeDTcc mht_3(mht_3_v, 293, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tfg-to-tfe.cc", "SplitNextIteration");

  // TODO(b/207144333): Supports callback for unregistered ops
  block.walk([&](Operation *op) {
    if (!op->getName().getStringRef().equals("tfg.NextIteration")) return;
    mlir::OpBuilder builder(op);

    llvm::SmallVector<Value, 2> new_operands;
    FilterOutBlockArgControlDep(op->getOperands().drop_front(), new_operands);

    auto source_op = builder.create<tf_executor::NextIterationSourceOp>(
        op->getLoc(), op->getOperand(0).getType());
    builder.create<tf_executor::NextIterationSinkOp>(
        op->getLoc(), source_op.token(), /*input=*/op->getOperand(0),
        /*controlInputs=*/new_operands);
    op->replaceAllUsesWith(
        ValueRange({source_op.output(), source_op.control()}));
    op->erase();
  });
}

class ConvertGraphOp : public OpConversionPattern<tfg::GraphOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      tfg::GraphOp graph, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStfgSLtoSLtfeDTcc mht_4(mht_4_v, 322, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tfg-to-tfe.cc", "matchAndRewrite");

    Location loc = graph.getLoc();
    // To keep the import-as-graph logic taken by TFG, we create `void func()`
    // to contain the ops in the tfg::GraphOp. That means the arguments/results
    // will be the operations inside the function body rather than representing
    // them in the function signature.
    FunctionType func_type = rewriter.getFunctionType({}, {});
    FuncOp func = rewriter.create<FuncOp>(loc, kImportModelDefaultGraphFuncName,
                                          func_type);
    rewriter.setInsertionPointToStart(func.addEntryBlock());
    auto executor_graph =
        rewriter.create<tf_executor::GraphOp>(loc, func_type.getResults());
    rewriter.inlineRegionBefore(graph.nodes(), executor_graph.body(),
                                executor_graph.body().end());

    // Add terminator of tf_executor::graph
    rewriter.setInsertionPointToEnd(&executor_graph.body().front());
    rewriter.create<tf_executor::FetchOp>(loc);

    // Add terminator of func
    rewriter.setInsertionPointToEnd(&func.getBody().front());
    rewriter.create<func::ReturnOp>(loc);

    rewriter.replaceOp(graph.getOperation(), func.getOperation()->getResults());

    return success();
  }
};

class ConvertGraphFuncOp : public OpConversionPattern<tfg::GraphFuncOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      tfg::GraphFuncOp graph_func, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStfgSLtoSLtfeDTcc mht_5(mht_5_v, 360, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tfg-to-tfe.cc", "matchAndRewrite");

    assert(!graph_func.generic());
    Location loc = graph_func.getLoc();
    FunctionType ftype = graph_func.getFunctionType();

    FuncOp func = rewriter.create<FuncOp>(
        graph_func.getLoc(),
        graph_func->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
            .getValue(),
        ftype);

    func->setAttrs(graph_func->getAttrs());

    llvm::SmallVector<Type> arg_types;
    llvm::SmallVector<Type> res_types;
    llvm::SmallVector<DictionaryAttr> arg_attrs;
    llvm::SmallVector<DictionaryAttr> res_attrs;
    if (failed(FilterTfgSpecificArgResultAttributes(
            getContext(), ftype.getInputs(), graph_func.getAllArgAttrs(),
            arg_types, arg_attrs)) ||
        failed(FilterTfgSpecificArgResultAttributes(
            getContext(), ftype.getResults(), graph_func.getAllResultAttrs(),
            res_types, res_attrs)))
      return failure();

    // Update the function type which has excluded the control args.
    func->setAttr("function_type", TypeAttr::get(rewriter.getFunctionType(
                                       arg_types, res_types)));

    // Update arg/result attributes.
    func.setAllArgAttrs(arg_attrs);
    func.setAllResultAttrs(res_attrs);

    rewriter.setInsertionPointToStart(func.addEntryBlock());
    // In TFE, the function body is inlined in a GraphOp. Create a GraphOp
    // instance and move the regions from GraphFuncOp to GraphOp.
    auto executor_graph = rewriter.create<tf_executor::GraphOp>(
        loc, func.getFunctionType().getResults());

    // Replace the uses of block arguments with function arguments. Note that we
    // can't erase the arguments here because the operations may still use them
    // and these uses will be dropped after legalization of each op.
    unsigned idx = 0;
    Block &block = graph_func.body().front();
    for (auto iter = block.args_begin(), end_iter = block.args_end();
         iter != end_iter; ++iter) {
      if (!iter->getType().isa<tfg::ControlType>())
        iter->replaceAllUsesWith(func.getBody().getArgument(idx++));
    }

    rewriter.inlineRegionBefore(graph_func.body(), executor_graph.body(),
                                executor_graph.body().end());

    rewriter.setInsertionPointToEnd(&func.getBody().front());
    rewriter.create<func::ReturnOp>(
        loc, executor_graph.getOperation()->getResults());

    rewriter.replaceOp(graph_func.getOperation(),
                       func.getOperation()->getResults());

    return success();
  }
};

class ConvertReturnOp : public OpConversionPattern<tfg::ReturnOp> {
 public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      tfg::ReturnOp ret, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStfgSLtoSLtfeDTcc mht_6(mht_6_v, 432, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tfg-to-tfe.cc", "matchAndRewrite");

    rewriter.replaceOpWithNewOp<tf_executor::FetchOp>(ret.getOperation(),
                                                      adaptor.getOperands());
    return success();
  }
};

class ConvertControlTriggerOp : public ConversionPattern {
 public:
  explicit ConvertControlTriggerOp(MLIRContext *context)
      : ConversionPattern("tfg.ControlTrigger", PatternBenefit(1), context) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStfgSLtoSLtfeDTcc mht_7(mht_7_v, 445, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tfg-to-tfe.cc", "ConvertControlTriggerOp");
}

  LogicalResult matchAndRewrite(
      Operation *op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStfgSLtoSLtfeDTcc mht_8(mht_8_v, 452, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tfg-to-tfe.cc", "matchAndRewrite");

    llvm::SmallVector<Type, 2> new_types(op->getResultTypes());
    new_types.back() = rewriter.getType<tf_executor::ControlType>();

    llvm::SmallVector<Value, 2> new_operands;
    FilterOutBlockArgControlDep(operands, new_operands);

    rewriter.replaceOpWithNewOp<tf_executor::ControlTriggerOp>(
        op, new_types, new_operands, op->getAttrs());
    return success();
  }
};

class ConvertEnterOp : public ConversionPattern {
 public:
  explicit ConvertEnterOp(MLIRContext *context)
      : ConversionPattern("tfg.Enter", PatternBenefit(1), context) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStfgSLtoSLtfeDTcc mht_9(mht_9_v, 471, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tfg-to-tfe.cc", "ConvertEnterOp");
}

  LogicalResult matchAndRewrite(
      Operation *op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStfgSLtoSLtfeDTcc mht_10(mht_10_v, 478, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tfg-to-tfe.cc", "matchAndRewrite");

    llvm::SmallVector<Type, 2> new_types(op->getResultTypes());
    new_types.back() = rewriter.getType<tf_executor::ControlType>();

    llvm::SmallVector<Value, 2> new_operands;
    FilterOutBlockArgControlDep(operands, new_operands);

    rewriter.replaceOpWithNewOp<tf_executor::EnterOp>(
        op, new_types, new_operands, op->getAttrs());
    return success();
  }
};

class ConvertExitOp : public ConversionPattern {
 public:
  explicit ConvertExitOp(MLIRContext *context)
      : ConversionPattern("tfg.Exit", PatternBenefit(1), context) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStfgSLtoSLtfeDTcc mht_11(mht_11_v, 497, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tfg-to-tfe.cc", "ConvertExitOp");
}

  LogicalResult matchAndRewrite(
      Operation *op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStfgSLtoSLtfeDTcc mht_12(mht_12_v, 504, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tfg-to-tfe.cc", "matchAndRewrite");

    llvm::SmallVector<Type, 2> new_types(op->getResultTypes());
    new_types.back() = rewriter.getType<tf_executor::ControlType>();

    llvm::SmallVector<Value, 2> new_operands;
    FilterOutBlockArgControlDep(operands, new_operands);

    rewriter.replaceOpWithNewOp<tf_executor::ExitOp>(
        op, new_types, new_operands, op->getAttrs());
    return success();
  }
};

class ConvertLoopCondOp : public ConversionPattern {
 public:
  explicit ConvertLoopCondOp(MLIRContext *context)
      : ConversionPattern("tfg.LoopCond", PatternBenefit(1), context) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStfgSLtoSLtfeDTcc mht_13(mht_13_v, 523, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tfg-to-tfe.cc", "ConvertLoopCondOp");
}

  LogicalResult matchAndRewrite(
      Operation *op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStfgSLtoSLtfeDTcc mht_14(mht_14_v, 530, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tfg-to-tfe.cc", "matchAndRewrite");

    llvm::SmallVector<Type, 2> new_types(op->getResultTypes());
    new_types.back() = rewriter.getType<tf_executor::ControlType>();

    llvm::SmallVector<Value, 2> new_operands;
    FilterOutBlockArgControlDep(operands, new_operands);

    rewriter.replaceOpWithNewOp<tf_executor::LoopCondOp>(
        op, new_types, new_operands, op->getAttrs());
    return success();
  }
};

class ConvertMergeOp : public ConversionPattern {
 public:
  explicit ConvertMergeOp(MLIRContext *context)
      : ConversionPattern("tfg.Merge", PatternBenefit(1), context) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStfgSLtoSLtfeDTcc mht_15(mht_15_v, 549, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tfg-to-tfe.cc", "ConvertMergeOp");
}

  LogicalResult matchAndRewrite(
      Operation *op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStfgSLtoSLtfeDTcc mht_16(mht_16_v, 556, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tfg-to-tfe.cc", "matchAndRewrite");

    llvm::SmallVector<Type, 2> new_types(op->getResultTypes());
    new_types.back() = rewriter.getType<tf_executor::ControlType>();

    llvm::SmallVector<Value, 2> new_operands;
    FilterOutBlockArgControlDep(operands, new_operands);

    rewriter.replaceOpWithNewOp<tf_executor::MergeOp>(
        op, new_types, new_operands, op->getAttrs());
    return success();
  }
};

class ConvertSwitchOp : public ConversionPattern {
 public:
  explicit ConvertSwitchOp(MLIRContext *context)
      : ConversionPattern("tfg.Switch", PatternBenefit(1), context) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStfgSLtoSLtfeDTcc mht_17(mht_17_v, 575, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tfg-to-tfe.cc", "ConvertSwitchOp");
}

  LogicalResult matchAndRewrite(
      Operation *op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStfgSLtoSLtfeDTcc mht_18(mht_18_v, 582, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tfg-to-tfe.cc", "matchAndRewrite");

    llvm::SmallVector<Type, 2> new_types(op->getResultTypes());
    new_types.back() = rewriter.getType<tf_executor::ControlType>();

    llvm::SmallVector<Value, 2> new_operands;
    FilterOutBlockArgControlDep(operands, new_operands);

    rewriter.replaceOpWithNewOp<tf_executor::SwitchOp>(
        op, new_types, new_operands, op->getAttrs());
    return success();
  }
};

class ConvertSwitchNOp : public ConversionPattern {
 public:
  explicit ConvertSwitchNOp(MLIRContext *context)
      : ConversionPattern("tfg.SwitchN", PatternBenefit(1), context) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStfgSLtoSLtfeDTcc mht_19(mht_19_v, 601, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tfg-to-tfe.cc", "ConvertSwitchNOp");
}

  LogicalResult matchAndRewrite(
      Operation *op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStfgSLtoSLtfeDTcc mht_20(mht_20_v, 608, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tfg-to-tfe.cc", "matchAndRewrite");

    llvm::SmallVector<Type, 2> new_types(op->getResultTypes());
    new_types.back() = rewriter.getType<tf_executor::ControlType>();

    llvm::SmallVector<Value, 2> new_operands;
    FilterOutBlockArgControlDep(operands, new_operands);

    rewriter.replaceOpWithNewOp<tf_executor::SwitchNOp>(
        op, new_types, new_operands, op->getAttrs());
    return success();
  }
};

class ConvertGeneralOp : public ConversionPattern {
 public:
  ConvertGeneralOp(MLIRContext *context,
                   const DenseSet<StringRef> &func_symbols)
      : ConversionPattern(MatchAnyOpTypeTag(), PatternBenefit(1), context),
        func_symbols_(func_symbols) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStfgSLtoSLtfeDTcc mht_21(mht_21_v, 629, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tfg-to-tfe.cc", "ConvertGeneralOp");
}

  LogicalResult matchAndRewrite(
      Operation *op, llvm::ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStfgSLtoSLtfeDTcc mht_22(mht_22_v, 636, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tfg-to-tfe.cc", "matchAndRewrite");

    if (!llvm::isa<tfg::TFGraphDialect>(op->getDialect())) return failure();

    Location loc = op->getLoc();
    llvm::SmallVector<mlir::Type, 2> new_types(op->getResultTypes());
    // Update the control type from tf_type.control to tf_executor.control.
    new_types.back() = rewriter.getType<tf_executor::ControlType>();

    // Control operand is attached on tf_executor::IslandOp.
    llvm::SmallVector<Value> island_control_operands;
    llvm::SmallVector<Value> inner_op_operands;

    for (Value value : operands) {
      // Because of the property of graph region, the control operands may
      // not have been converted to tf_executor::ControlType.
      if (value.getType().isa<tfg::ControlType>() ||
          value.getType().isa<tf_executor::ControlType>()) {
        if (!value.isa<BlockArgument>())
          island_control_operands.push_back(value);
      } else {
        inner_op_operands.push_back(value);
      }
    }

    auto island = rewriter.create<tf_executor::IslandOp>(
        loc, new_types, island_control_operands);
    island.body().push_back(new mlir::Block);

    rewriter.setInsertionPointToEnd(&island.body().front());

    // Control dependency has been applied on tf_executor.island. Remove it
    // while creating the tf operations.
    new_types.pop_back();

    llvm::SmallVector<std::unique_ptr<Region>, 1> new_regions;
    for (auto &region : op->getRegions()) {
      new_regions.push_back(std::make_unique<Region>());
      new_regions.back()->takeBody(region);
    }

    llvm::SmallVector<NamedAttribute, 4> attrs;
    if (failed(ReformatOpAttributes(getContext(), op->getAttrs(), attrs)))
      return failure();

    Operation *inner_op;

    StringRef op_name = op->getName().stripDialect();
    if (!func_symbols_.contains(op_name)) {
      std::string tf_op_name = llvm::formatv(
          "{0}.{1}", TF::TensorFlowDialect::getDialectNamespace(), op_name);
      OperationState state =
          OperationState(loc, tf_op_name, inner_op_operands, new_types, attrs,
                         op->getSuccessors(), new_regions);
      inner_op = rewriter.create(state);
    } else {
      bool disable_call_shape_inference = false;
      if (op->hasAttr("_disable_call_shape_inference")) {
        disable_call_shape_inference =
            op->getAttrOfType<BoolAttr>("_disable_call_shape_inference")
                .getValue();
      }
      inner_op =
          rewriter.create<LegacyCallOp>(loc, new_types, inner_op_operands,
                                        op_name, disable_call_shape_inference);
    }

    rewriter.create<tf_executor::YieldOp>(loc, inner_op->getResults());

    rewriter.replaceOp(op, island.getOperation()->getResults());

    return success();
  }

 private:
  const DenseSet<StringRef> &func_symbols_;
};

class LegalizeTFGToTFE : public TF::LegalizeTFGToTFPassBase<LegalizeTFGToTFE> {
  void getDependentDialects(DialectRegistry &registry) const override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStfgSLtoSLtfeDTcc mht_23(mht_23_v, 717, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tfg-to-tfe.cc", "getDependentDialects");

    RegisterAllTensorFlowDialects(registry);
  }

  void runOnOperation() override;
};

}  // namespace

void LegalizeTFGToTFE::runOnOperation() {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStensorflowPStransformsPStfgSLtoSLtfeDTcc mht_24(mht_24_v, 729, "", "./tensorflow/compiler/mlir/tensorflow/transforms/tfg-to-tfe.cc", "LegalizeTFGToTFE::runOnOperation");

  MLIRContext &context = getContext();
  ModuleOp module = getOperation();

  DenseSet<StringRef> func_symbols;
  for (auto &op : module.getBodyRegion().getOps()) {
    if (auto func = llvm::dyn_cast<tfg::GraphFuncOp>(op)) {
      func_symbols.insert(
          func->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName())
              .getValue());
    }
  }

  ConversionTarget target(context);
  target.addLegalDialect<TF::TensorFlowDialect>();
  target.addLegalDialect<tf_executor::TensorFlowExecutorDialect>();
  target.addLegalOp<ModuleOp>();
  target.addLegalOp<FuncOp>();
  target.addLegalOp<func::ReturnOp>();

  RewritePatternSet patterns(&context);
  patterns.add<ConvertGraphOp>(&context);
  patterns.add<ConvertGraphFuncOp>(&context);
  patterns.add<ConvertReturnOp>(&context);
  patterns.add<ConvertGeneralOp>(&context, func_symbols);
  // Control flow V1 operation conversion patterns.
  patterns.add<ConvertControlTriggerOp>(&context);
  patterns.add<ConvertEnterOp>(&context);
  patterns.add<ConvertExitOp>(&context);
  patterns.add<ConvertLoopCondOp>(&context);
  patterns.add<ConvertMergeOp>(&context);
  patterns.add<ConvertSwitchOp>(&context);
  patterns.add<ConvertSwitchNOp>(&context);
  FrozenRewritePatternSet finalPatterns(std::move(patterns));

  // Turn the graph region into SSACFG region by applying an order to the
  // operations.
  for (auto &op : module.getBodyRegion().getOps()) {
    for (auto &region : op.getRegions()) {
      for (auto &block : region) {
        // Split tfg.NextIteration to break the cycle.
        SplitNextIteration(block);
        tfg::SortTopologically(&block);
      }
    }
  }

  // Version information is embedded in graph operation in TFG. In TFE, it's
  // embedded in the module operation.
  for (auto &op : module.getBodyRegion().getOps()) {
    auto graph = dyn_cast<tfg::GraphOp>(op);
    if (!graph) continue;
    Builder b(&context);
    auto producer = b.getNamedAttr(
        "producer", b.getI32IntegerAttr(graph.version().getProducer()));
    auto min_consumer = b.getNamedAttr(
        "min_consumer", b.getI32IntegerAttr(graph.version().getMinConsumer()));
    auto bad_consumers = b.getNamedAttr(
        "bad_consumers", b.getI32ArrayAttr(graph.version().getBadConsumers()));
    module->setAttr("tf.versions",
                    b.getDictionaryAttr(llvm::ArrayRef<NamedAttribute>(
                        {producer, min_consumer, bad_consumers})));
    break;
  }

  if (failed(applyFullConversion(module.getOperation(), target, finalPatterns)))
    signalPassFailure();

  // The uses of arg control dependency has been dropped. We can safely remove
  // the block argument here.
  module.walk([&](tf_executor::GraphOp graph) {
    graph.body().front().eraseArguments([](BlockArgument arg) { return true; });
  });
}

std::unique_ptr<Pass> CreateLegalizeTFGToTFEPass() {
  return std::make_unique<LegalizeTFGToTFE>();
}

}  // end namespace TF
}  // end namespace mlir
