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
class MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc {
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
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc() {
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

#include "tensorflow/core/transforms/consolidate_attrs/pass.h"

#include <memory>
#include <utility>

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/Sequence.h"
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/core/ir/dialect.h"
#include "tensorflow/core/ir/ops.h"
#include "tensorflow/core/ir/tf_op_wrapper.h"
#include "tensorflow/core/ir/types/dialect.h"
#include "tensorflow/core/ir/utility.h"
#include "tensorflow/core/transforms/pass_detail.h"

namespace mlir {
namespace tfg {

static const char *kRegenerateOutputShapes = "tfg.regenerate_output_shapes";

// Returns true if an attribute is an array of shapes;
static bool IsArrayOfShapes(ArrayAttr array) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_0(mht_0_v, 212, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "IsArrayOfShapes");

  return llvm::all_of(array,
                      [](Attribute attr) { return attr.isa<ShapeAttr>(); });
}

// Given a tensor type and shape information, try to refine the type.
static Type GetReifiedType(Type orig, ShapeAttr shape) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_1(mht_1_v, 221, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "GetReifiedType");

  Type element_type = orig.cast<ShapedType>().getElementType();
  TensorType inferred;
  if (shape.hasRank()) {
    inferred = RankedTensorType::get(shape.getShape(), element_type);
  } else {
    inferred = UnrankedTensorType::get(element_type);
  }
  Type reified_type = tf_type::GetCastCompatibleType(inferred, orig);
  // If the types are not compatible, return the original type.
  return reified_type ? reified_type : orig;
}

namespace {
// CRTP base class for consolidate attribute passes. This base class defines
// cached identifiers for the attributes.
template <typename PassT>
class AttributesPassBase : public PassWrapper<PassT, OperationPass<>> {
 public:
  LogicalResult initialize(MLIRContext *context) override {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_2(mht_2_v, 243, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "initialize");

    input_shapes_id_ = StringAttr::get(context, "tf._input_shapes");
    regenerate_input_shapes_id_ =
        StringAttr::get(context, "tfg.regenerate_input_shapes");
    output_shapes_id_ = StringAttr::get(context, "tf._output_shapes");
    regenerate_output_shapes_id_ =
        StringAttr::get(context, "tfg.regenerate_output_shapes");
    handle_data_id_ = StringAttr::get(context, "tfg.handle_data");
    dtype_id_ = StringAttr::get(context, "tfg.dtype");
    is_ref_id_ = StringAttr::get(context, "tfg.is_ref");
    control_type_ = ControlType::get(context);
    return success();
  }

 protected:
  // Identifier for `tf._input_shapes`.
  StringAttr input_shapes_id_;
  // Identifier for `tf._regenerate_input_shapes`.
  StringAttr regenerate_input_shapes_id_;
  // Identifier for `tf._output_shapes`.
  StringAttr output_shapes_id_;
  // Identifier for `tf._regenerate_output_shapes`.
  StringAttr regenerate_output_shapes_id_;
  // Identifier for `tfg.handle_data`.
  StringAttr handle_data_id_;
  // Identifier for `tfg.dtype`.
  StringAttr dtype_id_;
  // Identifier for `tfg.is_ref`.
  StringAttr is_ref_id_;
  // Cacched control type.
  ControlType control_type_;
};

class ConsolidateAttributesPassImpl
    : public AttributesPassBase<ConsolidateAttributesPassImpl> {
 public:
  void runOnOperation() override;

 private:
  // Reify `tf._input_shapes`, `tf._output_shapes` and `tfg.handle_data` into
  // the types of the function arguments. Drop the attributes `tfg.dtype` and
  // `tfg.is_ref`. Return the the new argument attributes.
  ArrayAttr reifyAndDropFunctionArgumentAttributes(GraphFuncOp func);
  // Reify `tf._output_shapes` and `tfg.handle_data` into the types of the
  // function results. Drop the attribute `tfg.dtype`. Return the new result
  // attributes.
  ArrayAttr reifyAndDropFunctionResultAttributes(GraphFuncOp func);

  // Refine a type with `tf._output_shapes`.
  Type refineTypeWithOutputShapes(Type type, Attribute output_shapes_attr);
  // Refine a type with `tfg.handle_data`.
  Type refineTypeWithHandleData(Type type, Attribute handle_data);
};
}  // namespace

Type ConsolidateAttributesPassImpl::refineTypeWithOutputShapes(
    Type type, Attribute output_shapes_attr) {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_3(mht_3_v, 302, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "ConsolidateAttributesPassImpl::refineTypeWithOutputShapes");

  auto output_shapes = output_shapes_attr.dyn_cast_or_null<ArrayAttr>();
  if (!output_shapes || output_shapes.size() != 1 ||
      !IsArrayOfShapes(output_shapes))
    return type;
  return GetReifiedType(type, output_shapes[0].cast<ShapeAttr>());
}

Type ConsolidateAttributesPassImpl::refineTypeWithHandleData(
    Type type, Attribute handle_data) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_4(mht_4_v, 314, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "ConsolidateAttributesPassImpl::refineTypeWithHandleData");

  if (!handle_data) return type;
  SmallVector<TensorType> subtypes;
  // Because `tfg.handle_data` is a TFG internal attribute, it will be
  // well-formed.
  for (Type type : handle_data.cast<ArrayAttr>().getAsValueRange<TypeAttr>())
    subtypes.push_back(type.cast<TensorType>());
  auto resource =
      UnrankedTensorType::get(ResourceType::get(subtypes, &getContext()));
  Type reified = tf_type::GetCastCompatibleType(resource, type);
  return reified ? reified : type;
}

ArrayAttr ConsolidateAttributesPassImpl::reifyAndDropFunctionArgumentAttributes(
    GraphFuncOp func) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_5(mht_5_v, 331, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "ConsolidateAttributesPassImpl::reifyAndDropFunctionArgumentAttributes");

  // Get the input shapes attribute if there is one. Only use if the number of
  // shapes matches the number of arguments.
  ArrayAttr input_shapes =
      func->removeAttr(input_shapes_id_).dyn_cast_or_null<ArrayAttr>();
  unsigned num_args = func.getNumArguments() / 2;
  if (input_shapes) {
    if (input_shapes.size() != num_args || !IsArrayOfShapes(input_shapes)) {
      input_shapes = {};
    } else {
      func->setAttr(regenerate_input_shapes_id_, UnitAttr::get(&getContext()));
    }
  }

  SmallVector<Attribute> arg_attrs;
  auto empty_dict = DictionaryAttr::get(&getContext());
  for (auto i : llvm::seq<unsigned>(0, num_args)) {
    BlockArgument arg = GraphFuncOp::getDataValue(func.body(), i);
    NamedAttrList attrs(func.getArgAttrs(arg.getArgNumber()));
    Type arg_type = arg.getType();
    if (Attribute output_shapes = attrs.erase(output_shapes_id_)) {
      arg_type = refineTypeWithOutputShapes(arg_type, output_shapes);
      attrs.set(regenerate_output_shapes_id_, UnitAttr::get(&getContext()));
    }
    arg_type = refineTypeWithHandleData(arg_type, attrs.erase(handle_data_id_));
    if (input_shapes)
      arg_type = GetReifiedType(arg_type, input_shapes[i].cast<ShapeAttr>());
    arg.setType(arg_type);
    attrs.erase(dtype_id_);
    attrs.erase(is_ref_id_);
    arg_attrs.append({attrs.getDictionary(&getContext()), empty_dict});
  }
  return ArrayAttr::get(&getContext(), arg_attrs);
}

ArrayAttr ConsolidateAttributesPassImpl::reifyAndDropFunctionResultAttributes(
    GraphFuncOp func) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_6(mht_6_v, 370, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "ConsolidateAttributesPassImpl::reifyAndDropFunctionResultAttributes");

  SmallVector<Attribute> ret_attrs;
  // The result types are propagated to the data operands to `return`.
  auto ret_op = cast<ReturnOp>(func.body().front().getTerminator());
  for (auto &it :
       llvm::enumerate(func.getAllResultAttrs().getAsRange<DictionaryAttr>())) {
    NamedAttrList attrs(it.value());
    Value ret = ret_op.getOperand(it.index());
    Type ret_type = ret.getType();
    if (Attribute output_shapes = attrs.erase(output_shapes_id_)) {
      ret_type = refineTypeWithOutputShapes(ret_type, output_shapes);
      attrs.set(regenerate_output_shapes_id_, UnitAttr::get(&getContext()));
    }
    ret_type = refineTypeWithHandleData(ret_type, attrs.erase(handle_data_id_));
    ret.setType(ret_type);
    attrs.erase(dtype_id_);
    ret_attrs.push_back(attrs.getDictionary(&getContext()));
  }
  return ArrayAttr::get(&getContext(), ret_attrs);
}

namespace {
// This pattern reifies an op's result shape info into the result types and
// drops the output shapes attributes.
class ReifyOperationOutputShapes : public RewritePattern {
 public:
  ReifyOperationOutputShapes(MLIRContext *context, PatternBenefit benefit,
                             StringRef attr_name)
      : RewritePattern(Pattern::MatchAnyOpTypeTag(), benefit, context),
        output_shapes_id_(StringAttr::get(context, attr_name)) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_7(mht_7_v, 402, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "ReifyOperationOutputShapes");
}

  // Returns true if this instance of the pattern should match the op.
  virtual bool shouldMatch(Operation *op) const = 0;

  LogicalResult match(Operation *op) const override {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_8(mht_8_v, 410, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "match");

    return success(shouldMatch(op) && op->hasAttr(output_shapes_id_));
  }

  void rewrite(Operation *op, PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_9(mht_9_v, 417, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "rewrite");

    rewriter.updateRootInPlace(op, [&] {
      auto output_shapes =
          op->removeAttr(output_shapes_id_).dyn_cast_or_null<ArrayAttr>();
      ResultRange results = TFOp(op).getNonControlResults();
      if (!output_shapes || results.size() != output_shapes.size() ||
          !IsArrayOfShapes(output_shapes))
        return;

      for (auto it :
           llvm::zip(results, output_shapes.getAsRange<ShapeAttr>())) {
        Value result = std::get<0>(it);
        result.setType(GetReifiedType(result.getType(), std::get<1>(it)));
      }
      rewriteImpl(op, rewriter);
    });
  }

  virtual void rewriteImpl(Operation *op, PatternRewriter &rewriter) const {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_10(mht_10_v, 438, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "rewriteImpl");
}

 private:
  // Identifier for `_output_shapes`.
  StringAttr output_shapes_id_;
};

// This pattern matches and TFG op and reifies `_output_shapes`. The pattern
// leaves behind an attribute `_regenerate_output_shapes` that is used by the
// converse pattern to detect whether the attribute should be materialized.
class ReifyTFGOpOutputShapes : public ReifyOperationOutputShapes {
 public:
  explicit ReifyTFGOpOutputShapes(MLIRContext *context)
      : ReifyOperationOutputShapes(context, /*benefit=*/1, "_output_shapes"),
        dialect_(context->getOrLoadDialect<TFGraphDialect>()),
        regenerate_output_shapes_id_(
            StringAttr::get(context, kRegenerateOutputShapes)) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_11(mht_11_v, 457, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "ReifyTFGOpOutputShapes");
}

  bool shouldMatch(Operation *op) const override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_12(mht_12_v, 462, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "shouldMatch");

    return op->getDialect() == dialect_;
  }

  void rewriteImpl(Operation *op, PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_13(mht_13_v, 469, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "rewriteImpl");

    op->setAttr(regenerate_output_shapes_id_, rewriter.getUnitAttr());
  }

 private:
  // Cached TFG dialect instance.
  TFGraphDialect *dialect_;
  // Identifier to `_regenerate_output_shapes`.
  StringAttr regenerate_output_shapes_id_;
};

// This pattern matches `If`, `Case`, and `While` and reifies their
// `output_shapes` attribute.
struct ReifyCFOpOutputShapes : public ReifyOperationOutputShapes {
  // Set a higher benefit to ensure that "output_shapes" is reified before
  // "_output_shapes".
  explicit ReifyCFOpOutputShapes(MLIRContext *context)
      : ReifyOperationOutputShapes(context, /*benefit=*/2, "output_shapes") {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_14(mht_14_v, 489, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "ReifyCFOpOutputShapes");
}

  bool shouldMatch(Operation *op) const override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_15(mht_15_v, 494, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "shouldMatch");

    return isa<IfOp, StatelessIfOp, StatefulIfOp, CaseOp, StatelessCaseOp,
               StatefulCaseOp, WhileOp, StatelessWhileOp, StatefulWhileOp>(op);
  }
};

// This pattern removes a list of attributes from the given op types.
template <typename... OpTs>
class DropAttributes : public RewritePattern {
 public:
  // Create the pattern. Specify which attributes to remove.
  DropAttributes(MLIRContext *context, ArrayRef<StringRef> attr_names)
      : RewritePattern(Pattern::MatchAnyOpTypeTag(), /*benefit=*/1, context) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_16(mht_16_v, 509, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "DropAttributes");

    for (StringRef attr_name : attr_names)
      attr_ids_.push_back(StringAttr::get(context, attr_name));
  }

  // Remove the specified attributes from the op. Fail if none of the attributes
  // were present.
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_17(mht_17_v, 520, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "matchAndRewrite");

    if (!isa<OpTs...>(op)) return failure();
    rewriter.startRootUpdate(op);
    if (!llvm::count_if(attr_ids_, [&](StringAttr attr_id) {
          return op->removeAttr(attr_id);
        })) {
      rewriter.cancelRootUpdate(op);
      return failure();
    }
    rewriter.finalizeRootUpdate(op);
    return success();
  }

 private:
  // The identifiers of the attributes to remove.
  SmallVector<StringAttr> attr_ids_;
};
}  // namespace

template <typename... OpTs>
static std::unique_ptr<RewritePattern> RemoveAttributes(
    MLIRContext *context, ArrayRef<StringRef> attr_names) {
  return std::make_unique<DropAttributes<OpTs...>>(context, attr_names);
}

void ConsolidateAttributesPassImpl::runOnOperation() {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_18(mht_18_v, 548, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "ConsolidateAttributesPassImpl::runOnOperation");

  // Reify operation attributes.
  RewritePatternSet patterns(&getContext());
  patterns.insert<ReifyTFGOpOutputShapes, ReifyCFOpOutputShapes>(&getContext());
  patterns.add(RemoveAttributes<IfOp, StatelessIfOp, StatefulIfOp>(
      &getContext(), {"Tcond", "Tin", "Tout"}));
  patterns.add(RemoveAttributes<CaseOp, StatelessCaseOp, StatefulCaseOp>(
      &getContext(), {"Tin", "Tout"}));
  patterns.add(
      RemoveAttributes<WhileOp, StatelessWhileOp, StatefulWhileOp, ForOp>(
          &getContext(), {"T"}));
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    getOperation()->emitError(getArgument() + " pass failed");
    signalPassFailure();
    return;
  }

  // If the pass was run on a function, reify its attributes and then rebuild
  // the signature. Because the attributes may have conflicting type info, the
  // order in which we visit the attributes is the priority.
  auto func = dyn_cast<GraphFuncOp>(getOperation());
  if (!func || func.generic()) return;
  ArrayAttr arg_attrs = reifyAndDropFunctionArgumentAttributes(func);
  ArrayAttr res_attrs = reifyAndDropFunctionResultAttributes(func);
  Block &body = func.body().front();
  auto type = FunctionType::get(
      &getContext(), body.getArgumentTypes(),
      TFOp(body.getTerminator()).getNonControlOperands().getTypes());
  NamedAttrList attrs(func->getAttrDictionary());
  attrs.set(func.function_typeAttrName(), TypeAttr::get(type));
  attrs.set(func.arg_attrsAttrName(), arg_attrs);
  attrs.set(func.res_attrsAttrName(), res_attrs);
  func->setAttrs(attrs.getDictionary(&getContext()));
}

namespace {
class PrepareAttributesForExportPassImpl
    : public AttributesPassBase<PrepareAttributesForExportPassImpl> {
 public:
  void runOnOperation() override;

 private:
  // Materialize required `tfg.` attributes for export. Also, adds
  // `tf._input_shapes` to the function attributes. And `tf._output_shapes` and
  // `tf._handle_data` to the argument and result attributes.
  void prepareFunctionAttributes(GraphFuncOp func);

  // Prepare attributes for a single type.
  DictionaryAttr prepareAttributesFor(Type type, DictionaryAttr attr_dict);
};
}  // namespace

void PrepareAttributesForExportPassImpl::prepareFunctionAttributes(
    GraphFuncOp func) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_19(mht_19_v, 605, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "PrepareAttributesForExportPassImpl::prepareFunctionAttributes");

  NamedAttrList attrs(func->getAttrDictionary());
  SmallVector<Attribute> input_shapes, arg_attrs, res_attrs;
  for (auto it :
       llvm::zip(func.getArgumentTypes(),
                 func.getAllArgAttrs().getAsRange<DictionaryAttr>())) {
    Type type = std::get<0>(it);
    DictionaryAttr attrs = std::get<1>(it);
    if (type == control_type_) {
      arg_attrs.push_back(attrs);
      continue;
    }
    arg_attrs.push_back(prepareAttributesFor(type, attrs));
    if (auto ranked = type.dyn_cast<RankedTensorType>()) {
      input_shapes.push_back(ShapeAttr::get(&getContext(), ranked.getShape()));
    } else {
      input_shapes.push_back(ShapeAttr::get(&getContext(), llvm::None));
    }
  }
  for (auto it :
       llvm::zip(func.getResultTypes(),
                 func.getAllResultAttrs().getAsRange<DictionaryAttr>()))
    res_attrs.push_back(prepareAttributesFor(std::get<0>(it), std::get<1>(it)));

  // Add input shapes only if its regeneration is required.
  if (attrs.erase(regenerate_input_shapes_id_))
    attrs.set(input_shapes_id_, ArrayAttr::get(&getContext(), input_shapes));
  attrs.set(func.arg_attrsAttrName(), ArrayAttr::get(&getContext(), arg_attrs));
  attrs.set(func.res_attrsAttrName(), ArrayAttr::get(&getContext(), res_attrs));
  func->setAttrs(attrs.getDictionary(&getContext()));
}

DictionaryAttr PrepareAttributesForExportPassImpl::prepareAttributesFor(
    Type type, DictionaryAttr attr_dict) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_20(mht_20_v, 641, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "PrepareAttributesForExportPassImpl::prepareAttributesFor");

  NamedAttrList attrs(attr_dict);
  // Add shape data if requested.
  if (attrs.erase(regenerate_output_shapes_id_)) {
    auto shape = ShapeAttr::get(&getContext(),
                                type.isa<RankedTensorType>()
                                    ? type.cast<RankedTensorType>().getShape()
                                    : Optional<ArrayRef<int64_t>>());
    attrs.set(output_shapes_id_, ArrayAttr::get(&getContext(), {shape}));
  }
  auto element_type = type.cast<TensorType>().getElementType();
  if (auto resource = element_type.dyn_cast<ResourceType>()) {
    SmallVector<Attribute> handle_data;
    for (TensorType subtype : resource.getSubtypes())
      handle_data.push_back(TypeAttr::get(subtype));
    // Only bother adding handle data if there are subtypes.
    if (!handle_data.empty())
      attrs.set(handle_data_id_, ArrayAttr::get(&getContext(), handle_data));
  }
  if (element_type.isa<tf_type::TensorFlowRefType>())
    attrs.set(is_ref_id_, UnitAttr::get(&getContext()));
  return attrs.getDictionary(&getContext());
}

// Get the element types of the values as an array attributes.
static ArrayAttr GetElementTypesAttr(PatternRewriter &rewriter,
                                     ValueRange values) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_21(mht_21_v, 670, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "GetElementTypesAttr");

  SmallVector<Attribute> types;
  for (Value value : values) {
    types.push_back(
        TypeAttr::get(value.getType().cast<TensorType>().getElementType()));
  }
  return rewriter.getArrayAttr(types);
}

namespace {
// Base class for patterns that materialize control-flow op attributes. This
// patterns contains a cached control type.
template <typename OpT>
class MaterializeAttrsPattern : public OpRewritePattern<OpT> {
 public:
  // Create the pattern with a cached control type instance.
  explicit MaterializeAttrsPattern(ControlType control_type)
      : OpRewritePattern<OpT>(control_type.getContext()),
        control_type_(control_type) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_22(mht_22_v, 691, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "MaterializeAttrsPattern");
}

  // Get an array of the element types of the data arguments of the op. The
  // arguments exclude "op-specific" operands such as if condition, case branch
  // index, and for loop indices.
  ArrayAttr getArgumentElementTypesAttr(PatternRewriter &rewriter,
                                        OpT op) const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_23(mht_23_v, 700, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "getArgumentElementTypesAttr");

    return GetElementTypesAttr(
        rewriter, SplitDataAndControlValues(op.args(), control_type_).first);
  }

 private:
  // The cached control type.
  ControlType control_type_;
};

template <typename IfLikeOp>
struct MaterializeIfAttrs : public MaterializeAttrsPattern<IfLikeOp> {
  using MaterializeAttrsPattern<IfLikeOp>::MaterializeAttrsPattern;

  // Materialize `Tcond`, `Tin`, and `Tout`.
  LogicalResult matchAndRewrite(IfLikeOp op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_24(mht_24_v, 719, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "matchAndRewrite");

    if (op.Tcond() && op.Tin() && op.Tout()) return failure();
    NamedAttrList attrs(op->getAttrDictionary());
    attrs.set(
        op.TcondAttrName(),
        TypeAttr::get(
            op.cond().getType().template cast<TensorType>().getElementType()));
    attrs.set(op.TinAttrName(),
              this->getArgumentElementTypesAttr(rewriter, op));
    attrs.set(op.ToutAttrName(), GetElementTypesAttr(rewriter, op.outs()));
    rewriter.updateRootInPlace(
        op, [&] { op->setAttrs(attrs.getDictionary(op->getContext())); });
    return success();
  }
};

template <typename CaseLikeOp>
struct MaterializeCaseAttrs : public MaterializeAttrsPattern<CaseLikeOp> {
  using MaterializeAttrsPattern<CaseLikeOp>::MaterializeAttrsPattern;

  // Materialize `Tin` and `Tout`.
  LogicalResult matchAndRewrite(CaseLikeOp op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_25(mht_25_v, 744, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "matchAndRewrite");

    if (op.Tin() && op.Tout()) return failure();
    NamedAttrList attrs(op->getAttrDictionary());
    attrs.set(op.TinAttrName(),
              this->getArgumentElementTypesAttr(rewriter, op));
    attrs.set(op.ToutAttrName(), GetElementTypesAttr(rewriter, op.outs()));
    rewriter.updateRootInPlace(
        op, [&] { op->setAttrs(attrs.getDictionary(op->getContext())); });
    return success();
  }
};

template <typename WhileOrForLikeOp>
struct MaterializeTAttr : public MaterializeAttrsPattern<WhileOrForLikeOp> {
  using MaterializeAttrsPattern<WhileOrForLikeOp>::MaterializeAttrsPattern;

  // Materialize `T`.
  LogicalResult matchAndRewrite(WhileOrForLikeOp op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_26(mht_26_v, 765, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "matchAndRewrite");

    if (op.T()) return failure();
    rewriter.updateRootInPlace(
        op, [&] { op.TAttr(this->getArgumentElementTypesAttr(rewriter, op)); });
    return success();
  }
};

// Base class for a pattern that
class MaterializeOutputShapesBase : public RewritePattern {
 public:
  explicit MaterializeOutputShapesBase(MLIRContext *context,
                                       StringRef attr_name)
      : RewritePattern(Pattern::MatchAnyOpTypeTag(), /*benefit=*/1, context),
        attr_id_(StringAttr::get(context, attr_name)) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_27(mht_27_v, 782, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "MaterializeOutputShapesBase");
}

  virtual bool shouldMatch(Operation *op) const = 0;

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_28(mht_28_v, 790, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "matchAndRewrite");

    // Exclude internal TFG ops.
    if (isa<ReturnOp>(op)) return failure();
    if (!shouldMatch(op) || op->hasAttr(attr_id_)) return failure();
    ResultRange results = TFOp(op).getNonControlResults();

    SmallVector<Attribute> shapes;
    for (Value result : results) {
      if (auto ranked = result.getType().dyn_cast<RankedTensorType>()) {
        shapes.push_back(ShapeAttr::get(op->getContext(), ranked.getShape()));
      } else {
        shapes.push_back(ShapeAttr::get(op->getContext(), llvm::None));
      }
    }
    rewriter.updateRootInPlace(op, [&] {
      op->setAttr(attr_id_, rewriter.getArrayAttr(shapes));
      rewriteImpl(op, rewriter);
    });
    return success();
  }

  virtual void rewriteImpl(Operation *op, PatternRewriter &rewriter) const {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_29(mht_29_v, 814, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "rewriteImpl");
}

 private:
  // Cached identifier for the output shapes attribute.
  StringAttr attr_id_;
};

// Materialize `_output_shapes` for any TFG op.
class MaterializeTFGOpOutputShapes : public MaterializeOutputShapesBase {
 public:
  explicit MaterializeTFGOpOutputShapes(MLIRContext *context)
      : MaterializeOutputShapesBase(context, "_output_shapes"),
        dialect_(context->getOrLoadDialect<TFGraphDialect>()),
        regenerate_output_shapes_id_(
            StringAttr::get(context, kRegenerateOutputShapes)) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_30(mht_30_v, 831, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "MaterializeTFGOpOutputShapes");
}

  bool shouldMatch(Operation *op) const override {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_31(mht_31_v, 836, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "shouldMatch");

    return op->getDialect() == dialect_ &&
           op->getAttrOfType<UnitAttr>(regenerate_output_shapes_id_);
  }

  void rewriteImpl(Operation *op, PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_32(mht_32_v, 844, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "rewriteImpl");

    op->removeAttr(regenerate_output_shapes_id_);
  }

 private:
  // Cached TFG dialect instance.
  TFGraphDialect *dialect_;
  // Identifier to `_regenerate_output_shapes`.
  StringAttr regenerate_output_shapes_id_;
};

// Materialize `output_shapes` for `If`, `Case`, and `While` ops.
struct MaterializeCFOpOutputShapes : public MaterializeOutputShapesBase {
  explicit MaterializeCFOpOutputShapes(MLIRContext *context)
      : MaterializeOutputShapesBase(context, "output_shapes") {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_33(mht_33_v, 861, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "MaterializeCFOpOutputShapes");
}

  bool shouldMatch(Operation *op) const override {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_34(mht_34_v, 866, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "shouldMatch");

    return isa<IfOp, StatelessIfOp, StatefulIfOp, CaseOp, StatelessCaseOp,
               StatefulCaseOp, WhileOp, StatelessWhileOp, StatefulWhileOp>(op);
  }
};
}  // namespace

template <template <typename OpT> class PatternT, typename... OpTs,
          typename... Args>
static void InsertPatterns(RewritePatternSet &patterns, Args &&...args) {
  patterns.insert<PatternT<OpTs>...>(std::forward<Args>(args)...);
}

void PrepareAttributesForExportPassImpl::runOnOperation() {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_35(mht_35_v, 882, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "PrepareAttributesForExportPassImpl::runOnOperation");

  RewritePatternSet patterns(&getContext());
  ControlType control_type = ControlType::get(&getContext());
  InsertPatterns<MaterializeIfAttrs, IfOp, StatelessIfOp, StatefulIfOp>(
      patterns, control_type);
  InsertPatterns<MaterializeCaseAttrs, CaseOp, StatelessCaseOp, StatefulCaseOp>(
      patterns, control_type);
  InsertPatterns<MaterializeTAttr, WhileOp, StatelessWhileOp, StatefulWhileOp,
                 ForOp>(patterns, control_type);
  patterns.insert<MaterializeTFGOpOutputShapes, MaterializeCFOpOutputShapes>(
      &getContext());
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    getOperation()->emitError(getArgument() + " pass failed");
    signalPassFailure();
    return;
  }

  // If the pass was run on a function, materialize attributes with type info.
  auto func = dyn_cast<GraphFuncOp>(getOperation());
  if (!func || func.generic()) return;
  prepareFunctionAttributes(func);
}

namespace {
struct ConsolidateAttributesPass
    : public ConsolidateAttributesBase<ConsolidateAttributesPass> {
  void runOnOperation() override {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_36(mht_36_v, 912, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "runOnOperation");

    // Run the sub-pass on both `tfg.graph` and `tfg.func`.
    PassManager mgr(&getContext());
    mgr.addNestedPass<GraphOp>(
        std::make_unique<ConsolidateAttributesPassImpl>());
    mgr.addNestedPass<GraphFuncOp>(
        std::make_unique<ConsolidateAttributesPassImpl>());
    if (failed(runPipeline(mgr, getOperation()))) signalPassFailure();
  }
};

struct PrepareAttributesForExportPass
    : public PrepareAttributesForExportBase<PrepareAttributesForExportPass> {
  void runOnOperation() override {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScorePStransformsPSconsolidate_attrsPSpassDTcc mht_37(mht_37_v, 928, "", "./tensorflow/core/transforms/consolidate_attrs/pass.cc", "runOnOperation");

    // Run the sub-pass on both `tfg.graph` and `tfg.func`.
    PassManager mgr(&getContext());
    mgr.addNestedPass<GraphOp>(
        std::make_unique<PrepareAttributesForExportPassImpl>());
    mgr.addNestedPass<GraphFuncOp>(
        std::make_unique<PrepareAttributesForExportPassImpl>());
    if (failed(runPipeline(mgr, getOperation()))) signalPassFailure();
  }
};
}  // namespace

std::unique_ptr<Pass> CreateConsolidateAttributesPass() {
  return std::make_unique<ConsolidateAttributesPass>();
}

std::unique_ptr<Pass> CreatePrepareAttributesForExportPass() {
  return std::make_unique<PrepareAttributesForExportPass>();
}

}  // namespace tfg
}  // namespace mlir
