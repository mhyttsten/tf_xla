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
class MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc() {
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

/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tfr/ir/tfr_ops.h"

#include <algorithm>
#include <iterator>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/Dialect/Shape/IR/Shape.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/DialectImplementation.h"  // from @llvm-project
#include "mlir/IR/FunctionImplementation.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/OpDefinition.h"  // from @llvm-project
#include "mlir/IR/OpImplementation.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/InliningUtils.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_attributes.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tfr/ir/tfr_types.h"

namespace mlir {

namespace TFR {

//===----------------------------------------------------------------------===//
// InlinerInterface
//===----------------------------------------------------------------------===//

namespace {
/// This class defines the interface for inlining within the TFR dialect.
class TFRInlinerInterface : public DialectInlinerInterface {
  using DialectInlinerInterface::DialectInlinerInterface;

 public:
  // Allow all call operations to be inlined.
  bool isLegalToInline(Operation *call, Operation *callable,
                       bool wouldBeCloned) const final {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_0(mht_0_v, 240, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "isLegalToInline");

    return true;
  }
  // Returns true if the given region 'src' can be inlined into the region
  // 'dest' that is attached to an operation registered to the current dialect.
  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       BlockAndValueMapping &) const final {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_1(mht_1_v, 249, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "isLegalToInline");

    return true;
  }

  // Returns true if the given operation 'op', that is registered to this
  // dialect, can be inlined into the region 'dest' that is attached to an
  // operation registered to the current dialect.
  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       BlockAndValueMapping &) const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_2(mht_2_v, 260, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "isLegalToInline");

    return true;
  }

  // Handle the given inlined terminator by replacing it with a new operation
  // as necessary. Required when the region has only one block.
  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToRepl) const final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_3(mht_3_v, 270, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "handleTerminator");

    auto retValOp = dyn_cast<TFRReturnOp>(op);
    if (!retValOp) return;

    for (auto ret_value : llvm::zip(valuesToRepl, retValOp.operands())) {
      std::get<0>(ret_value).replaceAllUsesWith(std::get<1>(ret_value));
    }
  }

  // Attempts to materialize a conversion for a type mismatch between a call
  // from this dialect, and a callable region. This method should generate an
  // operation that takes 'input' as the only operand, and produces a single
  // result of 'resultType'. If a conversion can not be generated, nullptr
  // should be returned.
  Operation *materializeCallConversion(OpBuilder &builder, Value input,
                                       Type result_type,
                                       Location conversion_loc) const final {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_4(mht_4_v, 289, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "materializeCallConversion");

    if (!input.getType().isa<IntegerType>() ||
        !result_type.isa<IntegerType>()) {
      return nullptr;
    }
    auto input_itype = input.getType().cast<IntegerType>();
    auto result_itype = result_type.cast<IntegerType>();
    if (input_itype.getWidth() == result_itype.getWidth()) return nullptr;
    if (input_itype.getWidth() > result_itype.getWidth()) {
      return builder.create<arith::TruncIOp>(conversion_loc, result_type,
                                             input);
    } else {
      return builder.create<arith::ExtSIOp>(conversion_loc, result_type, input);
    }
  }
};
}  // namespace

//===----------------------------------------------------------------------===//
// TFR Dialect
//===----------------------------------------------------------------------===//

TFRDialect::TFRDialect(MLIRContext *context)
    : Dialect(/*name=*/"tfr", context, TypeID::get<TFRDialect>()) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_5(mht_5_v, 315, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "TFRDialect::TFRDialect");

  // TFR depends on TensorFlow for its canonicalization
  context->getOrLoadDialect<TF::TensorFlowDialect>();

  addTypes<TFRTensorType, TFRTensorListType, TFRAttrType>();
  addOperations<
#define GET_OP_LIST
#include "tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc.inc"
      >();

  addInterfaces<TFRInlinerInterface>();
}

Operation *TFRDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_6(mht_6_v, 332, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "TFRDialect::materializeConstant");

  if (arith::ConstantOp::isBuildableWith(value, type))
    return builder.create<arith::ConstantOp>(loc, type, value);
  if (func::ConstantOp::isBuildableWith(value, type))
    return builder.create<func::ConstantOp>(loc, type,
                                            value.cast<FlatSymbolRefAttr>());
  return nullptr;
}

bool TFRType::classof(Type type) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_7(mht_7_v, 344, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "TFRType::classof");

  return llvm::isa<TFRDialect>(type.getDialect());
}

//===----------------------------------------------------------------------===//
// Custom op methods
//===----------------------------------------------------------------------===//

LogicalResult ConstantTensorOp::verify() {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_8(mht_8_v, 355, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "ConstantTensorOp::verify");

  ConstantTensorOp op = *this;
  auto input_type = op.arg().getType();
  auto output_type = op.out().getType();

  if (auto output_tensor_type = output_type.dyn_cast<TFRTensorType>()) {
    return success();
  }

  auto output_tensor_type = output_type.dyn_cast<RankedTensorType>();
  if (!output_tensor_type || !output_tensor_type.hasStaticShape()) {
    op.emitError("output type should be static and ranked.");
    return failure();
  }

  if (output_tensor_type.getRank() == 0) {
    bool same_scalar = output_tensor_type.getElementType() == input_type;
    if (!same_scalar) {
      op.emitError("input and output should have the same scalar types.");
    }
    return success(same_scalar);
  }

  if (auto input_vector_type = input_type.dyn_cast<VectorType>()) {
    bool same_element_type = output_tensor_type.getElementType() ==
                             input_vector_type.getElementType();
    bool same_shape =
        output_tensor_type.getShape() == input_vector_type.getShape();
    if (!same_element_type || !same_shape) {
      op.emitError("input and output should have same shape and element type.");
    }
    return success(same_element_type && same_shape);
  }

  op.emitError("input can not be converted to an output tensor.");
  return failure();
}

LogicalResult TFRFuncOp::verify() {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_9(mht_9_v, 396, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "TFRFuncOp::verify");

  TFRFuncOp func = *this;
  // Collect all attribute names used by the tensor and tensor list arguments
  // and returns. Also, collect the names of all the attribute arguments as the
  // defined list. Later on, the used attribute names will be verified to be in
  // the defined list.
  llvm::SmallVector<StringAttr, 4> input_used_attrs, output_used_attrs;

  // While scanning the arguments, record the start/end indices of each argument
  // type, so the order can be verified as well.
  // TODO(fengliuai): the attribute arguments with default values need to be
  // at the end?
  int first_tensor = -1, last_tensor = -1, first_tensor_list = -1,
      last_tensor_list = -1, first_attr = -1;
  for (auto arg : llvm::enumerate(func.getFunctionType().getInputs())) {
    Type arg_type = arg.value();

    if (auto tensor = arg_type.dyn_cast<TFRTensorType>()) {
      if (first_tensor == -1) {
        first_tensor = arg.index();
      }
      last_tensor = arg.index();
      auto used = tensor.getAttrKeys();
      input_used_attrs.append(used.begin(), used.end());
      continue;
    }

    if (auto tensor_list = arg_type.dyn_cast<TFRTensorListType>()) {
      if (first_tensor_list == -1) {
        first_tensor_list = arg.index();
      }
      last_tensor_list = arg.index();
      auto used = tensor_list.getAttrKeys();
      input_used_attrs.append(used.begin(), used.end());
      continue;
    }

    if (!arg_type.isa<TensorType>()) {
      if (first_attr == -1) {
        first_attr = arg.index();
      }
      auto name =
          func.getArgAttrOfType<StringAttr>(arg.index(), kAttrArgumentNameAttr);
      if (!name) {
        func.emitError(
            llvm::Twine(arg.index()) +
            " attribute argument doesn't have a tfr.name attribute.");
        return failure();
      }
      continue;
    }

    func.emitError("Builtin TensorType isn't allowed as the argument.");
    return failure();
  }

  // Collect all the undefined attributes used in the inputs.
  llvm::SmallVector<StringAttr, 4> undefined_attrs;
  for (auto attr : input_used_attrs) {
    if (!func->getAttr(attr.getValue())) {
      undefined_attrs.push_back(attr);
    }
  }

  // Verify the argument order: tensors, tensor list, attributes; and also
  // verify there is at most one tensor list argument.
  if (first_attr != -1 &&
      (first_attr < last_tensor_list || first_attr < last_tensor)) {
    func.emitError(
        "tfr.tensor/tfr.tensor_list argument should be before non tensor "
        "arguments.");
    return failure();
  }
  // The order between tensor arguments and tensor list arguments and the number
  // of tensor list arguments are verified only when they couldn't be determined
  // by the attributes.
  if (!undefined_attrs.empty()) {
    if (first_tensor_list != -1 && first_tensor_list < last_tensor) {
      func.emitError(
          "tfr.tensor argument should be before tfr.tensor_list argument.");
      return failure();
    }
    if (first_tensor_list != last_tensor_list) {
      func.emitError("More than one tfr.tensor_list argument isn't allowed.");
      return failure();
    }
  }

  // Verify the result order: tensor, tensor list, and also verify at most one
  // tensor list result.
  int undefined_input_attrs_number = undefined_attrs.size();
  bool seen_tensor_list = false, has_tensor_list_order_error = false,
       has_multiple_tensor_lists_error = false;
  for (auto result_type : func.getFunctionType().getResults()) {
    if (auto tensor = result_type.dyn_cast<TFRTensorType>()) {
      if (seen_tensor_list) {
        has_tensor_list_order_error = true;
      } else {
        auto used = tensor.getAttrKeys();
        output_used_attrs.append(used.begin(), used.end());
      }
      continue;
    }

    if (auto tensor_list = result_type.dyn_cast<TFRTensorListType>()) {
      if (seen_tensor_list) {
        has_multiple_tensor_lists_error = true;
      } else {
        seen_tensor_list = true;
        auto used = tensor_list.getAttrKeys();
        output_used_attrs.append(used.begin(), used.end());
      }
      continue;
    }

    func.emitError(
        "None tfr.tensor/tfr.tensor_list results aren't allowed as a "
        "result.");
    return failure();
  }

  // Collect all the undefined attributes used in the outputs.
  for (auto attr : output_used_attrs) {
    if (!func->getAttr(attr.getValue())) {
      undefined_attrs.push_back(attr);
    }
  }

  // Verify there are no tensor/tensor list order error and multiple tensor
  // list arguments error.
  if (undefined_input_attrs_number != undefined_attrs.size()) {
    if (has_tensor_list_order_error) {
      func.emitError(
          "tfr.tensor result should be before tfr.tensor_list result.");
      return failure();
    } else if (has_multiple_tensor_lists_error) {
      func.emitError("More than one tfr.tensor_list result isn't allowed.");
      return failure();
    }
  }

  // TODO(fengliuai): We might want to refine this constraint because the
  // tensor element type can be derived.
  if (!undefined_attrs.empty()) {
    llvm::SmallVector<std::string, 4> attr_names(undefined_attrs.size());
    std::transform(undefined_attrs.begin(), undefined_attrs.end(),
                   attr_names.begin(),
                   [](StringAttr attr) { return attr.getValue().str(); });
    func.emitError(llvm::Twine("Undefined attributes are used: ",
                               llvm::join(attr_names, ",")));
    return failure();
  }

  return success();
}

ParseResult TFRFuncOp::parse(OpAsmParser &parser, OperationState &result) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_10(mht_10_v, 555, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "TFRFuncOp::parse");

  auto build_func_type =
      [](Builder &builder, ArrayRef<Type> arg_types, ArrayRef<Type> results,
         function_interface_impl::VariadicFlag,
         std::string &) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_11(mht_11_v, 562, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "lambda");
 return builder.getFunctionType(arg_types, results); };
  return function_interface_impl::parseFunctionOp(
      parser, result, /*allowVariadic=*/false, build_func_type);
}

void TFRFuncOp::print(OpAsmPrinter &p) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_12(mht_12_v, 570, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "TFRFuncOp::print");

  function_interface_impl::printFunctionOp(p, *this, /*isVariadic=*/false);
}

}  // namespace TFR
}  // namespace mlir

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc.inc"

namespace mlir {
namespace TFR {
namespace {
class ConvertConstToTensorConst : public OpRewritePattern<ConstantTensorOp> {
  using OpRewritePattern<ConstantTensorOp>::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(ConstantTensorOp cst_tensor_op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_13(mht_13_v, 595, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "matchAndRewrite");

    Location loc = cst_tensor_op.getLoc();
    Type out_type = cst_tensor_op.getType();
    Operation *new_cst = nullptr;

    ArrayAttr array;
    if (matchPattern(cst_tensor_op.arg(), m_Constant(&array))) {
      llvm::DenseSet<Type> all_types;
      for (auto it : array) {
        all_types.insert(it.getType());
      }
      if (all_types.size() != 1) return failure();
      ShapedType new_out_type = RankedTensorType::get(
          {static_cast<int64_t>(array.size())}, *all_types.begin());
      DenseElementsAttr attr =
          DenseElementsAttr::get(new_out_type, array.getValue());
      new_cst = rewriter.create<TF::ConstOp>(loc, new_out_type, attr);
      if (out_type.isa<TFRTensorType>()) {
        new_cst = rewriter.create<CastOp>(loc, out_type, new_cst->getResult(0));
      }
      rewriter.replaceOp(cst_tensor_op, new_cst->getResult(0));
      return success();
    }

    Attribute scalar;
    if (matchPattern(cst_tensor_op.arg(), m_Constant(&scalar))) {
      Type new_out_type = RankedTensorType::get({}, scalar.getType());
      new_cst = rewriter.create<TF::ConstOp>(loc, new_out_type, scalar);
      if (out_type.isa<TFRTensorType>()) {
        new_cst = rewriter.create<CastOp>(loc, out_type, new_cst->getResult(0));
      }
      rewriter.replaceOp(cst_tensor_op, new_cst->getResult(0));
      return success();
    }
    return failure();
  }
};

inline bool isQuantizedType(Type type) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_14(mht_14_v, 636, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "isQuantizedType");

  auto tensor_type = type.dyn_cast<TensorType>();
  return (tensor_type &&
          tensor_type.getElementType().isa<quant::QuantizedType>());
}

class RemoveRedundantCast : public OpRewritePattern<CastOp> {
  using OpRewritePattern<CastOp>::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(CastOp cast_op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_15(mht_15_v, 650, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "matchAndRewrite");

    auto preceding_cast =
        llvm::dyn_cast_or_null<CastOp>(cast_op.arg().getDefiningOp());
    if (!preceding_cast) {
      return failure();
    }
    Value input = preceding_cast.arg();
    Type input_type = input.getType();
    Type output_type = cast_op.getType();

    // Preserve quantization information for intermediate tensors.
    auto intermediate_type = preceding_cast.getType();
    if (isQuantizedType(intermediate_type) || isQuantizedType(output_type)) {
      return failure();
    }

    auto input_tensor_type = input_type.dyn_cast<TensorType>();
    auto output_tensor_type = output_type.dyn_cast<TensorType>();
    if (!input_tensor_type || !output_tensor_type) {
      return failure();
    }

    // Canonicalize two tfr.cast pairs with different element type to
    // two tfr.casts with the same element type followed by a tf.Cast.
    if ((input_tensor_type.getElementType() !=
         output_tensor_type.getElementType()) &&
        !isQuantizedType(input_type) && !isQuantizedType(output_type)) {
      auto new_tfr_cast = rewriter.create<TFR::CastOp>(
          cast_op.getLoc(),
          output_tensor_type.clone(input_tensor_type.getElementType()),
          cast_op.arg());
      rewriter.replaceOpWithNewOp<TF::CastOp>(cast_op, output_type,
                                              new_tfr_cast);
      return success();
    }

    // If the two types are the same, the back-to-back tfr.cast ops can be
    // removed.
    if (input_type == output_type || output_type.isa<UnrankedTensorType>()) {
      rewriter.replaceOp(cast_op, {input});
      return success();
    }

    // If the rank of the input tensor isn't ranked, we replace the pair
    // with tf.EnsureShape op so it can be removed after shape inference or
    // confirmed at runtime.
    if (input_type.isa<UnrankedTensorType>()) {
      auto shape = output_type.cast<ShapedType>().getShape();
      auto shape_attr = TF::ShapeAttr::get(rewriter.getContext(), shape);
      rewriter.replaceOpWithNewOp<TF::EnsureShapeOp>(cast_op, output_type,
                                                     input, shape_attr);
      return success();
    }

    return failure();
  }
};

class GetTensorShape : public OpRewritePattern<GetShapeOp> {
  using OpRewritePattern<GetShapeOp>::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(GetShapeOp shape_op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_16(mht_16_v, 716, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "matchAndRewrite");

    Operation *preceding_op = shape_op.arg().getDefiningOp();
    if (auto cast_op = llvm::dyn_cast_or_null<CastOp>(preceding_op)) {
      // replace this pair by shape.shape_of, so the folding works.
      rewriter.replaceOpWithNewOp<shape::ShapeOfOp>(shape_op, cast_op.arg());
      return success();
    }
    return failure();
  }
};

class RemoveRedundantGetElement : public OpRewritePattern<GetElementOp> {
  using OpRewritePattern<GetElementOp>::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(GetElementOp ge_op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_17(mht_17_v, 735, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "matchAndRewrite");

    IntegerAttr index;
    if (!matchPattern(ge_op.index(), m_Constant(&index))) {
      return failure();
    }
    auto preceding_build_list = llvm::dyn_cast_or_null<BuildListOp>(
        ge_op.tensor_list().getDefiningOp());
    if (!preceding_build_list ||
        preceding_build_list.getNumOperands() <= index.getInt()) {
      return failure();
    }
    Value input = preceding_build_list.getOperand(index.getInt());
    Type output_type = ge_op.getType();
    if (input.getType() != output_type &&
        !output_type.isa<UnrankedTensorType>()) {
      return failure();
    }
    rewriter.replaceOp(ge_op, {input});
    return success();
  }
};

class RemoveRedundantGetLength : public OpRewritePattern<GetLengthOp> {
  using OpRewritePattern<GetLengthOp>::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(GetLengthOp gl_op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_18(mht_18_v, 765, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "matchAndRewrite");

    auto preceding_build_list = llvm::dyn_cast_or_null<BuildListOp>(
        gl_op.tensor_list().getDefiningOp());
    if (!preceding_build_list) {
      return failure();
    }
    int64_t num_tensors = preceding_build_list.getNumOperands();
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(
        gl_op, rewriter.getIndexAttr(num_tensors));
    return success();
  }
};

class BuildConstantListAsAttr : public OpRewritePattern<BuildListOp> {
  using OpRewritePattern<BuildListOp>::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(BuildListOp bl_op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_19(mht_19_v, 786, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "matchAndRewrite");

    SmallVector<Attribute, 4> array_list;
    array_list.reserve(bl_op.getNumOperands());
    for (const auto &operand : bl_op.getOperands()) {
      Attribute array_elt;
      if (!matchPattern(operand, m_Constant(&array_elt))) {
        return failure();
      }
      array_list.push_back(array_elt);
    }
    auto array_attr = rewriter.getArrayAttr(array_list);
    rewriter.replaceOpWithNewOp<TFR::ConstOp>(bl_op, array_attr);
    return success();
  }
};

quant::QuantizedType getQuantizedElementType(CastOp cast_op) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_20(mht_20_v, 805, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "getQuantizedElementType");

  if (!cast_op || !cast_op.getInputElementType()) {
    return {};
  }
  return cast_op.getInputElementType()
      .cast<TypeAttr>()
      .getValue()
      .dyn_cast<quant::QuantizedType>();
}

class RemoveRawDataOp : public OpRewritePattern<TFRQuantRawDataOp> {
  using OpRewritePattern<TFRQuantRawDataOp>::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(TFRQuantRawDataOp raw_data_op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_21(mht_21_v, 823, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "matchAndRewrite");

    auto preceding_op = raw_data_op.input().getDefiningOp();
    if (isa<BuildListOp>(preceding_op)) {
      return rewritePrecedingListOp(raw_data_op, rewriter);
    }

    auto preceding_cast = dyn_cast_or_null<CastOp>(preceding_op);
    if (!preceding_cast || !getQuantizedElementType(preceding_cast)) {
      return failure();
    }
    // If there are redundant casts, hoist output of raw data op originating op.
    if (preceding_cast.arg().getDefiningOp()) {
      auto redundant_cast = preceding_cast.arg().getDefiningOp<CastOp>();
      if (!redundant_cast ||
          redundant_cast.arg().getType() != preceding_cast.out().getType()) {
        return failure();
      }
      raw_data_op.output().replaceAllUsesWith(redundant_cast.arg());
    } else {
      // If the argument of cast op is input, then simply remove the RawData op.
      raw_data_op.output().replaceAllUsesWith(preceding_cast.out());
    }
    return success();
  }

  LogicalResult rewritePrecedingListOp(TFRQuantRawDataOp raw_data_op,
                                       PatternRewriter &rewriter) const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_22(mht_22_v, 852, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "rewritePrecedingListOp");

    llvm::SmallVector<Value> new_list_values;
    auto preceding_list = raw_data_op.input().getDefiningOp<BuildListOp>();
    for (Value operand : preceding_list.tensors()) {
      auto preceding_cast = operand.getDefiningOp<CastOp>();
      if (!preceding_cast || !getQuantizedElementType(preceding_cast)) {
        return failure();
      }

      // This function currently only supports the case with redundant casts.
      auto redundant_cast = preceding_cast.arg().getDefiningOp<CastOp>();
      if (!redundant_cast ||
          redundant_cast.arg().getType() != preceding_cast.out().getType()) {
        return failure();
      }

      new_list_values.push_back(redundant_cast.arg());
    }

    auto new_list = rewriter.create<BuildListOp>(
        raw_data_op.getLoc(), preceding_list.getType(), new_list_values);
    raw_data_op.output().replaceAllUsesWith(new_list.out());
    return success();
  }
};

class RemoveQParamsOp : public OpRewritePattern<TFRQuantQParamsOp> {
  using OpRewritePattern<TFRQuantQParamsOp>::OpRewritePattern;

 public:
  LogicalResult matchAndRewrite(TFRQuantQParamsOp qparams_op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_23(mht_23_v, 886, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "matchAndRewrite");

    auto cast_op = dyn_cast<TFR::CastOp>(qparams_op.input().getDefiningOp());
    auto cast_qtype = getQuantizedElementType(cast_op);
    if (!cast_qtype) {
      return failure();
    }

    TF::ConstOp scale_op;
    TF::ConstOp zp_op;

    // Reads quantization parameters from the quantized type, and converts
    // them to constants.
    rewriter.setInsertionPoint(qparams_op);
    Location loc = qparams_op->getLoc();
    if (auto qtype = cast_qtype.dyn_cast<quant::UniformQuantizedType>()) {
      scale_op = rewriter.create<TF::ConstOp>(
          loc, RankedTensorType::get({}, rewriter.getF32Type()),
          rewriter.getF32FloatAttr(qtype.getScale()));
      zp_op = rewriter.create<TF::ConstOp>(
          loc, RankedTensorType::get({}, rewriter.getI32Type()),
          rewriter.getI32IntegerAttr(qtype.getZeroPoint()));
    } else if (auto qtype =
                   cast_qtype.dyn_cast<quant::UniformQuantizedPerAxisType>()) {
      SmallVector<float> scales(qtype.getScales().begin(),
                                qtype.getScales().end());
      SmallVector<int32_t> zps(qtype.getZeroPoints().begin(),
                               qtype.getZeroPoints().end());
      const size_t num_channels = qtype.getScales().size();

      auto scales_type = RankedTensorType::get(
          {static_cast<int64_t>(num_channels)}, rewriter.getF32Type());
      auto scales_attr =
          DenseElementsAttr::get(scales_type, llvm::makeArrayRef(scales));
      scale_op = rewriter.create<TF::ConstOp>(loc, scales_attr);

      auto zps_type = RankedTensorType::get(
          {static_cast<int64_t>(num_channels)}, rewriter.getI32Type());
      auto zps_attr = DenseElementsAttr::get(zps_type, llvm::makeArrayRef(zps));
      zp_op = rewriter.create<TF::ConstOp>(loc, zps_attr);
    }
    if (!scale_op || !zp_op) {
      return failure();
    }
    auto scale_cast = rewriter.create<CastOp>(loc, qparams_op.scale().getType(),
                                              scale_op.output());
    auto zp_cast =
        rewriter.create<CastOp>(loc, qparams_op.zp().getType(), zp_op.output());

    qparams_op.scale().replaceAllUsesWith(scale_cast.out());
    qparams_op.zp().replaceAllUsesWith(zp_cast.out());
    return success();
  }
};

// TODO(b/193731721): Migrate tfr_ builtin canonicalizations to LowerTFROpPass
class RemoveScaleFactorOp : public OpRewritePattern<TFRQuantScaleFactorOp> {
  using OpRewritePattern<TFRQuantScaleFactorOp>::OpRewritePattern;

 public:
  // Replace quant_scale_factor with constant tensor equivalent to
  // TFR_ConstantTensorOp (
  //   ConstantOp (ConstAttr<F32Attr (in_scale[0] * in_scale[1] /
  //   out_scale))
  // )
  // Currently, all decompositions using this pattern (Conv2D, FC) have the
  // following preconditions:
  // * out_scale: float scalar attribute
  // * in_scale[0] (input scale): float scalar, given by tf.Const -> tfr.cast
  // * in_scale[1] (filter scale): float scalar/vector
  //     (per-tensor vs per-channel) quantization, given by tf.Const -> tfr.cast
  LogicalResult matchAndRewrite(TFRQuantScaleFactorOp scale_factor_op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_24(mht_24_v, 960, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "matchAndRewrite");

    auto out_scale_op =
        scale_factor_op.out_scale().getDefiningOp<arith::ConstantOp>();
    if (!out_scale_op) {
      return failure();
    }
    const double out_scale =
        out_scale_op.getValue().cast<FloatAttr>().getValueAsDouble();

    auto in_scales_op =
        scale_factor_op.in_scales().getDefiningOp<BuildListOp>();
    if (!in_scales_op || in_scales_op.getNumOperands() != 2) {
      // BuildListOp is variadic, but we require two values: input_scale
      // and filter_scale.
      return failure();
    }

    auto in_scale_op = in_scales_op.getOperand(0).getDefiningOp<CastOp>();
    if (!in_scale_op) {
      return failure();
    }

    DenseFPElementsAttr in_scale_attr;
    if (!matchPattern(in_scale_op.arg(), m_Constant(&in_scale_attr)) ||
        in_scale_attr.size() != 1) {
      return failure();
    }
    const float in_scale = in_scale_attr.getValues<float>()[0];
    auto filter_scale_op = in_scales_op.getOperand(1).getDefiningOp<CastOp>();
    if (!filter_scale_op) {
      return failure();
    }
    DenseFPElementsAttr filter_scale_attr;
    if (!matchPattern(filter_scale_op.arg(), m_Constant(&filter_scale_attr))) {
      return failure();
    }

    // The shape of scale_type is {} (rank 0) for per-tensor quantized tensor,
    // and {num_channels} (rank 1) for per-channel quantized one.
    auto scale_type = filter_scale_attr.getType().dyn_cast<RankedTensorType>();
    if (scale_type.getRank() != 0 && scale_type.getRank() != 1) {
      return failure();
    }
    SmallVector<float> scale_factors;
    scale_factors.reserve(filter_scale_attr.size());
    for (auto value : filter_scale_attr.getValues<APFloat>()) {
      scale_factors.push_back(in_scale * value.convertToFloat() / out_scale);
    }
    rewriter.setInsertionPoint(scale_factor_op);
    const Location loc = scale_factor_op->getLoc();
    auto result_scale_op = rewriter.create<TF::ConstOp>(
        loc,
        DenseElementsAttr::get(scale_type, llvm::makeArrayRef(scale_factors)));
    auto result_scale_cast_op = rewriter.create<CastOp>(
        loc, scale_factor_op.getType(), result_scale_op.output());
    scale_factor_op.scale_factor().replaceAllUsesWith(
        result_scale_cast_op.out());
    return success();
  }
};

class RemoveRescaleOp : public OpRewritePattern<TFRQuantRescaleOp> {
  using OpRewritePattern<TFRQuantRescaleOp>::OpRewritePattern;

 public:
  // Replace quant_rescale (input, scale, zp) with
  // tf.Cast(tf.Round(tf.Cast(input, f32) * scale) + tf.Cast(zp, f32), i32)
  LogicalResult matchAndRewrite(TFRQuantRescaleOp rescale_op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_25(mht_25_v, 1031, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "matchAndRewrite");

    Value input = rescale_op.input();
    Value scale = rescale_op.scale();
    Value zp = rescale_op.zp();

    const Location loc = rescale_op->getLoc();
    const auto result_types = rescale_op->getResultTypes();
    auto c_false =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(false));
    TypeAttr f32_attr = TypeAttr::get(rewriter.getF32Type());
    TFRAttrType output_type = TFRAttrType::get(rewriter.getContext());
    auto constant_f32_op = rewriter.create<ConstOp>(loc, output_type, f32_attr);
    TypeAttr i32_attr = TypeAttr::get(rewriter.getI32Type());
    auto constant_i32_op = rewriter.create<ConstOp>(loc, output_type, i32_attr);

    IntegerAttr zp_attr;
    if (!matchPattern(zp, m_Constant(&zp_attr))) {
      return failure();
    }
    rewriter.setInsertionPoint(zp.getDefiningOp());
    auto zp_tensor = rewriter.create<TF::ConstOp>(
        loc, RankedTensorType::get({}, zp.getType()), zp_attr);
    auto zp_cast = rewriter.create<CastOp>(
        loc, rewriter.getType<TFRTensorType>(), zp_tensor.output());

    rewriter.setInsertionPoint(rescale_op);
    auto cast_input_to_float_op = rewriter.create<CallOp>(
        loc, result_types,
        SymbolRefAttr::get(rewriter.getContext(), "tf__cast"),
        ArrayRef<Value>{input, constant_f32_op, c_false});
    auto input_x_scale_op = rewriter.create<CallOp>(
        loc, result_types, SymbolRefAttr::get(rewriter.getContext(), "tf__mul"),
        ArrayRef<Value>{cast_input_to_float_op.getResult(0), scale});
    auto round_rescaled_op = rewriter.create<CallOp>(
        loc, result_types,
        SymbolRefAttr::get(rewriter.getContext(), "tf__round"),
        ArrayRef<Value>{input_x_scale_op->getResult(0)});
    auto cast_zp_to_float_op = rewriter.create<CallOp>(
        loc, result_types,
        SymbolRefAttr::get(rewriter.getContext(), "tf__cast"),
        ArrayRef<Value>{zp_cast, constant_f32_op, c_false});
    auto recentered_op = rewriter.create<CallOp>(
        loc, result_types, SymbolRefAttr::get(rewriter.getContext(), "tf__add"),
        ArrayRef<Value>{round_rescaled_op->getResult(0),
                        cast_zp_to_float_op->getResult(0)});
    auto cast_output_to_i32 = rewriter.create<CallOp>(
        loc, result_types,
        SymbolRefAttr::get(rewriter.getContext(), "tf__cast"),
        ArrayRef<Value>{recentered_op->getResult(0), constant_i32_op, c_false});
    rescale_op.output().replaceAllUsesWith(cast_output_to_i32.getResult(0));
    return success();
  }
};

}  // namespace

void ConstantTensorOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                   MLIRContext *context) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_26(mht_26_v, 1091, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "ConstantTensorOp::getCanonicalizationPatterns");

  results.add<ConvertConstToTensorConst>(context);
}

void CastOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                         MLIRContext *context) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_27(mht_27_v, 1099, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "CastOp::getCanonicalizationPatterns");

  results.add<RemoveRedundantCast>(context);
}

void GetShapeOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_28(mht_28_v, 1107, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "GetShapeOp::getCanonicalizationPatterns");

  results.add<GetTensorShape>(context);
}

void GetElementOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_29(mht_29_v, 1115, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "GetElementOp::getCanonicalizationPatterns");

  results.add<RemoveRedundantGetElement>(context);
}

void GetLengthOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_30(mht_30_v, 1123, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "GetLengthOp::getCanonicalizationPatterns");

  results.add<RemoveRedundantGetLength>(context);
}

void BuildListOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_31(mht_31_v, 1131, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "BuildListOp::getCanonicalizationPatterns");

  results.add<BuildConstantListAsAttr>(context);
}

void TFRQuantRawDataOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                    MLIRContext *context) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_32(mht_32_v, 1139, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "TFRQuantRawDataOp::getCanonicalizationPatterns");

  results.add<RemoveRawDataOp>(context);
}

void TFRQuantQParamsOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                    MLIRContext *context) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_33(mht_33_v, 1147, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "TFRQuantQParamsOp::getCanonicalizationPatterns");

  results.add<RemoveQParamsOp>(context);
}

void TFRQuantRescaleOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                    MLIRContext *context) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_34(mht_34_v, 1155, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "TFRQuantRescaleOp::getCanonicalizationPatterns");

  results.add<RemoveRescaleOp>(context);
}

void TFRQuantScaleFactorOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_35(mht_35_v, 1163, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "TFRQuantScaleFactorOp::getCanonicalizationPatterns");

  results.add<RemoveScaleFactorOp>(context);
}

OpFoldResult TFR::EqualOp::fold(ArrayRef<Attribute> operands) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_36(mht_36_v, 1170, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "TFR::EqualOp::fold");

  assert(operands.size() == 2 && "equal op has two operands");
  auto ctx = getContext();
  if (operands[0] == operands[1]) return BoolAttr::get(ctx, true);
  return BoolAttr::get(ctx, false);
}

OpFoldResult ConstOp::fold(ArrayRef<Attribute> operands) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_37(mht_37_v, 1180, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "ConstOp::fold");

  assert(operands.empty() && "constant has no operands");

  // Return the held attribute value.
  return value();
}

// CallableOpInterface
Region *TFRFuncOp::getCallableRegion() {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_38(mht_38_v, 1191, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "TFRFuncOp::getCallableRegion");

  return isExternal() ? nullptr : &body().front();
}

// CallableOpInterface
ArrayRef<Type> TFRFuncOp::getCallableResults() {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_39(mht_39_v, 1199, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "TFRFuncOp::getCallableResults");

  return getFunctionType().getResults();
}

//===----------------------------------------------------------------------===//
// Dialect type definitions
//===----------------------------------------------------------------------===//

// Parses a TFR type.
//   tfr_type ::= tensor_type | tensor_list_type | attr_type
//   string_list ::= `[` string-literal (, string-literal)+ `]`
//   tensor_type ::= `tensor`
//                 | `tensor<` (string-literal | string_list)  '>'
//   tensor_list_type ::= `tensor_list`
//                      | `tensor_list<` (string-literal | string_list)  '>'
//   attr_type ::= `attr`
Type TFRDialect::parseType(DialectAsmParser &parser) const {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_40(mht_40_v, 1218, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "TFRDialect::parseType");

  Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());
  MLIRContext *ctx = loc.getContext();

  StringRef typeNameSpelling;
  if (failed(parser.parseKeyword(&typeNameSpelling))) return {};
  llvm::SmallVector<StringAttr, 4> attrs;
  if (succeeded(parser.parseOptionalLess())) {
    bool l_square_parsed = false;
    if (succeeded(parser.parseOptionalLSquare())) {
      l_square_parsed = true;
    }

    do {
      StringRef attr;
      if (failed(parser.parseKeyword(&attr))) return {};
      attrs.push_back(StringAttr::get(ctx, attr));
    } while (succeeded(parser.parseOptionalComma()));

    if (l_square_parsed && failed(parser.parseRSquare())) {
      parser.emitError(parser.getNameLoc(), "expected ']'");
    }

    if (failed(parser.parseGreater())) {
      parser.emitError(parser.getNameLoc(), "expected '>'");
    }
  }

  if (typeNameSpelling == "tensor") {
    return TFRTensorType::getChecked(attrs, loc);
  } else if (typeNameSpelling == "tensor_list") {
    return TFRTensorListType::getChecked(attrs, loc);
  } else if (typeNameSpelling == "attr") {
    return TFRAttrType::getChecked(loc, loc.getContext());
  } else {
    parser.emitError(parser.getNameLoc(), "unknown type " + typeNameSpelling);
    return {};
  }
}

void TFRDialect::printType(Type type, DialectAsmPrinter &os) const {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPStfrPSirPStfr_opsDTcc mht_41(mht_41_v, 1261, "", "./tensorflow/compiler/mlir/tfr/ir/tfr_ops.cc", "TFRDialect::printType");

  llvm::ArrayRef<StringAttr> attrs;

  if (type.isa<TFRAttrType>()) {
    os << "attr";
    return;
  }
  if (auto tensor_ty = type.dyn_cast<TFRTensorType>()) {
    attrs = tensor_ty.getAttrKeys();
    os << "tensor";
  } else if (auto tensor_list_ty = type.dyn_cast<TFRTensorListType>()) {
    attrs = tensor_list_ty.getAttrKeys();
    os << "tensor_list";
  } else {
    llvm_unreachable("Unhandled tfr type");
  }

  if (attrs.empty()) return;
  os << "<";

  if (attrs.size() > 1) {
    os << "[";
  }

  llvm::interleaveComma(attrs, os,
                        [&](StringAttr attr) { os << attr.getValue(); });

  if (attrs.size() > 1) {
    os << "]";
  }
  os << ">";
}

}  // namespace TFR
}  // namespace mlir
