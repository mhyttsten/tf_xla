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
class MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantize_composite_functionsDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantize_composite_functionsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantize_composite_functionsDTcc() {
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
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantOps.h"  // from @llvm-project
#include "mlir/Dialect/Quant/QuantTypes.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/Verifier.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassManager.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/passes.h"
#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/util.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_dialect.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace quant {
namespace {

constexpr char kQuantizeFuncName[] = "quantize_i8";
constexpr char kDequantizeFuncName[] = "dequantize_i8";
constexpr char kAttrMapAttribute[] = "attr_map";

class QuantizeCompositeFunctionsPass
    : public mlir::PassWrapper<QuantizeCompositeFunctionsPass,
                               OperationPass<ModuleOp>> {
 public:
  explicit QuantizeCompositeFunctionsPass() {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantize_composite_functionsDTcc mht_0(mht_0_v, 226, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize_composite_functions.cc", "QuantizeCompositeFunctionsPass");
}
  explicit QuantizeCompositeFunctionsPass(
      QuantizationMethod quantization_method)
      : quantization_method_(quantization_method) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantize_composite_functionsDTcc mht_1(mht_1_v, 232, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize_composite_functions.cc", "QuantizeCompositeFunctionsPass");
}

  StringRef getArgument() const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantize_composite_functionsDTcc mht_2(mht_2_v, 237, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize_composite_functions.cc", "getArgument");

    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "quant-quantize-composite-functions";
  }

  StringRef getDescription() const final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantize_composite_functionsDTcc mht_3(mht_3_v, 246, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize_composite_functions.cc", "getDescription");

    // This is a brief description of the pass.
    return "Quantize composite functions with QDQ input/outputs.";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantize_composite_functionsDTcc mht_4(mht_4_v, 254, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize_composite_functions.cc", "getDependentDialects");

    registry.insert<TF::TensorFlowDialect, QuantizationDialect>();
  }

 private:
  void runOnOperation() override;

  QuantizationMethod quantization_method_ =
      QuantizationMethod::kQuantizationAwareTraining;
};

LogicalResult CreateUniformQuantizedTypeParams(UniformQuantizedType qtype,
                                               Location loc,
                                               PatternRewriter& rewriter,
                                               Value& scale,
                                               Value& zero_point) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantize_composite_functionsDTcc mht_5(mht_5_v, 272, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize_composite_functions.cc", "CreateUniformQuantizedTypeParams");

  TensorType scale_type = RankedTensorType::get({}, rewriter.getF32Type());
  TensorType zero_point_type = scale_type.clone(rewriter.getI32Type());
  scale = rewriter.create<TF::ConstOp>(
      loc, scale_type,
      DenseFPElementsAttr::get(scale_type,
                               {static_cast<float>(qtype.getScale())}));
  zero_point = rewriter.create<TF::ConstOp>(
      loc, zero_point_type,
      DenseIntElementsAttr::get(zero_point_type,
                                {static_cast<int32_t>(qtype.getZeroPoint())}));
  return success(scale && zero_point);
}

LogicalResult CreateUniformQuantizedPerAxisTypeParams(
    UniformQuantizedPerAxisType qtype, Location loc, PatternRewriter& rewriter,
    Value& scale, Value& zero_point) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantize_composite_functionsDTcc mht_6(mht_6_v, 291, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize_composite_functions.cc", "CreateUniformQuantizedPerAxisTypeParams");

  // Consuming op should already know about Quantized channel information,
  // so not passing it during conversion. This design might change if needed.
  ArrayRef<double> scales = qtype.getScales();
  ArrayRef<int64_t> zero_points = qtype.getZeroPoints();
  const int num_channels = scales.size();
  TensorType scale_type = RankedTensorType::get(
      {static_cast<int64_t>(num_channels)}, rewriter.getF32Type());
  TensorType zero_point_type = scale_type.clone(rewriter.getI32Type());

  llvm::SmallVector<float, 4> float_scales;
  llvm::SmallVector<int32_t, 4> int32_zero_points;
  float_scales.reserve(num_channels);
  int32_zero_points.reserve(num_channels);
  for (int i = 0; i < num_channels; ++i) {
    float_scales.push_back(scales[i]);
    int32_zero_points.push_back(zero_points[i]);
  }
  scale = rewriter.create<TF::ConstOp>(
      loc, scale_type, DenseFPElementsAttr::get(scale_type, float_scales));
  zero_point = rewriter.create<TF::ConstOp>(
      loc, zero_point_type,
      DenseIntElementsAttr::get(zero_point_type, int32_zero_points));
  return success(scale && zero_point);
}

LogicalResult CreateQuantizationParams(QuantizedType elem_type, Location loc,
                                       PatternRewriter& rewriter, Value& scale,
                                       Value& zero_point) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantize_composite_functionsDTcc mht_7(mht_7_v, 322, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize_composite_functions.cc", "CreateQuantizationParams");

  if (!elem_type) {
    return failure();
  }
  if (auto qtype = elem_type.dyn_cast<UniformQuantizedType>()) {
    return CreateUniformQuantizedTypeParams(qtype, loc, rewriter, scale,
                                            zero_point);
  } else if (auto qtype = elem_type.dyn_cast<UniformQuantizedPerAxisType>()) {
    return CreateUniformQuantizedPerAxisTypeParams(qtype, loc, rewriter, scale,
                                                   zero_point);
  }
  return failure();
}

// Replaces quant.qcast op to composite quantize_i8 function.
class ReplaceQuantizePattern : public mlir::OpRewritePattern<QuantizeCastOp> {
 public:
  explicit ReplaceQuantizePattern(MLIRContext* context)
      : OpRewritePattern<QuantizeCastOp>(context) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantize_composite_functionsDTcc mht_8(mht_8_v, 343, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize_composite_functions.cc", "ReplaceQuantizePattern");
}

 private:
  LogicalResult matchAndRewrite(QuantizeCastOp q_op,
                                PatternRewriter& rewriter) const override {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantize_composite_functionsDTcc mht_9(mht_9_v, 350, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize_composite_functions.cc", "matchAndRewrite");

    auto output_type = q_op.getType().cast<TensorType>();
    auto elem_type = output_type.getElementType().dyn_cast<QuantizedType>();
    const Location loc = q_op->getLoc();
    Value scale, zero_point;

    if (failed(CreateQuantizationParams(elem_type, loc, rewriter, scale,
                                        zero_point))) {
      return failure();
    }

    SmallVector<Type> output_types = {
        output_type.clone(elem_type.getStorageType())};
    SmallVector<Value> args = {q_op.arg(), scale, zero_point};
    FlatSymbolRefAttr func_name =
        FlatSymbolRefAttr::get(rewriter.getStringAttr(kQuantizeFuncName));

    auto quantize_call = rewriter.create<TF::PartitionedCallOp>(
        loc, output_types, args, func_name,
        /*config=*/"", /*config_proto=*/"", /*executor_type=*/"");
    auto scast_op = rewriter.create<quant::StorageCastOp>(
        loc, output_type, quantize_call->getResult(0));
    q_op->replaceAllUsesWith(scast_op);
    return success();
  }
};

// Replaces quant.dcast op to composite dequantize_i8 function.
class ReplaceDequantizePattern
    : public mlir::OpRewritePattern<DequantizeCastOp> {
 public:
  explicit ReplaceDequantizePattern(MLIRContext* context)
      : OpRewritePattern<DequantizeCastOp>(context) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantize_composite_functionsDTcc mht_10(mht_10_v, 385, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize_composite_functions.cc", "ReplaceDequantizePattern");
}

 private:
  LogicalResult matchAndRewrite(DequantizeCastOp dq_op,
                                PatternRewriter& rewriter) const override {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantize_composite_functionsDTcc mht_11(mht_11_v, 392, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize_composite_functions.cc", "matchAndRewrite");

    auto input_type = dq_op.arg().getType().cast<TensorType>();
    auto elem_type = input_type.getElementType().dyn_cast<QuantizedType>();
    const Location loc = dq_op->getLoc();

    Value scale, zero_point;
    if (failed(CreateQuantizationParams(elem_type, loc, rewriter, scale,
                                        zero_point))) {
      return failure();
    }

    TensorType output_type = input_type.clone(elem_type.getStorageType());
    auto scast_op =
        rewriter.create<quant::StorageCastOp>(loc, output_type, dq_op.arg());

    FlatSymbolRefAttr func_name =
        FlatSymbolRefAttr::get(rewriter.getStringAttr(kDequantizeFuncName));
    SmallVector<Value> args = {scast_op->getResult(0), scale, zero_point};
    auto dequantize_call = rewriter.create<TF::PartitionedCallOp>(
        loc, dq_op.getResult().getType(), args, func_name,
        /*config=*/"", /*config_proto=*/"", /*executor_type=*/"");
    dq_op->replaceAllUsesWith(dequantize_call);
    return success();
  }
};

// Determines if all float input/outputs are now quantized.
bool IsQuantizedCall(TF::PartitionedCallOp call_op) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantize_composite_functionsDTcc mht_12(mht_12_v, 422, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize_composite_functions.cc", "IsQuantizedCall");

  bool has_quantized_types = false;
  for (Value input : call_op.args()) {
    if (auto type = input.getType().dyn_cast<TensorType>()) {
      if (type.getElementType().isa<FloatType>()) {
        return false;
      }
      if (type.getElementType().isa<QuantizedType>()) {
        has_quantized_types = true;
      }
    }
  }
  for (Value output : call_op.output()) {
    if (auto type = output.getType().dyn_cast<TensorType>()) {
      if (type.getElementType().isa<FloatType>()) {
        return false;
      }
      if (type.getElementType().isa<QuantizedType>()) {
        has_quantized_types = true;
      }
    }
  }
  return has_quantized_types;
}

// Transfers the attributes of the corresponding ops from the float function to
// the quantized function using the attr_map attribute. In the quantized
// function, this map (map1) is in {attr_name_1: attr_identifier} format; and in
// the float function, this map (map2) is in {attr_identifier: attr_name_2}
// format. Where, the attribute identifiers should match between two maps,
// attr_name_1 is the name of the of the attribute needs to be set in the
// quantized function, attr_name_2 is the name of the attribute corresponding to
// the attribute identifier in the float function.
LogicalResult TransferAttributes(FuncOp float_func, FuncOp quantized_func) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantize_composite_functionsDTcc mht_13(mht_13_v, 458, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize_composite_functions.cc", "TransferAttributes");

  // A map to find an attribute from its identifier.
  llvm::StringMap<Attribute> identifier_to_attr;
  for (Operation& inner_op : float_func.getBody().front().getOperations()) {
    if (!inner_op.hasAttr(kAttrMapAttribute)) continue;
    std::string attr_map_str =
        inner_op.getAttrOfType<StringAttr>(kAttrMapAttribute).str();
    for (absl::string_view element_str : absl::StrSplit(attr_map_str, ',')) {
      std::vector<absl::string_view> key_and_value_pair =
          absl::StrSplit(element_str, ':');
      if (key_and_value_pair.size() != 2) {
        float_func.emitError("The attr_map attribute is malformed");
        return failure();
      }
      identifier_to_attr.insert(
          {llvm::StringRef(std::string(key_and_value_pair[0])),
           inner_op.getAttr(
               llvm::StringRef(std::string(key_and_value_pair[1])))});
    }
  }

  // Set the attributes for ops with the attr_map attribute.
  for (Operation& inner_op : quantized_func.getBody().front().getOperations()) {
    if (!inner_op.hasAttr(kAttrMapAttribute)) continue;

    std::string attr_map_str =
        inner_op.getAttrOfType<StringAttr>(kAttrMapAttribute).str();
    for (absl::string_view element_str : absl::StrSplit(attr_map_str, ',')) {
      std::vector<absl::string_view> key_and_value_pair =
          absl::StrSplit(element_str, ':');
      if (key_and_value_pair.size() != 2) {
        float_func.emitError("The attr_map attribute is malformed");
        return failure();
      }
      if (identifier_to_attr.count(
              llvm::StringRef(std::string(key_and_value_pair[1]))) == 0) {
        float_func.emitWarning(absl::StrCat("Using the default value for the '",
                                            key_and_value_pair[0],
                                            "' attribute"));
        continue;
      }
      inner_op.setAttr(llvm::StringRef(std::string(key_and_value_pair[0])),
                       identifier_to_attr[llvm::StringRef(
                           std::string(key_and_value_pair[1]))]);
    }
    inner_op.removeAttr(kAttrMapAttribute);
  }
  return success();
}

// Unwraps quantization parameters of PartitionedCall ops with quantized
// input/outputs that are created from QuantizePass.
class QuantizeFunctionPattern
    : public mlir::OpRewritePattern<TF::PartitionedCallOp> {
 public:
  explicit QuantizeFunctionPattern(MLIRContext* context)
      : OpRewritePattern<TF::PartitionedCallOp>(context) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantize_composite_functionsDTcc mht_14(mht_14_v, 517, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize_composite_functions.cc", "QuantizeFunctionPattern");
}

 private:
  LogicalResult matchAndRewrite(TF::PartitionedCallOp call_op,
                                PatternRewriter& rewriter) const override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantize_composite_functionsDTcc mht_15(mht_15_v, 524, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize_composite_functions.cc", "matchAndRewrite");

    auto f_attr = call_op.fAttr().dyn_cast<FlatSymbolRefAttr>();
    // removeAttr will return nullptr if no attribute was removed.
    if (!call_op->removeAttr(kQuantTraitAttrName) || !f_attr) {
      return failure();
    }
    if (!f_attr.getValue().startswith("fused_") || !IsQuantizedCall(call_op)) {
      return failure();
    }

    llvm::Twine quantized_function_name = llvm::Twine(
        "quantized_", f_attr.getValue().substr(6).rsplit('_').first);

    SmallVector<Value, 4> args;
    SmallVector<Value, 4> qparam_args;
    SmallVector<Type, 4> result_types;

    for (Value arg : call_op.args()) {
      if (auto arg_type = arg.getType().dyn_cast<TensorType>()) {
        QuantizedType qtype =
            arg_type.getElementType().dyn_cast<QuantizedType>();
        if (qtype &&
            !qtype.isa<UniformQuantizedType, UniformQuantizedPerAxisType>()) {
          return failure();
        }
      }
    }

    for (Value result : call_op->getResults()) {
      if (auto result_type = result.getType().dyn_cast<TensorType>()) {
        QuantizedType qtype =
            result_type.getElementType().dyn_cast<QuantizedType>();
        if (qtype &&
            !qtype.isa<UniformQuantizedType, UniformQuantizedPerAxisType>()) {
          return failure();
        }
      }
    }

    rewriter.setInsertionPoint(call_op);
    for (Value arg : call_op.args()) {
      TensorType arg_type = arg.getType().dyn_cast<TensorType>();
      if (!arg_type) {
        args.push_back(arg);
        continue;
      }
      QuantizedType qtype = arg_type.getElementType().dyn_cast<QuantizedType>();
      if (!qtype) {
        args.push_back(arg);
        continue;
      }
      Value scale, zero_point;
      if (failed(CreateQuantizationParams(qtype, arg.getLoc(), rewriter, scale,
                                          zero_point))) {
        // As the quantized types are already checked, this is unexpected.
        call_op->emitError(
            "Failed to create quantization parameter for an argument.");
        return failure();
      }
      auto scast_op = rewriter.create<StorageCastOp>(
          arg.getLoc(), arg_type.clone(qtype.getStorageType()), arg);
      args.push_back(scast_op.getResult());
      qparam_args.push_back(scale);
      qparam_args.push_back(zero_point);
    }

    DenseMap<Value, StorageCastOp> replace_map;
    rewriter.setInsertionPointAfter(call_op);
    for (Value result : call_op->getResults()) {
      TensorType result_type = result.getType().dyn_cast<TensorType>();
      if (!result_type) {
        result_types.push_back(result.getType());
        continue;
      }
      QuantizedType qtype =
          result_type.getElementType().dyn_cast<QuantizedType>();
      if (!qtype) {
        result_types.push_back(result_type);
        continue;
      }
      Value scale, zero_point;
      if (failed(CreateQuantizationParams(qtype, result.getLoc(), rewriter,
                                          scale, zero_point))) {
        // As the quantized types are already checked, this is unexpected.
        call_op->emitError(
            "Failed to create quantization parameter for a result.");
        return failure();
      }
      auto scast_op =
          rewriter.create<StorageCastOp>(call_op.getLoc(), result_type, result);
      replace_map.insert(std::make_pair(result, scast_op));

      result_types.push_back(result_type.clone(qtype.getStorageType()));
      qparam_args.push_back(scale);
      qparam_args.push_back(zero_point);
    }

    for (auto replace_pair : replace_map) {
      Value result = replace_pair.first;
      StorageCastOp scast_op = replace_pair.second;
      result.replaceAllUsesExcept(scast_op, scast_op);
    }

    args.insert(args.end(), qparam_args.begin(), qparam_args.end());

    // Make a copy of the quantized function.
    auto module = call_op->getParentOfType<ModuleOp>();
    SymbolTable symbol_table(module);
    FuncOp float_func =
        dyn_cast<FuncOp>(symbol_table.lookup(f_attr.getValue()));
    FuncOp quantized_func =
        dyn_cast<FuncOp>(symbol_table.lookup(quantized_function_name.str()));
    rewriter.setInsertionPointAfter(float_func);
    FuncOp new_quantized_func = dyn_cast<FuncOp>(quantized_func->clone());
    if (new_quantized_func == nullptr) {
      return failure();
    }
    StringAttr new_quant_func_name = symbol_table.insert(new_quantized_func);

    // Set the attributes for ops with the attr_map attribute.
    if (failed(TransferAttributes(float_func, new_quantized_func))) {
      return failure();
    }

    rewriter.setInsertionPoint(call_op);
    rewriter.replaceOpWithNewOp<TF::PartitionedCallOp>(
        call_op, result_types, args,
        FlatSymbolRefAttr::get(new_quant_func_name));

    return success();
  }
};

// Converts const -> quant.qcast pattern to quantized constant, after
// quantization parameters are safely included to each quantize composite
// functions.
class QuantizeConstPattern : public OpRewritePattern<QuantizeCastOp> {
 public:
  // This pattern should have larger benefit than ReplaceQuantizePattern
  explicit QuantizeConstPattern(MLIRContext* context)
      : OpRewritePattern<QuantizeCastOp>(context, /*benefit=*/10) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantize_composite_functionsDTcc mht_16(mht_16_v, 667, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize_composite_functions.cc", "QuantizeConstPattern");
}
  LogicalResult matchAndRewrite(QuantizeCastOp q_op,
                                PatternRewriter& rewriter) const override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSquantizationPStensorflowPSpassesPSquantize_composite_functionsDTcc mht_17(mht_17_v, 672, "", "./tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize_composite_functions.cc", "matchAndRewrite");

    DenseFPElementsAttr attr;
    if (!matchPattern(q_op.arg(), m_Constant(&attr))) {
      return failure();
    }

    ShapedType tensor_qtype = q_op.getResult().getType().cast<ShapedType>();
    Attribute quantized_attr;
    quantized_attr = Quantize(attr, tensor_qtype);
    if (!quantized_attr) {
      return failure();
    }

    Type storage_type =
        tensor_qtype.getElementType().cast<QuantizedType>().getStorageType();
    ShapedType new_type = tensor_qtype.clone(storage_type);
    Location loc = q_op.arg().getLoc();
    auto const_op = rewriter.create<TF::ConstOp>(loc, new_type, quantized_attr);
    // Add scast op to match quantize -> composition pattern. The added scast
    // is then removed by canonicalization. ([scast - scast] -> [])
    auto scast_op = rewriter.create<quant::StorageCastOp>(loc, tensor_qtype,
                                                          const_op.output());
    q_op->replaceAllUsesWith(scast_op);
    return success();
  }
};

static PassRegistration<QuantizeCompositeFunctionsPass> pass;

#include "tensorflow/compiler/mlir/quantization/tensorflow/passes/quantize_composite_functions.inc"

void QuantizeCompositeFunctionsPass::runOnOperation() {
  MLIRContext* ctx = &getContext();
  ModuleOp module = getOperation();

  PassManager pm(ctx);
  // Intermediate output from QuantizePass will have PartitionedCall ops with
  // quantized input and output types, which are not allowed in TF dialect.
  // This can be removed when the composite call supports quantized types.
  pm.enableVerifier(false);

  pm.addNestedPass<FuncOp>(CreatePrepareQuantizePass(quantization_method_));
  pm.addNestedPass<FuncOp>(CreateQuantizePass());
  pm.addNestedPass<FuncOp>(CreatePostQuantizePass());
  if (failed(pm.run(module))) {
    signalPassFailure();
  }

  RewritePatternSet patterns(ctx);
  patterns.add<QuantizeFunctionPattern>(ctx);
  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
    signalPassFailure();
  }

  // Constant quantization is a lossy transformation, so they are applied only
  // after all the other patterns have been aplied.
  RewritePatternSet patterns_2(ctx);
  populateWithGenerated(patterns_2);
  patterns_2.add<ReplaceQuantizePattern, ReplaceDequantizePattern,
                 QuantizeConstPattern>(ctx);
  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns_2))) ||
      failed(verify(module))) {
    signalPassFailure();
  }
}

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>> CreateQuantizeCompositeFunctionsPass(
    QuantizationMethod quantization_method) {
  return std::make_unique<QuantizeCompositeFunctionsPass>(quantization_method);
}

}  // namespace quant
}  // namespace mlir
