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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc() {
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

// This transformation pass prepares for legalization to the TFLite dialect by
// converting Tensorlist operations in TensorFlow dialect into operations that
// can be legalized to TensorFlow Lite dialect with simple replacements.  The
// newly created operations are in the TensorFlow dialect if the operation can
// be represented using a TensorFlow op. Otherwise, TensorFlow Lite dialect op
// is used.

#include <climits>
#include <cstdint>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"  // from @llvm-project
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Block.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/Operation.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/PatternMatch.h"  // from @llvm-project
#include "mlir/IR/SymbolTable.h"  // from @llvm-project
#include "mlir/IR/TypeRange.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Types.h"  // from @llvm-project
#include "mlir/IR/UseDefLists.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/DialectConversion.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_a_m.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops_n_z.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_types.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/error_util.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/tensor_list.h"

#define DEBUG_TYPE "tf-tfl-legalization"

//===----------------------------------------------------------------------===//
// The actual LowerStaticTensorList Pass.
//
namespace mlir {
namespace {

/// Lower TensorList ops in functions for subsequent legalization.
struct LowerStaticTensorListPass
    : public PassWrapper<LowerStaticTensorListPass, OperationPass<ModuleOp>> {
  LowerStaticTensorListPass() = default;
  LowerStaticTensorListPass(const LowerStaticTensorListPass &) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_0(mht_0_v, 257, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "LowerStaticTensorListPass");
}
  explicit LowerStaticTensorListPass(bool allow_tensorlist_pass_through,
                                     bool default_to_single_batch,
                                     bool enable_dynamic_update_slice) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_1(mht_1_v, 263, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "LowerStaticTensorListPass");

    this->allow_tensorlist_pass_through = allow_tensorlist_pass_through;
    this->default_to_single_batch = default_to_single_batch;
    this->enable_dynamic_update_slice = enable_dynamic_update_slice;
  }

  StringRef getArgument() const final {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_2(mht_2_v, 272, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "getArgument");

    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tfl-lower-static-tensor-list";
  }
  StringRef getDescription() const final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_3(mht_3_v, 280, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "getDescription");

    // This is a brief description of the pass.
    return "Lower TensorList ops within TensorFlow Lite dialect";
  }

  void runOnOperation() override;

  Option<bool> allow_tensorlist_pass_through{
      *this, "allow-tensorlist-pass-through",
      llvm::cl::desc(
          "When specified to true, if the tensorlist ops can't be properly "
          "legalized by this pass, then the IR won't be changed so that "
          "tensorlist ops can pass through (default false)"),
      llvm::cl::init(false)};

  Option<bool> default_to_single_batch{
      *this, "default-to-single-batch",
      llvm::cl::desc(
          "When specified to true, if the tensorlist ops has unspecified batch "
          "size, this pass will assume that the batch size is one to proceed "
          "tensorlist op lowering (default true)"),
      llvm::cl::init(false)};

  Option<bool> enable_dynamic_update_slice{
      *this, "enable-dynamic-update-slice",
      llvm::cl::desc("When specified to true, lower TensorListSetItem with "
                     "DynamicUpdateSlice op (default false)"),
      llvm::cl::init(false)};
};

Value CreateI32SplatConst(Location loc, PatternRewriter *rewriter,
                          ArrayRef<int64_t> shape, int32_t val) {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_4(mht_4_v, 314, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "CreateI32SplatConst");

  RankedTensorType type =
      RankedTensorType::get(shape, rewriter->getIntegerType(32));
  DenseElementsAttr attr =
      DenseElementsAttr::get(type, rewriter->getI32IntegerAttr(val));
  return rewriter->create<arith::ConstantOp>(loc, type, attr);
}

Value CreateI64SplatConst(Location loc, PatternRewriter *rewriter,
                          ArrayRef<int64_t> shape, int64_t val) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_5(mht_5_v, 326, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "CreateI64SplatConst");

  RankedTensorType type =
      RankedTensorType::get(shape, rewriter->getIntegerType(64));
  DenseElementsAttr attr =
      DenseElementsAttr::get(type, rewriter->getI64IntegerAttr(val));
  return rewriter->create<arith::ConstantOp>(loc, type, attr);
}

Value CreateI32SplatTensor(Location loc, PatternRewriter *rewriter,
                           Value shape_tensor, int32_t val) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_6(mht_6_v, 338, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "CreateI32SplatTensor");

  Value scalar_val = CreateI32SplatConst(loc, rewriter, {}, val);
  return rewriter->create<TF::FillOp>(
      loc, RankedTensorType::get({-1}, rewriter->getIntegerType(32)),
      shape_tensor, scalar_val);
}

// Returns a new type by prepending the specified dimension to the shape of
// the given type if it is a ranked type.
Type PrependLeadingDimIfRanked(int64_t dim, Type type,
                               PatternRewriter *rewriter) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_7(mht_7_v, 351, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "PrependLeadingDimIfRanked");

  Type dtype = getElementTypeOrSelf(type);
  if (RankedTensorType ty = type.dyn_cast<RankedTensorType>()) {
    llvm::SmallVector<int64_t, 4> shape = {dim};
    shape.append(ty.getShape().begin(), ty.getShape().end());
    return RankedTensorType::get(shape, dtype);
  }
  return type;
}

Type GetTensorTypeForTensorList(Type element_type, TF::VariantType handle_dtype,
                                PatternRewriter *rewriter) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_8(mht_8_v, 365, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "GetTensorTypeForTensorList");

  // If the variant type in the output handle has item shape available, use it
  // to derive the output shape by setting unknown leading dimension.
  // Otherwise, result type will be of unranked type.
  if (handle_dtype.getSubtypes().empty()) {
    return UnrankedTensorType::get(element_type);
  }
  return PrependLeadingDimIfRanked(-1, handle_dtype.getSubtypes()[0], rewriter);
}

// Gets the index of tensorlist arguments which size might get changed by the
// function.
llvm::SmallSet<int, 4> GetResizedTensorListIndexes(
    FuncOp func, const llvm::SmallSet<int, 4> &tensor_list_args) {
  // `indexes` stores the argument index of tensorlists which size may get
  // updated in the function.
  llvm::SmallSet<int, 4> indexes;
  for (BlockArgument &arg : func.getArguments()) {
    if (tensor_list_args.contains(arg.getArgNumber())) {
      for (const mlir::OpOperand &use : arg.getUses()) {
        mlir::Operation *op = use.getOwner();
        // Currently we only check if the tensorlist argument is consumed by
        // `TensorListPushBack` or `TensorListResize`, since those are the only
        // length-mutating ops supported in this pass.
        if (llvm::isa<TF::TensorListPushBackOp>(op) ||
            llvm::isa<TF::TensorListResizeOp>(op)) {
          indexes.insert(arg.getArgNumber());
        }
      }
    }
  }
  return indexes;
}

// Creates a slice of the tensorlist `input_list`, starting from
// [start_index, 0, ...0], with size [size, -1, ...-1].
//
// Requires that `start_index` and `size` are scalar tensors and
// `item_position_shape` is a 1-D tensor with only one element equal to the rank
// of an item in the tensorlist.
TF::SliceOp CreateSliceOpForTensorList(Location loc, Value input_list,
                                       Value start_index, Value size,
                                       Value item_rank, Type result_type,
                                       PatternRewriter *rewriter) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_9(mht_9_v, 411, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "CreateSliceOpForTensorList");

  // Create the start position of slice. This is done by concatenating
  // `start_index` and `partial_start_position` together.
  IntegerType shape_dtype = rewriter->getIntegerType(32);
  RankedTensorType position_type = RankedTensorType::get({-1}, shape_dtype);
  Value partial_start_position =
      CreateI32SplatTensor(loc, rewriter, item_rank, 0);
  Value scalar_zero = CreateI32SplatConst(loc, rewriter, {}, 0);
  RankedTensorType vector_type = RankedTensorType::get({1}, shape_dtype);
  auto expanded_start_index = rewriter->create<TF::ExpandDimsOp>(
      loc, vector_type, start_index, scalar_zero);
  auto start_position = rewriter->create<TF::ConcatOp>(
      loc, position_type, scalar_zero,
      ArrayRef<Value>({expanded_start_index, partial_start_position}));

  // Create the slice size tensor. This is done by concatenating `size` and
  // `partial_size`.
  auto size_leading_dim =
      rewriter->create<TF::ExpandDimsOp>(loc, vector_type, size, scalar_zero);
  Value partial_size = CreateI32SplatTensor(loc, rewriter, item_rank, -1);
  auto slice_size = rewriter->create<TF::ConcatOp>(
      loc, position_type, scalar_zero,
      ArrayRef<Value>({size_leading_dim, partial_size}));

  return rewriter->create<TF::SliceOp>(loc, result_type, input_list,
                                       start_position, slice_size);
}

template <typename OpT>
class TensorListOpConverterBase : public OpConversionPattern<OpT> {
 public:
  explicit TensorListOpConverterBase<OpT>(MLIRContext *context,
                                          bool allow_tensorlist_pass_through,
                                          bool default_to_single_batch)
      : OpConversionPattern<OpT>::OpConversionPattern(context),
        allow_tensorlist_pass_through_(allow_tensorlist_pass_through),
        default_to_single_batch_(default_to_single_batch) {}

 protected:
  // This flag will control the behavior of error emitting during rewrite:
  // 1) If it's true, then patterns will only emit errors during debug or
  // tracing mode. 2) If it's false, then patterns will emit standard errors
  // when there is a rewrite failure.
  bool allow_tensorlist_pass_through_;

  // This flag will control the behavior of setting the batch size one when the
  // given batch size is None in order to force to proceed the tensor list op
  // lowerings.
  bool default_to_single_batch_;
};

// Converts tf.Const containing variant of type TensorList to a tensor of
// primitive element types. Each of the individual tensor in the list is
// converted to an ElementsAttr and then those are packed together using
// tf.Pack op.
struct ConvertConst : public OpConversionPattern<TF::ConstOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TF::ConstOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_10(mht_10_v, 474, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "matchAndRewrite");

    // Verify that the opaque elements attribute contains tensor of type variant
    // and scalar shape. The variant type should hold a TensorList.
    auto opaque_attr = op.value().dyn_cast<OpaqueElementsAttr>();
    if (!opaque_attr) return failure();
    tensorflow::Tensor tensor;
    if (!tensorflow::ConvertToTensor(opaque_attr, &tensor).ok())
      return failure();
    if (tensor.dtype() != tensorflow::DT_VARIANT) return failure();
    if (!tensorflow::TensorShapeUtils::IsScalar(tensor.shape()))
      return failure();

    const tensorflow::TensorList *list =
        tensor.scalar<tensorflow::Variant>()().get<tensorflow::TensorList>();
    if (!list) return failure();

    // Verify output type is variant and contains exactly one ranked subtypes.
    auto variant_ty =
        getElementTypeOrSelf(op.getType()).dyn_cast<TF::VariantType>();
    if (!variant_ty) return failure();
    ArrayRef<TensorType> subtypes = variant_ty.getSubtypes();
    if (subtypes.size() != 1) return failure();
    RankedTensorType list_element_ty =
        subtypes.front().dyn_cast<RankedTensorType>();
    if (!list_element_ty) return failure();

    // Extract tensor elements for the TensorList and construct result type
    // based on the number of elements and element shape.
    const std::vector<tensorflow::Tensor> &tensors = list->tensors();
    llvm::SmallVector<int64_t, 4> result_shape = {
        static_cast<int64_t>(tensors.size())};
    result_shape.append(list_element_ty.getShape().begin(),
                        list_element_ty.getShape().end());
    auto result_ty =
        RankedTensorType::get(result_shape, list_element_ty.getElementType());

    // If the list is empty, directly create the final result instead of
    // creating the tf.Pack op. tf.Pack op requires at least one operand.
    if (tensors.empty()) {
      tensorflow::Tensor tensor(list->element_dtype,
                                tensorflow::TensorShape(result_shape));
      auto attr_or = tensorflow::ConvertTensor(tensor, &rewriter);
      if (!attr_or.ok()) return failure();
      rewriter.replaceOpWithNewOp<TF::ConstOp>(op, attr_or.ValueOrDie());
      return success();
    }

    // Extract individual tensor list element and combine them using the tf.Pack
    // op.
    Location loc = op.getLoc();
    llvm::SmallVector<Value, 4> values;
    values.reserve(tensors.size());
    for (const tensorflow::Tensor &tensor : tensors) {
      auto attr_or = tensorflow::ConvertTensor(tensor, &rewriter);
      if (!attr_or.ok()) return failure();

      auto value = rewriter.create<TF::ConstOp>(loc, attr_or.ValueOrDie());
      values.push_back(value);
    }
    rewriter.replaceOpWithNewOp<TF::PackOp>(
        op, result_ty, values, /*axis=*/rewriter.getI64IntegerAttr(0));
    return success();
  }
};

struct ConvertTensorListSetItem
    : public OpConversionPattern<TF::TensorListSetItemOp> {
  explicit ConvertTensorListSetItem(MLIRContext *context,
                                    bool enable_dynamic_update_slice = false)
      : OpConversionPattern<TF::TensorListSetItemOp>(context),
        enable_dynamic_update_slice(enable_dynamic_update_slice) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_11(mht_11_v, 547, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "ConvertTensorListSetItem");
}

  LogicalResult matchAndRewrite(
      TF::TensorListSetItemOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_12(mht_12_v, 554, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "matchAndRewrite");

    if (enable_dynamic_update_slice) {
      return matchAndRewriteImplWithDynamicUpdateSlice(op, adaptor, rewriter);
    } else {
      return matchAndRewriteImplWithSliceAndConcat(op, adaptor, rewriter);
    }
  }

  // This function rewrites the original op into a series of slice and concat op
  // to produce the same result. It first slices the first `$index` rows. Then
  // expands the dimension of the `$item`, followed by another slice of the
  // remaining rows starting from `$index` + 1. Lastly it concatenates the
  // three parts together.
  // On a high level, it's doing something like:
  // def : Pat<(TF_TensorListSetItemOp $input, $index, $item),
  //      (Concat
  //        concat_dim = 0,
  //        (Slice $input, [0, 0, ...], (Concat (ExpandDims $index, expand_dim =
  //        0), [-1, -1, ...])), (ExpandDims $item, expand_dim = 0), (Slice
  //        $input, [$index + 1, 0, 0, ...], [-1, -1, ...]))>;
  LogicalResult matchAndRewriteImplWithSliceAndConcat(
      TF::TensorListSetItemOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_13(mht_13_v, 579, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "matchAndRewriteImplWithSliceAndConcat");

    Location loc = op.getLoc();
    Value input = adaptor.getOperands()[0];
    Value index = adaptor.getOperands()[1];
    Value item = adaptor.getOperands()[2];

    IntegerType shape_dtype = rewriter.getIntegerType(32);
    auto item_rank = rewriter.create<TF::RankOp>(
        loc, RankedTensorType::get({}, shape_dtype), item);
    Value scalar_zero = CreateI32SplatConst(loc, &rewriter, {}, 0);

    // Calculate `index` + 1, which is used to generate the start position for
    // the second slice op.
    auto suffix_start =
        rewriter.create<TF::AddOp>(loc, index.getType(), index,
                                   CreateI32SplatConst(loc, &rewriter, {}, 1));

    auto item_position_shape = rewriter.create<TF::ExpandDimsOp>(
        loc, RankedTensorType::get({1}, shape_dtype), item_rank, scalar_zero);
    // Create two slice ops.
    Type element_type = input.getType().cast<TensorType>().getElementType();
    UnrankedTensorType unranked_tensor = UnrankedTensorType::get(element_type);
    Value scalar_minus_one = CreateI32SplatConst(loc, &rewriter, {}, -1);
    TF::SliceOp slice1 =
        CreateSliceOpForTensorList(loc, /*input_list=*/input,
                                   /*start_index=*/scalar_zero,
                                   /*size=*/index,
                                   /*item_rank=*/item_position_shape,
                                   /*result_type=*/unranked_tensor, &rewriter);
    TF::SliceOp slice2 =
        CreateSliceOpForTensorList(loc, /*input_list=*/input,
                                   /*start_index=*/suffix_start,
                                   /*size=*/scalar_minus_one,
                                   /*item_rank=*/item_position_shape,
                                   /*result_type=*/unranked_tensor, &rewriter);

    // Expand the dimension of item so that it will have the same rank with
    // input.
    auto expanded_item = rewriter.create<TF::ExpandDimsOp>(
        op.getLoc(), unranked_tensor, item, scalar_zero);

    // Concatenate three parts together to generate the final result.
    rewriter.replaceOpWithNewOp<TF::ConcatOp>(
        op, input.getType(), scalar_zero,
        ArrayRef<Value>({slice1, expanded_item, slice2}));
    return success();
  }

  // This function rewrites the original op into a XLA DynamicUpdateSlice op.
  // |item| is expanded to have the same dimension as input_handle and
  // |index| is expanded to [index, 0, 0, ...] as the indices to input_handle.
  // On a high level, it's doing something like:
  // def : Pat<(TensorListSetItem($input_handle, $index, $item)),
  //           (XlaDynamicUpdateSlice($input_handle, ExpandDims($item, 0),
  //              Concat(ExpandDims($index, 0), [0, 0, 0, ...])))>
  LogicalResult matchAndRewriteImplWithDynamicUpdateSlice(
      TF::TensorListSetItemOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_14(mht_14_v, 639, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "matchAndRewriteImplWithDynamicUpdateSlice");

    Location loc = op.getLoc();
    Value input = adaptor.getOperands()[0];
    Value index = adaptor.getOperands()[1];
    Value item = adaptor.getOperands()[2];

    IntegerType shape_dtype = rewriter.getIntegerType(32);
    auto item_rank = rewriter.create<TF::RankOp>(
        loc, RankedTensorType::get({}, shape_dtype), item);
    Value scalar_zero = CreateI32SplatConst(loc, &rewriter, {}, 0);

    // Concat(ExpandDims(index, 0), [0, 0, 0, ...])
    RankedTensorType position_type = RankedTensorType::get({-1}, shape_dtype);
    auto item_position_shape = rewriter.create<TF::ExpandDimsOp>(
        loc, RankedTensorType::get({1}, shape_dtype), item_rank, scalar_zero);
    Value partial_index =
        CreateI32SplatTensor(loc, &rewriter, item_position_shape, 0);
    RankedTensorType vector_type = RankedTensorType::get({1}, shape_dtype);
    auto expanded_index =
        rewriter.create<TF::ExpandDimsOp>(loc, vector_type, index, scalar_zero);
    auto start_position = rewriter.create<TF::ConcatOp>(
        loc, position_type, scalar_zero,
        ArrayRef<Value>({expanded_index, partial_index}));

    // Expand the dimension of item so that it will have the same rank with
    // input.
    // ExpandDims(item, 0)
    Type element_type = input.getType().cast<TensorType>().getElementType();
    UnrankedTensorType unranked_tensor = UnrankedTensorType::get(element_type);
    auto expanded_item = rewriter.create<TF::ExpandDimsOp>(
        op.getLoc(), unranked_tensor, item, scalar_zero);

    // Update the element with XlaDynamicUpdateSliceOp.
    rewriter.replaceOpWithNewOp<TF::XlaDynamicUpdateSliceOp>(
        op, input.getType(), input, expanded_item, start_position);
    return success();
  }

  bool enable_dynamic_update_slice;
};

// Rewrites op of the template type initializing a TensorList with a list of ops
// to generate an equivalent raw tensor. Derived classes are required to
// override GetNumElements method.
template <typename OpT>
struct ConvertTensorListInitOp : public TensorListOpConverterBase<OpT> {
  using TensorListOpConverterBase<OpT>::TensorListOpConverterBase;
  using TensorListOpConverterBase<OpT>::allow_tensorlist_pass_through_;
  using TensorListOpConverterBase<OpT>::default_to_single_batch_;

  // Create and return a 1-d tensor with exactly one element equal to the number
  // of list elements to initialize the output tensor list with.
  virtual Value GetNumElements(OpT op, ValueRange operands,
                               PatternRewriter *rewriter) const = 0;

  // Rewrites the original op into `tf.fill`. The result tensor shape is
  // [num_element, element_shape]. All the values in the result tensor will be
  // initialized to 0.
  LogicalResult matchAndRewrite(
      OpT op, typename OpT::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_15(mht_15_v, 702, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "matchAndRewrite");

    Type dtype = op.element_dtype();
    if (!(dtype.isF16() || dtype.isF32() || dtype.isF64() ||
          dtype.isInteger(1) || dtype.isInteger(8) || dtype.isInteger(16) ||
          dtype.isInteger(32) || dtype.isInteger(64))) {
      const char *error_info =
          "requires element_dtype to be 1-bit/8-bit/16-bit/32-bit/64-bit "
          "integer or 16-bit/32-bit/64-bit float type during TF Lite "
          "transformation pass";
      return allow_tensorlist_pass_through_
                 ? rewriter.notifyMatchFailure(op, error_info)
                 : op.emitOpError(error_info);
    }

    Value element_shape = adaptor.getOperands()[0];
    Type shape_dtype = getElementTypeOrSelf(element_shape.getType());
    // If the `element_shape` is a scalar, we try to acquire its shape by
    // looking at the first `TensorListSetItemOp` writing to this tensor list.
    // Here we assume that the element_shape won't be changed before calling
    // the first `TensorListSetItemOp`.
    if (auto shaped_type = element_shape.getType().dyn_cast<ShapedType>()) {
      if (shaped_type.hasRank() && shaped_type.getRank() == 0) {
        bool element_shape_acquired = false;
        auto uses = op.getResult().getUses();
        for (auto &use : llvm::make_early_inc_range(uses)) {
          if (TF::TensorListSetItemOp set_op =
                  llvm::dyn_cast<TF::TensorListSetItemOp>(use.getOwner())) {
            element_shape = rewriter.create<TF::ShapeOp>(
                op.getLoc(), RankedTensorType::get({-1}, shape_dtype),
                set_op.item());
            element_shape_acquired = true;
          } else if (TF::WhileOp while_op =
                         llvm::dyn_cast<TF::WhileOp>(use.getOwner())) {
            // Tensorlist is passed into a while loop, check inside the body
            // function.
            auto inside_uses = while_op.body_function()
                                   .getArgument(use.getOperandNumber())
                                   .getUses();
            for (auto &inside_use : llvm::make_early_inc_range(inside_uses)) {
              if (TF::TensorListSetItemOp set_op =
                      llvm::dyn_cast<TF::TensorListSetItemOp>(
                          inside_use.getOwner())) {
                if (auto shaped_type =
                        set_op.item().getType().dyn_cast<ShapedType>()) {
                  if (shaped_type.hasStaticShape()) {
                    RankedTensorType type = RankedTensorType::get(
                        {shaped_type.getRank()}, rewriter.getIntegerType(32));
                    SmallVector<Attribute, 4> shape_attr;
                    for (int64_t dim : shaped_type.getShape()) {
                      shape_attr.push_back(rewriter.getI32IntegerAttr(dim));
                    }
                    DenseElementsAttr attr =
                        DenseElementsAttr::get(type, shape_attr);
                    element_shape = rewriter.create<arith::ConstantOp>(
                        op.getLoc(), type, attr);
                    element_shape_acquired = true;
                    break;
                  }
                }
              }
            }
          }
          if (element_shape_acquired) break;
        }
        if (!element_shape_acquired) {
          const char *error_info =
              "requires element_shape to be 1D tensor during TF Lite "
              "transformation pass";
          return allow_tensorlist_pass_through_
                     ? rewriter.notifyMatchFailure(op, error_info)
                     : op.emitOpError(error_info);
        }
      }
    }

    DenseIntElementsAttr dense_elem_attr;
    if (matchPattern(element_shape, m_Constant(&dense_elem_attr))) {
      // Note: It's technically unsafe to rewrite
      //     TensorListReserve(num_element, element_shape)
      // to
      //     Fill(Concat(num_element, element_shape), 0)
      // because element_shape may contain -1 to represent unknown dimension.
      //
      // In real world use cases (e.g. Keras RNN), `element_shape` is usually
      // a constant, and the first dimension of `element_shape` is usually
      // batch dimension. Currently TFLiteConverter always rewrite unknown
      // batch dimension to 1, therefore we also rewrite unknown dimension in
      // `element_shape` to 1 here.
      //
      // This workaround enables converting Keras RNN without specifying batch
      // dimension. This isn't guaranteed to work, but it doesn't break any
      // non-broken cases either (since it's already broken if `element_shape`
      // contains -1).
      // TODO(b/142096690): Support dynamic element shape and remove the
      // workaround.
      SmallVector<int32_t, 4> new_element_shape_values;

      auto int_values = dense_elem_attr.getValues<APInt>();
      for (auto it = int_values.begin(); it != int_values.end(); ++it) {
        auto dim_value = (*it).getSExtValue();
        if (it == int_values.begin() && dim_value == -1) {
          if (!default_to_single_batch_) {
            const char *error_info =
                "requires element_shape to be static during TF Lite "
                "transformation pass";
            return allow_tensorlist_pass_through_
                       ? rewriter.notifyMatchFailure(op, error_info)
                       : op.emitOpError(error_info);
          }
          dim_value = 1;
        }
        new_element_shape_values.push_back(dim_value);
      }

      auto attr = DenseIntElementsAttr::get(
          element_shape.getType().cast<ShapedType>(), new_element_shape_values);
      auto new_element_shape = rewriter.create<arith::ConstantOp>(
          op.getLoc(), element_shape.getType(), attr);
      element_shape = new_element_shape;
    }

    int64_t result_rank = -1;  // -1 means unknown result rank.
    Type element_dtype = op.element_dtype();
    Type result_type = UnrankedTensorType::get(element_dtype);
    Value leading_dim = GetNumElements(op, adaptor.getOperands(), &rewriter);
    if (auto element_type =
            op.element_type().template dyn_cast<RankedTensorType>()) {
      result_rank = element_type.getRank() + 1;
      int64_t leading_dim_v = -1;
      ElementsAttr element_attr;
      if (matchPattern(leading_dim, m_Constant(&element_attr))) {
        leading_dim_v = element_attr.getValues<APInt>()[0].getSExtValue();
      }
      SmallVector<int64_t, 4> result_shape = {leading_dim_v};
      ArrayRef<int64_t> shape = element_type.getShape();
      result_shape.append(shape.begin(), shape.end());
      result_type = RankedTensorType::get(result_shape, element_dtype);
    }

    // Create a 1-D RankedTensorType for result's shape. Number of elements in
    // it is equal to the rank of the result, if known. Otherwise, the number of
    // elements are unknown and represented with -1. In both cases, we can
    // specify dimension using rank of the result.
    Type shape_type = RankedTensorType::get({result_rank}, shape_dtype);

    Location loc = op.getLoc();
    // Add number of elements as the prefix to the element shape to get shape of
    // the output tensor.
    Value scalar_zero = CreateI32SplatConst(loc, &rewriter, {}, 0);
    auto list_shape = rewriter.create<TF::ConcatOp>(
        loc, shape_type, scalar_zero,
        ArrayRef<Value>({leading_dim, element_shape}));

    // Create a zero-initialized constant tensor that has the same type
    // as specified by element_dtype.
    RankedTensorType zero_type = RankedTensorType::get({}, element_dtype);
    Attribute zero_attr = rewriter.getZeroAttr(zero_type);
    auto zero = rewriter.create<arith::ConstantOp>(loc, zero_type, zero_attr);

    rewriter.replaceOpWithNewOp<TF::FillOp>(op, result_type, list_shape, zero);
    return success();
  }
};

struct ConvertTensorListReserve
    : public ConvertTensorListInitOp<TF::TensorListReserveOp> {
  explicit ConvertTensorListReserve(MLIRContext *context,
                                    bool allow_tensorlist_pass_through,
                                    bool default_to_single_batch)
      : ConvertTensorListInitOp(context, allow_tensorlist_pass_through,
                                default_to_single_batch) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_16(mht_16_v, 875, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "ConvertTensorListReserve");
}

  Value GetNumElements(TF::TensorListReserveOp op, ValueRange operands,
                       PatternRewriter *rewriter) const override {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_17(mht_17_v, 881, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "GetNumElements");

    Value scalar_zero = CreateI32SplatConst(op.getLoc(), rewriter, {}, 0);
    Type shape_dtype = getElementTypeOrSelf(op.element_shape().getType());
    Value num_elements = operands[1];
    IntegerAttr attr;
    if (matchPattern(num_elements, m_Constant(&attr))) {
      return CreateI32SplatConst(op.getLoc(), rewriter, {1}, attr.getInt());
    }
    if (auto const_op = num_elements.getDefiningOp<TF::ConstOp>()) {
      return CreateI32SplatConst(op->getLoc(), rewriter, {1},
                                 (*const_op.value()
                                       .cast<DenseElementsAttr>()
                                       .getValues<APInt>()
                                       .begin())
                                     .getSExtValue());
    }
    return rewriter->create<TF::ExpandDimsOp>(
        op.getLoc(), RankedTensorType::get({1}, shape_dtype), num_elements,
        scalar_zero);
  }
};

// Note that we ignore the second operand `max_num_elements` as we don't have
// any restrictions on the number of elements we can support. So this may
// have a different behavior compared to TensorFlow in case of errors.
struct ConvertEmptyTensorList
    : public ConvertTensorListInitOp<TF::EmptyTensorListOp> {
  explicit ConvertEmptyTensorList(MLIRContext *context,
                                  bool allow_tensorlist_pass_through,
                                  bool default_to_single_batch)
      : ConvertTensorListInitOp(context, allow_tensorlist_pass_through,
                                default_to_single_batch) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_18(mht_18_v, 915, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "ConvertEmptyTensorList");
}

  Value GetNumElements(TF::EmptyTensorListOp op, ValueRange operands,
                       PatternRewriter *rewriter) const override {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_19(mht_19_v, 921, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "GetNumElements");

    return CreateI32SplatConst(op.getLoc(), rewriter, {1}, 0);
  }
};

struct ConvertTensorListPushBack
    : public OpConversionPattern<TF::TensorListPushBackOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TF::TensorListPushBackOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_20(mht_20_v, 935, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "matchAndRewrite");

    Value input_handle = adaptor.getOperands()[0];
    Value item = adaptor.getOperands()[1];

    // Expand the shape of the item so that it will have rank same as the input
    // tensor and it is compatible for the Concat Op.
    Type expanded_item_type =
        PrependLeadingDimIfRanked(1, item.getType(), &rewriter);
    Location loc = op.getLoc();
    Value scalar_zero = CreateI32SplatConst(loc, &rewriter, {}, 0);
    auto expanded_item = rewriter.create<TF::ExpandDimsOp>(
        loc, expanded_item_type, item, scalar_zero);

    Type elem_type = getElementTypeOrSelf(item);
    auto handle_dtype = getElementTypeOrSelf(op.output_handle().getType())
                            .cast<TF::VariantType>();
    Type result_type =
        GetTensorTypeForTensorList(elem_type, handle_dtype, &rewriter);

    // Concatenate tensor stored in the input handle with the expanded item to
    // get a tensor equivalent to the TensorList generated by this op.
    rewriter.replaceOpWithNewOp<TF::ConcatOp>(
        op, result_type, scalar_zero,
        ArrayRef<Value>({input_handle, expanded_item}));
    return success();
  }
};

// Rewrites `TensorListResize` op into a functional `If` op and several basic
// TF ops to match the op semantics of Tensorflow. Basically, it does:
// 1) If the requested size is smaller or equal than the input tensorlist's
// size, rewrite it to a Slice op so that only the first 'size' rows are
// returned. 2) If the requested size is larger than the input tensorlist's
// size. We need to create an additional tensorlist with 'size - input_size'
// elements, and append it to the end of the input tensorlist.
struct ConvertTensorListResize
    : public OpConversionPattern<TF::TensorListResizeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TF::TensorListResizeOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_21(mht_21_v, 979, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "matchAndRewrite");

    Value input_handle = adaptor.getOperands()[0];
    Value size = adaptor.getOperands()[1];

    Location loc = op.getLoc();
    Value scalar_zero = CreateI32SplatConst(loc, &rewriter, {}, 0);

    // Compute the input tensorlist's length and store it in `input_size`.
    IntegerType shape_dtype = rewriter.getIntegerType(32);
    auto input_size = rewriter.create<TF::TensorListLengthOp>(
        loc, RankedTensorType::get({}, shape_dtype), op.getOperand(0));

    // Infer result type of this op based on TF's shape inference result.
    Type elem_type = getElementTypeOrSelf(input_handle);
    auto handle_dtype = getElementTypeOrSelf(op.output_handle().getType())
                            .cast<TF::VariantType>();
    Type result_type =
        GetTensorTypeForTensorList(elem_type, handle_dtype, &rewriter);

    // Compute the difference of `size` and `input_size`, and store it in
    // `size_diff`, which is then consumed by `if_cond`.
    auto size_diff = rewriter.create<TF::SubOp>(
        loc, RankedTensorType::get({}, shape_dtype), size, input_size);
    auto if_cond = rewriter.create<TF::GreaterOp>(
        loc, RankedTensorType::get({}, rewriter.getI1Type()), size_diff,
        scalar_zero);

    // Build the argument/result types for if branch function.
    auto input_shape = rewriter.create<TF::ShapeOp>(
        loc, RankedTensorType::get({-1}, shape_dtype), input_handle);

    Type branch_args_type[] = {input_handle.getType(), input_shape.getType(),
                               size_diff.getType(), size.getType()};
    Type branch_result_type[] = {result_type};
    auto func_type = FunctionType::get(rewriter.getContext(), branch_args_type,
                                       branch_result_type);

    // Create functions in a higher scope before restoring the insertion point.
    // Additionally, create the SymbolTable before further modifying the module.
    auto original_point = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointAfter(op->getParentOfType<FuncOp>());
    SymbolTable manager(op->getParentOfType<ModuleOp>());

    // Constructs `then_branch`, which is executed when `if_cond` evaluates to
    // true.
    auto then_branch_op = rewriter.create<FuncOp>(loc, "cond_true", func_type);
    CreateCondTrueBranch(op, shape_dtype, result_type, then_branch_op,
                         &rewriter);
    then_branch_op.setVisibility(FuncOp::Visibility::Private);

    // Constructs `else_branch`, which is executed when `if_cond` evaluates to
    // false.
    auto else_branch_op = rewriter.create<FuncOp>(loc, "cond_false", func_type);
    CreateCondFalseBranch(loc, shape_dtype, result_type, else_branch_op,
                          &rewriter);
    else_branch_op.setVisibility(FuncOp::Visibility::Private);

    // Inserts the two blocks' names into the symbol table held by the module.
    // Using SymbolTable will ensure that the inserted symbol names are
    // unique.
    manager.insert(then_branch_op);
    manager.insert(else_branch_op);

    rewriter.restoreInsertionPoint(original_point);
    rewriter.replaceOpWithNewOp<TF::IfOp>(
        op, result_type, if_cond,
        /*input=*/
        ArrayRef<Value>({input_handle, input_shape, size_diff, size}),
        /*then_branch=*/
        mlir::SymbolRefAttr::get(then_branch_op),
        /*else_branch=*/
        mlir::SymbolRefAttr::get(else_branch_op),
        /*is_stateless=*/rewriter.getBoolAttr(true));
    return success();
  }

 private:
  // When the input tensorlist's size is smaller than the requested size,
  // then branch is executed.
  // Create a new tensorlist of size 'size - input_size' and concat it
  // with the input tensorlist.
  void CreateCondTrueBranch(TF::TensorListResizeOp resize_op, Type shape_dtype,
                            Type result_type, FuncOp branch_func,
                            ConversionPatternRewriter *rewriter) const {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_22(mht_22_v, 1065, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "CreateCondTrueBranch");

    auto guard = OpBuilder::InsertionGuard(*rewriter);
    auto inputs = branch_func.getFunctionType().getInputs();
    Block *block = rewriter->createBlock(
        &branch_func.getBody(), branch_func.begin(), inputs,
        SmallVector<Location>(inputs.size(), branch_func.getLoc()));

    auto input_shape = block->getArgument(1);
    auto size_diff = block->getArgument(2);
    auto input = block->getArgument(0);

    Location loc = resize_op.getLoc();
    // Get the element shape by slicing from index 1 in the input shape.
    Value slice_size = CreateI32SplatConst(loc, rewriter, {1}, -1);
    Value scalar_zero = CreateI32SplatConst(loc, rewriter, {}, 0);
    Value slice_start = CreateI32SplatConst(loc, rewriter, {1}, 1);
    auto elem_shape = rewriter->create<TF::SliceOp>(
        loc, RankedTensorType::get({-1}, shape_dtype), input_shape, slice_start,
        slice_size);
    auto extended_part = rewriter->create<TF::TensorListReserveOp>(
        loc, resize_op.output_handle().getType(), elem_shape, size_diff);
    // `ConcatOp` expects non-variant-typed input. Insert a
    // `TensorListStackOp` here to convert type from variant to non-variant.
    // Note that we are using the same `result_type` for both the
    // `TensorListStackOp` and `ConcatOp`, since the first dimension of the
    // shape specified by `result_type` is -1.
    auto stacked_extended_part = rewriter->create<TF::TensorListStackOp>(
        loc, result_type, extended_part,
        /*element_shape=*/CreateI32SplatConst(loc, rewriter, {}, -1),
        /*num_elements=*/rewriter->getI32IntegerAttr(-1));
    auto concat_op = rewriter->create<TF::ConcatOp>(
        loc, result_type, scalar_zero,
        ArrayRef<Value>({input, stacked_extended_part}));
    rewriter->create<func::ReturnOp>(loc, ArrayRef<Value>({concat_op}));
  }

  void CreateCondFalseBranch(Location loc, Type shape_dtype, Type result_type,
                             FuncOp branch_func,
                             ConversionPatternRewriter *rewriter) const {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_23(mht_23_v, 1106, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "CreateCondFalseBranch");

    // When the input tensorlist's size is larger or equal than the requested
    // size, the else branch is executed.
    // Slice the first 'size' rows from the input tensorlist.
    auto guard = OpBuilder::InsertionGuard(*rewriter);
    auto inputs = branch_func.getFunctionType().getInputs();
    Block *block = rewriter->createBlock(
        &branch_func.getBody(), branch_func.begin(), inputs,
        SmallVector<Location>(inputs.size(), branch_func.getLoc()));

    Value scalar_zero = CreateI32SplatConst(loc, rewriter, {}, 0);
    Value vector_one = CreateI32SplatConst(loc, rewriter, {1}, 1);
    auto input = block->getArgument(0);
    auto size = block->getArgument(3);

    // Subtract `input_rank` by 1 to get the item's rank, which is used as
    // `partial_position_shape`.
    auto input_rank = rewriter->create<TF::RankOp>(
        loc, RankedTensorType::get({}, shape_dtype), input);
    auto partial_position_shape = rewriter->create<TF::SubOp>(
        loc, RankedTensorType::get({1}, shape_dtype), input_rank, vector_one);
    auto slice_op =
        CreateSliceOpForTensorList(loc, /*input_list=*/input,
                                   /*start_index=*/scalar_zero, /*size=*/size,
                                   /*item_rank=*/partial_position_shape,
                                   /*result_type=*/result_type, rewriter);
    rewriter->create<func::ReturnOp>(loc, ArrayRef<Value>({slice_op}));
  }
};

struct ConvertTensorListGetItem
    : public OpConversionPattern<TF::TensorListGetItemOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TF::TensorListGetItemOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_24(mht_24_v, 1145, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "matchAndRewrite");

    Value input = adaptor.getOperands()[0];
    Value index = adaptor.getOperands()[1];
    rewriter.replaceOpWithNewOp<TF::GatherOp>(op, op.getType(), input, index,
                                              rewriter.getBoolAttr(true));
    return success();
  }
};

struct ConvertTensorListLength
    : public OpConversionPattern<TF::TensorListLengthOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TF::TensorListLengthOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_25(mht_25_v, 1163, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "matchAndRewrite");

    Location loc = op.getLoc();
    Value input_handle = adaptor.getOperands()[0];

    BoolAttr true_attr = rewriter.getBoolAttr(true);
    auto shape = rewriter.create<TF::ShapeOp>(loc, input_handle,
                                              /*use_32bit=*/true_attr);
    rewriter.replaceOpWithNewOp<TF::GatherOp>(
        op, op.getType(), shape, CreateI32SplatConst(loc, &rewriter, {}, 0),
        /*validate_indices=*/true_attr);
    return success();
  }
};

struct ConvertTensorListStack
    : public OpConversionPattern<TF::TensorListStackOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TF::TensorListStackOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_26(mht_26_v, 1186, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "matchAndRewrite");

    Location loc = op.getLoc();
    Value input = adaptor.getOperands()[0];
    Value element_shape = adaptor.getOperands()[1];

    // If the `element_shape` is a known constant (which is defined when calling
    // `tensor_list_stack`) and also valid (not scalar), we rewrite this op to a
    // trivial Reshape op (that doesn't actually change the input's shape) and
    // also populate the shape info to the op result. The shape of the
    // tensorlist is inferred from `num_elements` and `element_shape`.
    auto ranked_type = element_shape.getType().dyn_cast<RankedTensorType>();
    DenseIntElementsAttr dense_elem_attr;
    if ((ranked_type && ranked_type.getRank() == 0) ||
        !matchPattern(element_shape, m_Constant(&dense_elem_attr))) {
      // If no constant is spotted, just forward the operand.
      rewriter.replaceOp(op, {input});
      return success();
    }

    RankedTensorType shape_type =
        RankedTensorType::get({-1}, rewriter.getIntegerType(32));
    auto new_shape = rewriter.create<TF::ShapeOp>(loc, shape_type, input);
    SmallVector<int64_t, 8> output_shape(/*Size=*/1, op.num_elements());
    for (const auto &dim : dense_elem_attr.getValues<APInt>())
      output_shape.push_back(dim.getSExtValue());
    RankedTensorType result_type =
        RankedTensorType::get(output_shape, getElementTypeOrSelf(input));
    rewriter.replaceOpWithNewOp<TF::ReshapeOp>(op, result_type, input,
                                               new_shape);
    return success();
  }
};

// Converts `TensorListConcatV2` into Unpack and Concat. First we unpack
// the input tensorlist along the first dimension, which results in N (where N
// is the first dim's size) tensors (each with shape [element_shape]). Then
// we concatenate all those tensors along the first dimension.
// The pattern will be rejected if either `element_shape` is not constant, or
// the first dimension of `input` is not known.
struct ConvertTensorListConcatV2
    : public TensorListOpConverterBase<TF::TensorListConcatV2Op> {
  using TensorListOpConverterBase<
      TF::TensorListConcatV2Op>::TensorListOpConverterBase;
  using TensorListOpConverterBase<
      TF::TensorListConcatV2Op>::allow_tensorlist_pass_through_;

  LogicalResult matchAndRewrite(
      TF::TensorListConcatV2Op op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_27(mht_27_v, 1237, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "matchAndRewrite");

    Location loc = op.getLoc();
    Value input = adaptor.getOperands()[0];
    Value element_shape = adaptor.getOperands()[1];

    // Only match when `element_shape` is a constant.
    DenseIntElementsAttr dense_elem_attr;
    if (!matchPattern(element_shape, m_Constant(&dense_elem_attr))) {
      const char *error_info = "requires element_shape to be a constant";
      return allow_tensorlist_pass_through_
                 ? rewriter.notifyMatchFailure(op, error_info)
                 : op.emitOpError(error_info);
    }
    llvm::SmallVector<int64_t, 4> output_shape;
    for (const auto &dim : dense_elem_attr.getValues<APInt>()) {
      output_shape.push_back(dim.getSExtValue());
    }

    // First unpack the input tensor along the first dimension.
    Type input_element_type = getElementTypeOrSelf(input);
    int64_t num_unpacked = 0;
    if (auto type = input.getType().dyn_cast<RankedTensorType>()) {
      if (type.getDimSize(0) > 0) {
        num_unpacked = type.getDimSize(0);
      } else {
        const char *error_info =
            "requires the first dimension of input tensor to have > 0 "
            "dimension";
        return allow_tensorlist_pass_through_
                   ? rewriter.notifyMatchFailure(op, error_info)
                   : op.emitOpError(error_info);
      }
    }
    llvm::SmallVector<Type, 1> unpack_output_type;
    unpack_output_type.insert(
        unpack_output_type.begin(), num_unpacked,
        RankedTensorType::get(output_shape, input_element_type));
    auto unpack = rewriter.create<TF::UnpackOp>(loc, unpack_output_type, input,
                                                /*axis=*/0);

    // Concatenate the unpacked tensors along the first dimension.
    // Since we're concatenating along first dimension, change its dim size to
    // -1.
    output_shape[0] = -1;
    Value scalar_zero = CreateI32SplatConst(loc, &rewriter, {}, 0);
    auto concat = rewriter.create<TF::ConcatOp>(
        loc, RankedTensorType::get(output_shape, input_element_type),
        scalar_zero, unpack->getResults());
    // `lengths` is only useful for computing gradient. For now we just return
    // a placeholder tensor.
    rewriter.replaceOp(
        op, {concat.getResult(), CreateI64SplatConst(loc, &rewriter, {0}, 0)});
    return success();
  }
};

struct ConvertIdentity : public OpConversionPattern<TF::IdentityOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TF::IdentityOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_28(mht_28_v, 1301, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "matchAndRewrite");

    Value input = adaptor.getOperands()[0];
    rewriter.replaceOpWithNewOp<TF::IdentityOp>(
        op, input.getType(), adaptor.getOperands(), op->getAttrs());
    return success();
  }
};

struct ConvertReturn : public OpConversionPattern<func::ReturnOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::ReturnOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_29(mht_29_v, 1317, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "matchAndRewrite");

    rewriter.updateRootInPlace(op,
                               [&] { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

struct ConvertYield : public OpConversionPattern<TF::YieldOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TF::YieldOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_30(mht_30_v, 1332, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "matchAndRewrite");

    rewriter.updateRootInPlace(op,
                               [&] { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

// Returns an unranked tensor type with an element of the same type as `value`
// if `type` is a tensor of variant. Otherwise, returns `type` unmodified.
Type VariantToUnrankedTensorType(Type type, Value value) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_31(mht_31_v, 1344, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "VariantToUnrankedTensorType");

  TF::VariantType variant_ty =
      getElementTypeOrSelf(type).dyn_cast<TF::VariantType>();
  if (!variant_ty) {
    return type;
  }
  if (!variant_ty.getSubtypes().empty()) {
    // Short-circut if the variant type has subtype info.
    return UnrankedTensorType::get(
        variant_ty.getSubtypes()[0].getElementType());
  }
  Type value_type = value.getType();
  Type element_type;
  variant_ty = value_type.dyn_cast<TF::VariantType>();
  if (variant_ty && !variant_ty.getSubtypes().empty()) {
    element_type = variant_ty.getSubtypes()[0].getElementType();
  } else {
    element_type = getElementTypeOrSelf(value_type);
  }
  return UnrankedTensorType::get(element_type);
}

// Returns true if we can deduce the type is tensorlist.
bool IsTensorListType(Type type, llvm::Optional<Value> value) {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_32(mht_32_v, 1370, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "IsTensorListType");

  TF::VariantType variant_ty =
      getElementTypeOrSelf(type).dyn_cast<TF::VariantType>();
  if (!variant_ty) {
    return false;
  }
  // Check there is only one subtype contained in the variant type. Note that
  // when `subtypes.size() == 1` does not always mean the type is actually
  // a tensorlist. We probably need some form of data flow analysis.
  if (variant_ty.getSubtypes().size() == 1) {
    return true;
  }
  // If subtype info is not available, check if the value is used by any of
  // the following TensorList operations.
  if (!value.hasValue()) {
    return false;
  }
  for (const mlir::OpOperand &use : value.getValue().getUses()) {
    mlir::Operation *op = use.getOwner();
    if (llvm::isa<TF::TensorListGetItemOp>(op) ||
        llvm::isa<TF::TensorListLengthOp>(op) ||
        llvm::isa<TF::TensorListPushBackOp>(op) ||
        llvm::isa<TF::TensorListReserveOp>(op) ||
        llvm::isa<TF::TensorListSetItemOp>(op) ||
        llvm::isa<TF::TensorListStackOp>(op) ||
        llvm::isa<TF::TensorListResizeOp>(op)) {
      return true;
    }
  }
  return false;
}

// Returns a set of integers that correspond to the tensorlist arguments in
// the function.
llvm::SmallSet<int, 4> GetTensorListArgumentsIndex(FuncOp func) {
  llvm::SmallSet<int, 4> set;
  for (const auto &arg_and_idx : llvm::enumerate(func.getArguments())) {
    if (IsTensorListType(arg_and_idx.value().getType(), arg_and_idx.value())) {
      set.insert(arg_and_idx.index());
    }
  }
  return set;
}

// Returns a set of integers that correspond to the tensorlist results in the
// function.
llvm::SmallSet<int, 4> GetTensorListResultsIndex(FuncOp func) {
  llvm::SmallSet<int, 4> set;

  for (const auto &result_and_idx :
       llvm::enumerate(func.getFunctionType().getResults())) {
    if (IsTensorListType(result_and_idx.value(), llvm::None)) {
      set.insert(result_and_idx.index());
    }
  }
  return set;
}

// Updates the tensorlist types based on the input index. If the tensorlist's
// size isn't changed(which is indicated by `resized_tensor_list_index`), then
// we will use the original operand's type, otherwise update it with the
// unranked tensor type.
template <typename R>
void UpdateTensorListTypes(
    const llvm::SmallSet<int, 4> &tensor_list_index,
    const llvm::SmallSet<int, 4> &resized_tensor_list_index,
    ArrayRef<Type> types, R &&range, ValueRange operands,
    llvm::SmallVectorImpl<Type> *updated_types) {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_33(mht_33_v, 1440, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "UpdateTensorListTypes");

  int i = 0;
  for (const auto it : llvm::zip(types, range, operands)) {
    if (tensor_list_index.count(i)) {
      // Only change the tensorlist's type to unranked tensor if it has been
      // resized.
      if (resized_tensor_list_index.count(i)) {
        updated_types->push_back(
            VariantToUnrankedTensorType(std::get<0>(it), std::get<1>(it)));
      } else {
        updated_types->push_back(std::get<2>(it).getType());
      }
    } else {
      updated_types->push_back(std::get<0>(it));
    }
    ++i;
  }
}

// Updates the tensorlist types to unranked tensor types based on the input
// index.
template <typename R>
void ChangeVariantToUnrankedTensorType(
    const llvm::SmallSet<int, 4> &tensor_list_index, ArrayRef<Type> types,
    R &&range, llvm::SmallVectorImpl<Type> *updated_types) {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_34(mht_34_v, 1467, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "ChangeVariantToUnrankedTensorType");

  int i = 0;
  for (const auto it : llvm::zip(types, range)) {
    if (tensor_list_index.count(i)) {
      updated_types->push_back(
          VariantToUnrankedTensorType(std::get<0>(it), std::get<1>(it)));
    } else {
      updated_types->push_back(std::get<0>(it));
    }
    ++i;
  }
}

// Updates the specified function's type and region signature.
void UpdateFunctionAndRegionType(ConversionPatternRewriter &rewriter,
                                 FuncOp func,
                                 llvm::ArrayRef<Type> updated_argument_types,
                                 llvm::ArrayRef<Type> updated_result_types) {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_35(mht_35_v, 1487, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "UpdateFunctionAndRegionType");

  // Change `func`'s argument type to `unranked_argument_types`. If its
  // return types contain a `DT_VARIANT`, change it to the unranked type
  // derived from the corresponding argument.
  rewriter.updateRootInPlace(func, [&] {
    func.setType(FunctionType::get(func.getContext(), updated_argument_types,
                                   updated_result_types));
  });
  Region &entry = func.getRegion();
  TypeConverter::SignatureConversion signature_conversion(
      entry.getNumArguments());
  for (const BlockArgument &arg : entry.getArguments()) {
    signature_conversion.addInputs(arg.getArgNumber(),
                                   updated_argument_types[arg.getArgNumber()]);
  }
  rewriter.applySignatureConversion(&entry, signature_conversion);
}

// Changes the function type of `cond_func` and `body_func` for the given While
// op.
LogicalResult UpdateFunctionTypesForWhileOp(
    ConversionPatternRewriter &rewriter, TF::WhileOp op, ValueRange operands,
    const llvm::SmallSet<int, 4> &tensor_list_args,
    const llvm::SmallSet<int, 4> &resized_tensor_lists) {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_36(mht_36_v, 1513, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "UpdateFunctionTypesForWhileOp");

  int func_index = 0;
  for (FuncOp func : {op.cond_function(), op.body_function()}) {
    ++func_index;
    if (!func) continue;

    FunctionType func_type = func.getFunctionType();
    int num_inputs = func_type.getNumInputs();
    int num_results = func_type.getNumResults();

    // For each argument type in function's arguments, change it to uranked
    // tensor type if it's a variant type.
    SmallVector<Type, 8> updated_argument_types;
    updated_argument_types.reserve(num_inputs);
    UpdateTensorListTypes<mlir::OperandRange>(
        tensor_list_args, resized_tensor_lists, func_type.getInputs(),
        op.getOperands(), operands, &updated_argument_types);

    // Change all DT_VARIANT result types in function results to unranked tensor
    // type with element type derived from the corresponding input operand. This
    // is correct because while body's inputs and results have the same type.
    SmallVector<Type, 8> updated_result_types;
    updated_result_types.reserve(num_results);
    if (func_index == 1) {
      // We only update the result types for the body function.
      for (Type ty : func_type.getResults()) {
        updated_result_types.push_back(ty);
      }
    } else {
      UpdateTensorListTypes<mlir::OperandRange>(
          tensor_list_args, resized_tensor_lists, func_type.getResults(),
          op.getOperands(), operands, &updated_result_types);
    }

    UpdateFunctionAndRegionType(rewriter, func, updated_argument_types,
                                updated_result_types);
  }
  return success();
}

// Changes the function type of `then_function` and `else_function` for the
// given If op.
LogicalResult UpdateFunctionTypesForIfOp(
    ConversionPatternRewriter &rewriter, TF::IfOp op, ValueRange operands,
    const llvm::SmallSet<int, 4> &tensor_list_args,
    const llvm::SmallSet<int, 4> &resized_tensor_lists,
    llvm::ArrayRef<Type> updated_result_types) {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_37(mht_37_v, 1562, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "UpdateFunctionTypesForIfOp");

  for (FuncOp func : {op.else_function(), op.then_function()}) {
    if (!func) continue;

    FunctionType func_type = func.getFunctionType();
    int num_inputs = func_type.getNumInputs();

    // Update the argument types of the function. If it's a tensorlist and
    // is not resized inside the function, we will use the corresponding
    // operand's type, otherwise change its type to unranked tensor type.
    SmallVector<Type, 8> updated_argument_types;
    updated_argument_types.reserve(num_inputs);
    UpdateTensorListTypes<mlir::OperandRange>(
        tensor_list_args, resized_tensor_lists, func_type.getInputs(),
        op.getOperands().drop_front(), operands.drop_front(),
        &updated_argument_types);

    UpdateFunctionAndRegionType(rewriter, func, updated_argument_types,
                                updated_result_types);
  }
  return success();
}

// Returns a `llvm::DenseMap` which maps from the index of tensorlist in the
// result, to the index of the same tensorlist in the arguments. For `If` op's
// branch functions, the results and arguments are not usually matched 1-1. This
// will let us konw which tensorlist result maps to which tensorlist in the
// arguments. Once we know this info it will help us decide the types of the
// result tensorlist based on the operand's of the `If` op.
llvm::DenseMap<int, int> MapTensorListResultToArgument(FuncOp func) {
  // `map_fn` will trace upwards along the use-def chain of the ssa value. It
  // starts from the last ssa value (returned by the function), and check its
  // parent op iteratively. If the root ssa value appears in the function's
  // argument list, it will return the index of the corresponding argument,
  // otherwise it will return -1.
  auto map_fn = [](Value value) -> int {
    Value parent = value;
    while (true) {
      if (auto identity = parent.getDefiningOp<TF::IdentityOp>()) {
        parent = identity.input();
      } else if (auto set_item =
                     parent.getDefiningOp<TF::TensorListSetItemOp>()) {
        parent = set_item.input_handle();
      } else {
        break;
      }
    }
    if (auto block_arg = parent.dyn_cast<mlir::BlockArgument>()) {
      return block_arg.getArgNumber();
    }
    // Returns -1 if we don't find which this result maps to.
    return -1;
  };

  llvm::SmallVector<Value, 4> returns;
  for (auto res : func.getBody().back().getTerminator()->getOperands()) {
    returns.push_back(res);
  }
  llvm::DenseMap<int, int> result;
  for (const auto &result_and_idx : llvm::enumerate(returns)) {
    if (IsTensorListType(result_and_idx.value().getType(),
                         result_and_idx.value())) {
      int arg_idx = map_fn(result_and_idx.value());
      if (arg_idx != -1) {
        result.insert({result_and_idx.index(), arg_idx});
      }
    }
  }
  return result;
}

// Updates the tensorlist result types for the `If` Op. If the tensorlist result
// maps to a specific argument (indicated by `tensor_list_map`), and also that
// tensorlist argument's shape isn't changed (indicated by
// `resized_tensor_list_index`), we will update this tensorlist's result type to
// the corresponding operand's type. In all other cases we change the
// tensorlist's type to unranked tensor type.
template <typename R>
void UpdateTensorListResultTypesForIf(
    const llvm::SmallSet<int, 4> &tensor_list_index,
    const llvm::SmallSet<int, 4> &resized_tensor_list_index,
    const llvm::DenseMap<int, int> &tensor_list_map, ArrayRef<Type> types,
    R &&range, ValueRange operands,
    llvm::SmallVectorImpl<Type> *updated_types) {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_38(mht_38_v, 1648, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "UpdateTensorListResultTypesForIf");

  int i = 0;
  for (const auto it : llvm::zip(types, range)) {
    if (!tensor_list_index.count(i)) {
      updated_types->push_back(std::get<0>(it));
      ++i;
      continue;
    }
    auto iter = tensor_list_map.find(i);
    if (iter != tensor_list_map.end()) {
      int arg_idx = iter->second;
      if (!resized_tensor_list_index.count(arg_idx)) {
        // If the mapped tensorlist argument's size isn't changed, we will
        // use the corresponding `operand` type.
        updated_types->push_back(operands[arg_idx].getType());
        ++i;
        continue;
      }
    }
    updated_types->push_back(
        VariantToUnrankedTensorType(std::get<0>(it), std::get<1>(it)));
    ++i;
  }
}

struct ConvertIf : public OpConversionPattern<TF::IfOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TF::IfOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_39(mht_39_v, 1681, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "matchAndRewrite");

    // Find all Tensor List arugments.
    auto tensor_list_args = GetTensorListArgumentsIndex(op.else_function());
    auto tensor_list_results = GetTensorListResultsIndex(op.else_function());
    auto tensor_list_map = MapTensorListResultToArgument(op.else_function());
    llvm::SmallSet<int, 4> resized_tensor_lists =
        GetResizedTensorListIndexes(op.else_function(), tensor_list_args);

    llvm::SmallVector<Type, 8> result_types;
    result_types.reserve(op.getNumResults());
    llvm::SmallVector<Type, 4> op_result_types;
    for (Type ty : op.getResultTypes()) {
      op_result_types.push_back(ty);
    }

    UpdateTensorListResultTypesForIf<mlir::ResultRange>(
        tensor_list_results, resized_tensor_lists, tensor_list_map,
        op_result_types, op->getResults(), adaptor.getOperands().drop_front(),
        &result_types);

    // Create a new if op with new operands and updated result types.
    auto converted = rewriter.create<TF::IfOp>(
        op.getLoc(), result_types, adaptor.getOperands(), op->getAttrs());
    converted->removeAttr("T");
    (void)UpdateFunctionTypesForIfOp(rewriter, converted, adaptor.getOperands(),
                                     tensor_list_args, resized_tensor_lists,
                                     result_types);
    rewriter.replaceOp(op, converted.getResults());
    return success();
  }
};

struct ConvertWhile : public OpConversionPattern<TF::WhileOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TF::WhileOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_40(mht_40_v, 1721, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "matchAndRewrite");

    // Find all Tensor List arugments.
    auto tensor_list_args = GetTensorListArgumentsIndex(op.body_function());

    llvm::SmallVector<Type, 8> result_types;
    result_types.reserve(op.getNumOperands());
    // Change all DT_VARIANT result types to unranked tensor type.
    llvm::SmallVector<Type, 4> op_result_types;
    for (Type ty : op.getResultTypes()) {
      op_result_types.push_back(ty);
    }

    llvm::SmallSet<int, 4> resized_tensor_lists =
        GetResizedTensorListIndexes(op.body_function(), tensor_list_args);
    UpdateTensorListTypes<mlir::OperandRange>(
        tensor_list_args, resized_tensor_lists, op_result_types,
        op.getOperands(), adaptor.getOperands(), &result_types);

    // Create a new while op with new operands and updated result types.
    auto converted = rewriter.create<TF::WhileOp>(
        op.getLoc(), result_types, adaptor.getOperands(), op->getAttrs());
    converted->removeAttr("T");
    (void)UpdateFunctionTypesForWhileOp(rewriter, converted,
                                        adaptor.getOperands(), tensor_list_args,
                                        resized_tensor_lists);

    rewriter.replaceOp(op, converted.getResults());
    return success();
  }
};

struct ConvertWhileRegion : public OpConversionPattern<TF::WhileRegionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      TF::WhileRegionOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_41(mht_41_v, 1760, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "matchAndRewrite");

    llvm::SmallVector<Type, 8> result_types;
    result_types.reserve(op.getNumOperands());
    // Change all DT_VARIANT result types to unranked tensor type.
    for (auto it : llvm::zip(op.getResultTypes(), adaptor.getOperands()))
      result_types.push_back(
          VariantToUnrankedTensorType(std::get<0>(it), std::get<1>(it)));

    // Create a new while op with new operands and updated result types.
    auto converted = rewriter.create<TF::WhileRegionOp>(
        op.getLoc(), result_types, adaptor.getOperands(), op->getAttrs());

    // Inline the regions from the old while into the new one, and apply
    // signature conversion to inlined region.
    for (auto it : llvm::zip(op.getRegions(), converted.getRegions())) {
      Region &old_region = *std::get<0>(it);
      Region &new_region = *std::get<1>(it);

      Block &entry = old_region.front();
      // Build signature conversion for the region.
      TypeConverter::SignatureConversion signature_conversion(
          adaptor.getOperands().size());
      for (auto it : llvm::zip(entry.getArguments(), adaptor.getOperands())) {
        BlockArgument arg = std::get<0>(it);
        signature_conversion.addInputs(
            arg.getArgNumber(),
            VariantToUnrankedTensorType(arg.getType(), std::get<1>(it)));
      }

      rewriter.inlineRegionBefore(old_region, new_region, new_region.end());
      rewriter.applySignatureConversion(&new_region, signature_conversion);
    }

    rewriter.replaceOp(op, converted.getResults());
    return success();
  }
};

#include "tensorflow/compiler/mlir/lite/transforms/generated_lower_static_tensor_list.inc"

void LowerStaticTensorListPass::runOnOperation() {
  auto *context = &getContext();

  // TensorFlow operations that doesn't have operands and results of type
  // variant are legal. Here, we don't distinguish between variants encoding
  // TensorList or some other type as that information is not available here.
  // Partial legalization is used below to still allow ops with variant types
  // still.
  auto is_legal = [](Operation *op) {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_42(mht_42_v, 1811, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "lambda");

    auto is_not_variant = [](Type ty) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_43(mht_43_v, 1815, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "lambda");

      return !ty.cast<ShapedType>().getElementType().isa<TF::VariantType>();
    };
    return llvm::all_of(op->getOperandTypes(), is_not_variant) &&
           llvm::all_of(op->getResultTypes(), is_not_variant);
  };

  ConversionTarget target(*context);
  target.addDynamicallyLegalDialect<TF::TensorFlowDialect>(is_legal);
  target.addIllegalOp<TF::EmptyTensorListOp, TF::TensorListFromTensorOp,
                      TF::TensorListGetItemOp, TF::TensorListLengthOp,
                      TF::TensorListPushBackOp, TF::TensorListReserveOp,
                      TF::TensorListSetItemOp, TF::TensorListStackOp,
                      TF::TensorListResizeOp, TF::TensorListConcatV2Op>();
  // TODO(hinsu): Use TFLite constant op for constants.
  target.addLegalOp<arith::ConstantOp>();
  target.addLegalOp<FuncOp>();
  target.addDynamicallyLegalOp<func::ReturnOp>(is_legal);
  target.addDynamicallyLegalOp<TF::YieldOp>(is_legal);
  target.addLegalOp<TFL::CustomOp>();
  // Register fused LSTM/RNN ops as legal.
  target.addLegalOp<TFL::LSTMOp>();
  target.addLegalOp<TFL::UnidirectionalSequenceLSTMOp>();
  target.addLegalOp<TFL::UnidirectionalSequenceRNNOp>();
  target.addLegalOp<TFL::BidirectionalSequenceLSTMOp>();

  RewritePatternSet patterns(&getContext());
  populateWithGenerated(patterns);
  patterns.add<ConvertConst, ConvertIdentity, ConvertTensorListGetItem,
               ConvertTensorListLength, ConvertTensorListPushBack,
               ConvertTensorListStack, ConvertTensorListResize, ConvertWhile,
               ConvertWhileRegion, ConvertIf, ConvertReturn, ConvertYield>(
      context);
  patterns.add<ConvertTensorListSetItem>(context, enable_dynamic_update_slice);
  patterns.add<ConvertEmptyTensorList, ConvertTensorListConcatV2,
               ConvertTensorListReserve>(context, allow_tensorlist_pass_through,
                                         default_to_single_batch);
  ModuleOp module = getOperation();
  if (!allow_tensorlist_pass_through) {
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      module.emitError(
          "Lowering tensor list ops is failed. Please consider using Select TF "
          "ops and disabling `_experimental_lower_tensor_list_ops` flag in the "
          "TFLite converter object. For example, "
          "converter.target_spec.supported_ops = "
          "[tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]\\n "
          "converter._experimental_lower_tensor_list_ops = False");
      signalPassFailure();
    }
  } else {
    // If `allow_tensorlist_pass_through` is set to true, if legalization fails
    // we should not leak the diagnostic info outside this pass. Hence we use
    // a `StatusScopedDiagnosticHandler` here to capture diagnostics generated
    // within this pass.
    StatusScopedDiagnosticHandler handler(context);
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      auto _ = handler.ConsumeStatus();
    }
  }
}

}  // namespace

/// Creates an instance of the TensorFlow Lite dialect LowerStaticTensorList
/// pass.
std::unique_ptr<OperationPass<ModuleOp>> TFL::CreateLowerStaticTensorListPass(
    bool allow_tensorlist_pass_through, bool default_to_single_batch,
    bool enable_dynamic_update_slice) {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSlower_static_tensor_listDTcc mht_44(mht_44_v, 1885, "", "./tensorflow/compiler/mlir/lite/transforms/lower_static_tensor_list.cc", "TFL::CreateLowerStaticTensorListPass");

  return std::make_unique<LowerStaticTensorListPass>(
      allow_tensorlist_pass_through, default_to_single_batch,
      enable_dynamic_update_slice);
}

static PassRegistration<LowerStaticTensorListPass> pass;

}  // namespace mlir
