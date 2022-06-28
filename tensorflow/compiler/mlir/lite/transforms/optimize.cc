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
class MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc() {
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

// This transformation pass takes operations in TensorFlowLite dialect and
// optimizes them to resulting operations in TensorFlowLite dialect.

#include <algorithm>
#include <climits>
#include <cstdint>
#include <functional>
#include <iterator>
#include <map>
#include <numeric>
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Casting.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/Matchers.h"  // from @llvm-project
#include "mlir/IR/TypeUtilities.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/Support/LLVM.h"  // from @llvm-project
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/lite/ir/tfl_ops.h"
#include "tensorflow/compiler/mlir/lite/quantization/quantization_utils.h"
#include "tensorflow/compiler/mlir/lite/transforms/passes.h"
#include "tensorflow/compiler/mlir/lite/utils/attribute_utils.h"
#include "tensorflow/compiler/mlir/lite/utils/convert_type.h"
#include "tensorflow/compiler/mlir/lite/utils/validators.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"

namespace mlir {
namespace TFL {

//===----------------------------------------------------------------------===//
// The actual Optimize Pass.
namespace {
constexpr char kRelu[] = "RELU";
constexpr char kRelu6[] = "RELU6";
constexpr char kRelu1[] = "RELU_N1_TO_1";

bool L2NormalizeReduceAxis(Value sq_op, DenseElementsAttr axis) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_0(mht_0_v, 240, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "L2NormalizeReduceAxis");

  if (axis.getNumElements() == 0) {
    return false;
  }
  if (sq_op.getType().cast<ShapedType>().getRank() - 1 ==
          *axis.getValues<int>().begin() ||
      *axis.getValues<int>().begin() == -1) {
    return true;
  }
  if (sq_op.getType().cast<ShapedType>().getRank() != axis.getNumElements()) {
    return false;
  }
  auto shape = sq_op.getType().cast<ShapedType>();
  SmallVector<int, 4> elems{axis.getValues<int>().begin(),
                            axis.getValues<int>().end()};
  for (int i = 0; i < shape.getRank(); ++i) {
    if (i != elems[i]) return false;
  }
  return true;
}

using ::llvm::cast;

// Optimize TFLite operations in functions.
class OptimizePass : public PassWrapper<OptimizePass, OperationPass<FuncOp>> {
 public:
  OptimizePass() = default;
  OptimizePass(const OptimizePass &) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_1(mht_1_v, 270, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "OptimizePass");
}
  explicit OptimizePass(bool enable_canonicalization) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_2(mht_2_v, 274, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "OptimizePass");

    enable_canonicalization_ = enable_canonicalization;
  }

  StringRef getArgument() const final {
   std::vector<std::string> mht_3_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_3(mht_3_v, 281, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "getArgument");

    // This is the argument used to refer to the pass in
    // the textual format (on the commandline for example).
    return "tfl-optimize";
  }
  StringRef getDescription() const final {
   std::vector<std::string> mht_4_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_4(mht_4_v, 289, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "getDescription");

    // This is a brief description of the pass.
    return "Optimize within the TensorFlow Lite dialect";
  }

  void runOnOperation() override;

 private:
  Option<bool> enable_canonicalization_{
      *this, "enable-canonicalization",
      llvm::cl::desc("Enable canonicalization during optimization pass."),
      llvm::cl::init(false)};
};

// Returns whether the given type `a` is broadcast-compatible with `b`.
bool IsBroadcastableElementsAttrAndType(Type a, Type b) {
   std::vector<std::string> mht_5_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_5(mht_5_v, 307, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "IsBroadcastableElementsAttrAndType");

  return OpTrait::util::getBroadcastedType(a, b) != Type();
}

// Returns whether the resultant type of any broadcastable operation with
// operands `a` and `b` matches `expected_output`. Returns false if `a` is not
// broadcast-compatible with `b`.
bool OperandsBroadcastToOutputType(Type a, Type b, Type expected_output) {
   std::vector<std::string> mht_6_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_6(mht_6_v, 317, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "OperandsBroadcastToOutputType");

  Type output_element_type =
      expected_output.cast<ShapedType>().getElementType();
  Type broadcasted_type =
      OpTrait::util::getBroadcastedType(a, b, output_element_type);
  return broadcasted_type != Type() && broadcasted_type == expected_output;
}

// Returns whether if `type1` dimensions are the same as the ending dimensions
// of `type2`. This is more restricted than broadcastable.
bool IsTailOfShape(Type type1, Type type2) {
   std::vector<std::string> mht_7_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_7(mht_7_v, 330, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "IsTailOfShape");

  auto tail_type = type1.dyn_cast<ShapedType>();
  auto full_type = type2.dyn_cast<ShapedType>();
  if (!tail_type || !full_type || !tail_type.hasRank() ||
      !full_type.hasRank() || tail_type.getRank() > full_type.getRank())
    return false;
  auto i1 = tail_type.getShape().rbegin(), e1 = tail_type.getShape().rend();
  auto i2 = full_type.getShape().rbegin();
  return std::equal(i1, e1, i2);
}

bool CanFuseConvOrDepthwiseConvShapes(const ArrayRef<int64_t> filter_shape,
                                      const ArrayRef<int64_t> elements_shape,
                                      bool is_depthwise) {
   std::vector<std::string> mht_8_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_8(mht_8_v, 346, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "CanFuseConvOrDepthwiseConvShapes");

  // Also, val tensor must be of rank 1 or 0 (scalar).
  const auto elements_rank = elements_shape.size();
  if (elements_rank != 1 && elements_rank != 0) {
    return false;
  }
  auto elements_depth = elements_shape.empty() ? 1 : elements_shape.back();
  // If elements depth equals 1 (i.e., scalar or tensor with 1 element), then we
  // can let binary op to broadcast elements.
  if (elements_depth == 1) {
    return true;
  }

  // In TFLite Conv2D uses OHWI format for filter, and 1HWO for Depthwise Conv.
  // For conv:
  // Check if last dimension in filter equals the first dimension
  // For depthwise conv:
  // Check if the first in filter dimension equals the first dimension.
  if (filter_shape.empty() ||
      (is_depthwise ? filter_shape.back() != elements_depth
                    : filter_shape[0] != elements_depth))
    return false;
  return true;
}

bool CanFuseConvOrDepthwiseConv(Value filter, Attribute val,
                                bool is_depthwise) {
   std::vector<std::string> mht_9_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_9(mht_9_v, 375, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "CanFuseConvOrDepthwiseConv");

  const auto elements = val.dyn_cast<DenseElementsAttr>();
  if (!elements) {
    return false;
  }
  const auto elements_shape = elements.getType().getShape();
  const auto filter_shape = filter.getType().cast<ShapedType>().getShape();
  return CanFuseConvOrDepthwiseConvShapes(filter_shape, elements_shape,
                                          is_depthwise);
}

bool CanFuseConvOrDepthwiseConv(Attribute filter, Attribute val,
                                bool is_depthwise) {
   std::vector<std::string> mht_10_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_10(mht_10_v, 390, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "CanFuseConvOrDepthwiseConv");

  if (const auto elements = val.dyn_cast<DenseElementsAttr>()) {
    if (const auto filter_elements = filter.dyn_cast<DenseElementsAttr>()) {
      return CanFuseConvOrDepthwiseConvShapes(
          filter_elements.getType().getShape(), elements.getType().getShape(),
          is_depthwise);
    }
  }
  return false;
}

// Retuns true if we can eliminate the GatherNdOp or ScatterNdOp. When the value
// of `indices` are from 0 to n-1, the output tensor are identical to the
// `params`.
bool CanOptimizeIdentityGatherNdOrScatterNdOp(Value params,
                                              DenseIntElementsAttr indices,
                                              Type output_type) {
   std::vector<std::string> mht_11_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_11(mht_11_v, 409, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "CanOptimizeIdentityGatherNdOrScatterNdOp");

  auto params_type = params.getType().dyn_cast<RankedTensorType>();
  auto indices_type = indices.getType().dyn_cast<RankedTensorType>();
  // Checks the shape of `params` is [n, ...], shape of `indices` is [n, 1]. 2D
  // `indices` means it gets the first row of `params`. As long as indices
  // iterate the first row of `params`, the output is identical to input.
  if (!params_type || !indices_type || indices_type.getRank() != 2 ||
      indices_type.getDimSize(0) != params_type.getDimSize(0) ||
      indices_type.getDimSize(1) != 1)
    return false;

  // Checks the `params_type` is equal to `output_type`. If not equal, we
  // cannot replace the scatter_nd/gather_nd op with `params`.
  if (params_type != output_type) return false;

  // Checks the value in `indices` is from 0 to n-1.
  int cur_value = 0;
  for (const auto &v : indices.getValues<APInt>()) {
    if (v.getSExtValue() != cur_value) return false;
    ++cur_value;
  }

  return true;
}

// Returns true if we can eliminate the SliceOp. When the values of `begin` are
// all 0s and `size[i]` is equal to either -1 or `input.shape[i]`
// for each dim i, the output tensor is identical to `input`.
bool CanOptimizeIdentitySliceOp(Value input, Attribute begin, Attribute size) {
   std::vector<std::string> mht_12_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_12(mht_12_v, 440, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "CanOptimizeIdentitySliceOp");

  // Checks if `begin` and `size` are i32 or i64.
  auto begin_attr = begin.dyn_cast<DenseIntElementsAttr>();
  auto size_attr = size.dyn_cast<DenseIntElementsAttr>();
  if (!begin_attr || !size_attr) {
    return false;
  }

  auto begin_elem_ty = begin_attr.getType().getElementType();
  if (!begin_elem_ty.isInteger(32) && !begin_elem_ty.isInteger(64)) {
    return false;
  }
  auto size_elem_ty = size_attr.getType().getElementType();
  if (!size_elem_ty.isInteger(32) && !size_elem_ty.isInteger(64)) {
    return false;
  }

  // Checks if `input` is ranked and its rank is equal to number of elements in
  // `begin` and `size`.
  auto input_ty = input.getType().cast<ShapedType>();
  if (!input_ty.hasRank()) {
    return false;
  }

  int64_t rank = input_ty.getRank();
  if (rank != begin_attr.getNumElements() ||
      rank != size_attr.getNumElements()) {
    return false;
  }

  // Checks if `begin` is all 0s, and `size[i]` is equal to either -1 or
  // `input.shape[i]`.
  for (uint64_t i = 0; i < rank; ++i) {
    if (begin_attr.getValues<APInt>()[i].getSExtValue() != 0) return false;
    int64_t si = size_attr.getValues<APInt>()[i].getSExtValue();
    if (si != -1 && si != input_ty.getDimSize(i)) return false;
  }

  return true;
}

// Expand Attribute 'a' to 4D with all 1s except 1 dimension.
// Which dimension depends on 'is_depthwise' is true or false.
ElementsAttr ExpandTo4DForConvImpl(Attribute a, bool is_depthwise) {
   std::vector<std::string> mht_13_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_13(mht_13_v, 486, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "ExpandTo4DForConvImpl");

  auto elements = a.dyn_cast<DenseElementsAttr>();
  auto shape = elements.getType().getShape();
  if (!shape.empty()) {
    // Checks that elements are essentially 1d.
    assert(elements.getNumElements() == shape.back());
  }
  std::vector<int64_t> shape_data = {1, 1, 1, 1};
  const int vector_length = elements.getNumElements();
  if (is_depthwise)
    shape_data[3] = vector_length;
  else
    shape_data[0] = vector_length;
  auto new_shape =
      RankedTensorType::get(shape_data, elements.getType().getElementType());
  return elements.reshape(new_shape);
}

ElementsAttr ExpandTo4DForConv(Attribute a) {
   std::vector<std::string> mht_14_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_14(mht_14_v, 507, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "ExpandTo4DForConv");

  return ExpandTo4DForConvImpl(a, false);
}

ElementsAttr ExpandTo4DForDepthwiseConv(Attribute a) {
   std::vector<std::string> mht_15_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_15(mht_15_v, 514, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "ExpandTo4DForDepthwiseConv");

  return ExpandTo4DForConvImpl(a, true);
}

TypeAttr RescaleQtype(Type input, Attribute factor) {
   std::vector<std::string> mht_16_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_16(mht_16_v, 521, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "RescaleQtype");

  return quant::RescaleQuantizedType(input, factor);
}

// Returns shape of a ranked tensor.
// Precondition: output_val's is ranked tensor.
DenseElementsAttr GetShape(Value output_val) {
   std::vector<std::string> mht_17_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_17(mht_17_v, 530, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "GetShape");

  auto output_type = output_val.getType().cast<RankedTensorType>();
  auto shape_vector = output_type.getShape();
  std::vector<int32_t> shape;
  shape.reserve(shape_vector.size());
  for (auto shape_object : shape_vector) {
    shape.push_back(shape_object);
  }
  return mlir::DenseElementsAttr::get(
      RankedTensorType::get(
          {static_cast<int>(shape.size())},
          mlir::IntegerType::get(output_val.getContext(), 32)),
      llvm::makeArrayRef(shape));
}

static Type GetShapeStrippedType(TypeAttr type_attr) {
   std::vector<std::string> mht_18_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_18(mht_18_v, 548, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "GetShapeStrippedType");

  auto type = type_attr.getValue();
  auto shaped_type = type.dyn_cast<ShapedType>();
  if (shaped_type) {
    return shaped_type.getElementType();
  } else {
    return type;
  }
}

// Returns `true` if reducing `axes` in `input` with `keep_dims=true` results in
// the specified `shape` and `false` otherwise.
static bool ShapeMatchesReduceWithKeepAxes(Value input,
                                           const mlir::Attribute &axes,
                                           const mlir::Attribute &shape) {
   std::vector<std::string> mht_19_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_19(mht_19_v, 565, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "ShapeMatchesReduceWithKeepAxes");

  RankedTensorType type = input.getType().dyn_cast_or_null<RankedTensorType>();
  if (!type) return false;

  DenseIntElementsAttr axes_attr =
      axes.dyn_cast_or_null<DenseIntElementsAttr>();
  DenseIntElementsAttr shape_attr =
      shape.dyn_cast_or_null<DenseIntElementsAttr>();
  if (!axes_attr || !shape_attr) return false;

  if (shape_attr.getNumElements() != type.getRank()) return false;

  llvm::SmallSet<uint64_t, 4> axes_set;
  for (auto a : axes_attr.getValues<APInt>()) {
    axes_set.insert(a.getZExtValue());
  }

  auto type_shape = type.getShape();
  for (uint64_t i = 0; i < type.getRank(); ++i) {
    if (axes_set.contains(i)) {
      if (shape_attr.getValues<APInt>()[i] != 1) return false;
    } else {
      if (shape_attr.getValues<APInt>()[i] != type_shape[i]) return false;
    }
  }
  return true;
}

// Returns `true` if all the `axes` dimensions of `input` are 1.
static bool AreInputDimensionsOneInAxes(Value input,
                                        const mlir::Attribute &axes) {
   std::vector<std::string> mht_20_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_20(mht_20_v, 598, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "AreInputDimensionsOneInAxes");

  RankedTensorType input_type =
      input.getType().dyn_cast_or_null<RankedTensorType>();
  if (!input_type) return false;
  auto type_shape = input_type.getShape();

  DenseIntElementsAttr axes_attr =
      axes.dyn_cast_or_null<DenseIntElementsAttr>();
  if (!axes_attr) return false;

  for (auto a : axes_attr.getValues<APInt>()) {
    int64_t axis = a.getSExtValue();
    if (axis < 0) {
      axis += type_shape.size();
    }
    if (axis < 0 || axis >= type_shape.size()) {
      // `axis` is not a valid axis in input.
      return false;
    }
    if (type_shape[axis] != 1) {
      return false;
    }
  }

  return true;
}

static bool FloatValueEquals(const Attribute &attr, double value) {
   std::vector<std::string> mht_21_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_21(mht_21_v, 628, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "FloatValueEquals");

  auto fp_attr = attr.dyn_cast_or_null<DenseFPElementsAttr>();
  if (!fp_attr) return false;

  if (fp_attr.isSplat()) {
    return fp_attr.getSplatValue<APFloat>().isExactlyValue(value);
  }
  return llvm::all_of(fp_attr.getValues<APFloat>(), [value](const APFloat &f) {
    return f.isExactlyValue(value);
  });
}

// Returns true if the value's element type is F32.
bool IsF32Value(Value value) {
   std::vector<std::string> mht_22_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_22(mht_22_v, 644, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "IsF32Value");

  return value.getType().cast<ShapedType>().getElementType().isF32();
}

// Returns the number of elements in attr if it is a DenseElementsAttr, 1
// otherwise, as an unranked int32 Attribute.
Attribute GetNumElementsOrOne(Attribute attr) {
   std::vector<std::string> mht_23_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_23(mht_23_v, 653, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "GetNumElementsOrOne");

  const auto dense_attr = attr.dyn_cast_or_null<DenseElementsAttr>();
  int32_t num_elements = dense_attr ? dense_attr.getNumElements() : 1;

  OpBuilder builder(attr.getContext());

  return DenseIntElementsAttr::get(
      RankedTensorType::get({}, builder.getI32Type()),
      {llvm::APInt(32, num_elements, true)});
}

bool HasExactlyTwoElements(Attribute attr) {
   std::vector<std::string> mht_24_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_24(mht_24_v, 667, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "HasExactlyTwoElements");

  const auto values = attr.dyn_cast_or_null<ElementsAttr>();
  if (!values) return false;
  return values.getNumElements() == 2;
}

// Returns true if attr is a DenseIntElementsAttr with the last element equal 1.
bool IsLastElementEqualsOne(Attribute attr) {
   std::vector<std::string> mht_25_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_25(mht_25_v, 677, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "IsLastElementEqualsOne");

  const auto ints = attr.dyn_cast_or_null<DenseIntElementsAttr>();
  if (!ints) return false;
  if (ints.empty()) return false;
  const auto last_element_index = ints.getNumElements() - 1;
  const auto iterator = ints.value_begin<int>();
  const int last_element = iterator[last_element_index];
  return last_element == 1;
}

// Reshapes value to a given shape.
Value ReshapeValueDroppingLastDim(OpBuilder &builder, Value value,
                                  Attribute shape) {
   std::vector<std::string> mht_26_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_26(mht_26_v, 692, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "ReshapeValueDroppingLastDim");

  // This function is always guarded with IsLastElementEqualsOne(), so we could
  // cast safely here.
  const auto old_shape = shape.cast<DenseIntElementsAttr>();
  auto iterator = old_shape.value_begin<int>();
  SmallVector<int, 4> new_shape;
  SmallVector<int64_t, 4> new_shape_i64;
  for (int i = 0; i < old_shape.size() - 1; ++i) {
    new_shape.push_back(*iterator);
    new_shape_i64.push_back(*iterator);
    ++iterator;
  }
  return builder.create<ReshapeOp>(
      value.getLoc(),
      RankedTensorType::get(
          new_shape_i64, value.getType().cast<ShapedType>().getElementType()),
      value,
      builder.create<arith::ConstantOp>(
          value.getLoc(), DenseIntElementsAttr::get(
                              RankedTensorType::get({old_shape.size() - 1},
                                                    builder.getI32Type()),
                              new_shape)));
}

// Returns true if val has a static shape and the last dimension equals 1.
bool IsLastDimensionEqualOne(Value val) {
   std::vector<std::string> mht_27_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_27(mht_27_v, 720, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "IsLastDimensionEqualOne");

  const auto val_type = val.getType().cast<ShapedType>();
  if (!val_type.hasStaticShape()) return false;
  const auto val_shape = val_type.getShape();
  if (val_shape.empty()) return false;
  const auto last_element = *val_shape.rbegin();
  return last_element == 1;
}

// Returns true if attr is a DenseIntElementsAttr of int32 or int64 values or an
// incrementing sequence from 0 to N-1.
//
// If such a value is used in an Equal operator, it can be replaced with OneHot.
bool IsOneHotIndexAttribute(Attribute attr) {
   std::vector<std::string> mht_28_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_28(mht_28_v, 736, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "IsOneHotIndexAttribute");

  const auto dense_attr = attr.dyn_cast_or_null<DenseIntElementsAttr>();
  if (!dense_attr) {
    return false;
  }
  auto index_type = dense_attr.getType();
  const auto index_elem_bits = index_type.getElementTypeBitWidth();
  if (index_elem_bits != 32 && index_elem_bits != 64) {
    return false;
  }
  if (index_type.getRank() != 1) {
    return false;
  }
  const auto elems = dense_attr.value_begin<APInt>();
  for (int i = 0; i < dense_attr.getNumElements(); ++i) {
    if (i != elems[i]) {
      return false;
    }
  }
  return true;
}

// Creates FullyConnected op from params and returns the output.
mlir::Value GetFcOutput(OpBuilder *builder,
                        ::mlir::Operation::result_range result, Value input,
                        Value filter, Value bias,
                        StringAttr fused_activation_function,
                        StringAttr weights_format, BoolAttr keep_num_dims,
                        BoolAttr asymmetric_quantize_inputs) {
   std::vector<std::string> mht_29_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_29(mht_29_v, 767, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "GetFcOutput");

  auto fc_op = builder->create<FullyConnectedOp>(
      result[0].getLoc(), result.getTypes(), input, filter, bias,
      fused_activation_function, weights_format, keep_num_dims,
      asymmetric_quantize_inputs);
  return fc_op->getResult(0);
}

// Returns true if 'value' represents a const ElementsAttr with all values
// equals to 0.0.
bool AllValuesAreZero(mlir::Value value) {
   std::vector<std::string> mht_30_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_30(mht_30_v, 780, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "AllValuesAreZero");

  if (!value) return false;
  DenseElementsAttr vals;
  if (!matchPattern(value, m_Constant(&vals))) return false;
  for (auto elem : vals.getValues<float>())
    if (elem != 0.0f) return false;
  return true;
}

// Converts an Attribute with a single value of float or integral type to an
// Attribute holding a single value of float type. If attr has no elements, the
// result is 0.0f.
Attribute ConvertSingleElementAttrToFloatAttr(Attribute attr) {
   std::vector<std::string> mht_31_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_31(mht_31_v, 795, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "ConvertSingleElementAttrToFloatAttr");

  const auto dense_fp_attr = attr.dyn_cast_or_null<DenseFPElementsAttr>();
  if (dense_fp_attr) {
    // Already float => return
    return dense_fp_attr;
  }

  OpBuilder builder(attr.getContext());

  const auto dense_int_attr = attr.dyn_cast<DenseIntElementsAttr>();
  const auto int_values = dense_int_attr.getValues<APInt>();
  float float_val = 0.0f;
  if (!int_values.empty()) {
    const APInt apint_val = *int_values.begin();
    if (dense_int_attr.getType().getElementType().isSignedInteger()) {
      // Get the sign-extended value (=>int64) if the type is signed.
      float_val = apint_val.getSExtValue();
    } else {
      // Get the zero-extended value (=>uint64) if unsigned or signless.
      float_val = apint_val.getZExtValue();
    }
  }
  return DenseFPElementsAttr::get(
      RankedTensorType::get({}, builder.getF32Type()),
      {llvm::APFloat(float_val)});
}

#include "tensorflow/compiler/mlir/lite/transforms/generated_optimize.inc"

// Fuse Add with proceeding FullyConnected.
// TODO(b/136285429): Move to tablegen when variadic is supported
struct FuseFullyConnectedAndAdd : public OpRewritePattern<TFL::AddOp> {
  using OpRewritePattern<TFL::AddOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::AddOp add_op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_32_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_32(mht_32_v, 833, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "matchAndRewrite");

    // Match Add.
    DenseElementsAttr added_value;
    Value constant_val = add_op.rhs();
    if (!matchPattern(constant_val, m_Constant(&added_value))) return failure();

    // Match Fully Connected.
    auto fc_op =
        dyn_cast_or_null<TFL::FullyConnectedOp>(add_op.lhs().getDefiningOp());
    if (!fc_op) return failure();

    // Check if the constant RHS is either 0D (scalar), or a 1D with
    // `{num_channels}` shape.
    auto constant_val_type = constant_val.getType().cast<TensorType>();

    // In TFLite FullyConnect definition, bias must be a 1D tensor where
    // the number of elements is equal to the number of channels.
    // If it's not 1D or 0D (which can be broadcasted to 1D), reject the
    // matching.
    bool is_scalar_rhs = false;
    if (constant_val_type.getRank() == 0) {
      is_scalar_rhs = true;
    } else if (constant_val_type.getRank() != 1) {
      return failure();
    }

    Value filter = fc_op.filter();
    Value bias = fc_op.bias();
    ElementsAttr bias_value;
    const bool is_none_bias = bias.getType().isa<NoneType>();
    if (fc_op.fused_activation_function() != "NONE") return failure();

    if (!is_none_bias && !matchPattern(bias, m_Constant(&bias_value)))
      return failure();

    // Rewrite
    if (is_none_bias) {
      if (is_scalar_rhs) {
        // If the `constant_val` is scalar, we must the shape of filter
        // to properly broadcast the scalar to `{num_channels}` shape.

        // Get the number of channels if possible.
        auto filter_type = filter.getType().dyn_cast<RankedTensorType>();
        // Filter must be a `2D` tensor with `{num_channels, num_features}`
        // shape. The following check is rejecting unknown rank (-1).
        if (filter_type == nullptr || filter_type.getRank() != 2) {
          return failure();
        }
        int num_channels = filter_type.getShape()[0];

        // Create a zero tensor with shape {num_channels}, and the type need to
        // be the same as constant_val.
        // This is a way to gracefully handle scalar tensor. The Add will always
        // be constant-folded away regardless if `constant_val` is a scalar or
        // not.
        RankedTensorType type = RankedTensorType::get(
            {num_channels}, constant_val_type.getElementType());
        auto attr = rewriter.getZeroAttr(type);
        bias = rewriter.create<arith::ConstantOp>(add_op.getLoc(), type, attr);
        auto none_af = rewriter.getStringAttr("NONE");
        bias =
            rewriter.create<AddOp>(add_op.getLoc(), bias, constant_val, none_af)
                .output();
      } else {
        // If there no pre-existing bias and the `constant_val` is 1D, simply
        // use `constant_val` as bias.
        bias = constant_val;
      }
    } else {
      auto none_af = rewriter.getStringAttr("NONE");
      bias =
          rewriter.create<AddOp>(add_op.getLoc(), bias, constant_val, none_af)
              .output();
    }

    auto fc = rewriter.create<TFL::FullyConnectedOp>(
        FusedLoc::get(fc_op.getContext(), {fc_op.getLoc(), add_op.getLoc()}),
        add_op.getType(),
        /*input=*/fc_op.input(),
        /*filter=*/filter,
        /*bias=*/bias,
        /*fused_activation_function=*/
        rewriter.getStringAttr(add_op.fused_activation_function()),
        /*weights_format=*/rewriter.getStringAttr(fc_op.weights_format()),
        /*keep_num_dims=*/rewriter.getBoolAttr(fc_op.keep_num_dims()),
        /*asymmetric_quantize_inputs=*/fc_op.asymmetric_quantize_inputsAttr());
    rewriter.replaceOp(add_op, fc.output());

    return success();
  }
};

// Replace ..
// FC(Add(lhs, rhs), filter, bias)
// .. with ..
// FC(lhs, filter, FC(rhs, filter, bias))
// .. if rhs, filter, and bias are all constants.
// The second FC will be constant folded to a single vector.
// TODO(b/136285429): Move to tablegen when variadic is supported
struct FuseAddAndFullyConnected
    : public OpRewritePattern<TFL::FullyConnectedOp> {
  using OpRewritePattern<TFL::FullyConnectedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::FullyConnectedOp fc_op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_33_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_33(mht_33_v, 940, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "matchAndRewrite");

    // This only works with default format.
    if (fc_op.weights_format() != "DEFAULT") return failure();

    // Match Add.
    auto add_op = dyn_cast_or_null<TFL::AddOp>(fc_op.input().getDefiningOp());
    if (!add_op) return failure();
    if (add_op.fused_activation_function() != "NONE") return failure();

    // Don't match adds where the added constant is not 1D.
    {
      auto addend_shape = add_op.rhs().getType().cast<ShapedType>();
      if (!addend_shape.hasStaticShape()) return failure();
      if (addend_shape.getShape().size() != 1) return failure();
    }

    // Calculate new bias.  Generate a new FC; it will be constant folded.
    auto old_bias = fc_op.bias();
    if (!old_bias || old_bias.getType().isa<NoneType>()) {
      // TODO(b/180752069): Figure out new bias' type when old bias is empty.
      return failure();
    }

    // The FC relies on constant folding, which is implemented on F32. Checks
    // types to be F32.
    {
      if (!IsF32Value(add_op.rhs()) || !IsF32Value(fc_op.filter()) ||
          !IsF32Value(old_bias))
        return failure();
    }

    auto new_bias = rewriter.create<TFL::FullyConnectedOp>(
        fc_op.getLoc(), old_bias.getType(),
        /*input=*/add_op.rhs(),
        /*filter=*/fc_op.filter(),
        /*bias=*/old_bias,
        /*fused_activation_function=*/rewriter.getStringAttr("NONE"),
        /*weights_format=*/rewriter.getStringAttr("DEFAULT"),
        /*keep_num_dims=*/rewriter.getBoolAttr(true),
        /*asymmetric_quantize_inputs=*/fc_op.asymmetric_quantize_inputsAttr());

    // Create the updated FC.
    auto new_fc = rewriter.create<TFL::FullyConnectedOp>(
        FusedLoc::get(add_op.getContext(), {add_op.getLoc(), fc_op.getLoc()}),
        fc_op.output().getTypes(),
        /*input=*/add_op.lhs(),
        /*filter=*/fc_op.filter(),
        /*bias=*/*new_bias.output().begin(),
        /*fused_activation_function=*/
        rewriter.getStringAttr(fc_op.fused_activation_function()),
        /*weights_format=*/rewriter.getStringAttr("DEFAULT"),
        /*keep_num_dims=*/rewriter.getBoolAttr(fc_op.keep_num_dims()),
        /*asymmetric_quantize_inputs=*/fc_op.asymmetric_quantize_inputsAttr());
    rewriter.replaceOp(fc_op.getOperation(), new_fc.output());

    return success();
  }
};

// Replace ..
// FC(Mul(lhs, rhs), filter, bias)
// .. with ..
// FC(lhs, Mul(filter, rhs), bias)
// .. if rhs, filter, and bias are all constants.
// The generated Mul will be constant folded to a single matrix.
struct FuseMulAndFullyConnected
    : public OpRewritePattern<TFL::FullyConnectedOp> {
  using OpRewritePattern<TFL::FullyConnectedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::FullyConnectedOp fc_op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_34_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_34(mht_34_v, 1013, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "matchAndRewrite");

    // This only works with default format.
    if (fc_op.weights_format() != "DEFAULT") return failure();

    // Match Mul.
    auto mul_op = dyn_cast_or_null<TFL::MulOp>(fc_op.input().getDefiningOp());
    if (!mul_op) return failure();
    if (mul_op.fused_activation_function() != "NONE") return failure();

    // Don't match muls where the multiplier constant is not 1D.
    {
      auto multiplier_shape = mul_op.rhs().getType().cast<ShapedType>();
      if (!multiplier_shape.hasStaticShape()) return failure();
      if (multiplier_shape.getShape().size() != 1) return failure();
    }

    // We rely on constant folding, implemented only for F32. Check types.
    if (!IsF32Value(mul_op.rhs()) || !IsF32Value(fc_op.filter())) {
      return failure();
    }

    auto location =
        FusedLoc::get(mul_op.getContext(), {mul_op.getLoc(), fc_op.getLoc()});

    auto new_filter = rewriter.create<TFL::MulOp>(
        location,
        /*lhs=*/fc_op.filter(),
        /*rhs=*/mul_op.rhs(),
        /*fused_activation_function=*/rewriter.getStringAttr("NONE"));
    // Create the updated FC.
    auto new_fc = rewriter.create<TFL::FullyConnectedOp>(
        location, fc_op.output().getTypes(),
        /*input=*/mul_op.lhs(),
        /*filter=*/new_filter,
        /*bias=*/fc_op.bias(),
        /*fused_activation_function=*/
        rewriter.getStringAttr(fc_op.fused_activation_function()),
        /*weights_format=*/rewriter.getStringAttr("DEFAULT"),
        /*keep_num_dims=*/rewriter.getBoolAttr(fc_op.keep_num_dims()),
        /*asymmetric_quantize_inputs=*/fc_op.asymmetric_quantize_inputsAttr());
    rewriter.replaceOp(fc_op.getOperation(), new_fc.output());

    return success();
  }
};

// TODO(b/136285429): Move to tablegen when variadic is supported.
template <typename ReluXOp, char const *Act>
struct FuseFullyConnectedAndReluX : public OpRewritePattern<ReluXOp> {
  using OpRewritePattern<ReluXOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ReluXOp relu_op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_35_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_35(mht_35_v, 1068, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "matchAndRewrite");

    Operation *input = relu_op.getOperand().getDefiningOp();
    if (!isa_and_nonnull<FullyConnectedOp>(input)) return failure();
    auto fully_connected_op = cast<FullyConnectedOp>(input);
    if (fully_connected_op.fused_activation_function() != "NONE")
      return failure();

    auto new_activation_func = rewriter.getStringAttr(Act);
    auto new_weights_format =
        rewriter.getStringAttr(fully_connected_op.weights_format());
    auto new_keep_num_dims =
        rewriter.getBoolAttr(fully_connected_op.keep_num_dims());
    auto fc = rewriter.create<FullyConnectedOp>(
        FusedLoc::get(relu_op.getContext(),
                      {fully_connected_op.getLoc(), relu_op.getLoc()}),
        relu_op.getType(), /*input=*/fully_connected_op.input(),
        /*filter=*/fully_connected_op.filter(),
        /*bias=*/fully_connected_op.bias(),
        /*fused_activation_function=*/new_activation_func,
        /*weights_format=*/new_weights_format,
        /*keep_num_dims=*/new_keep_num_dims,
        /*asymmetric_quantize_inputs=*/
        fully_connected_op.asymmetric_quantize_inputsAttr());
    rewriter.replaceOp(relu_op, fc.output());

    return success();
  }
};

// Fuse Mul with proceeding FullyConnected.
// TODO(b/136285429): Move to tablegen when variadic is supported
struct FuseFullyConnectedAndMul : public OpRewritePattern<TFL::MulOp> {
  using OpRewritePattern<TFL::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::MulOp mul_op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_36_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_36(mht_36_v, 1106, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "matchAndRewrite");

    // If we are broadcasting on the lhs then don't fold the multiply as it
    // would increase the amount of compute done by the fully connected op.
    if (mul_op.lhs().getType() != mul_op.getType()) return failure();

    // Mul.
    DenseElementsAttr cst;
    Value constant_val = mul_op.rhs();
    if (!matchPattern(constant_val, m_Constant(&cst))) return failure();

    // Fully Connected.
    auto fc_op =
        dyn_cast_or_null<TFL::FullyConnectedOp>(mul_op.lhs().getDefiningOp());
    if (!fc_op) return failure();
    Value filter = fc_op.filter();
    Value bias = fc_op.bias();
    ElementsAttr cst_tmp;
    if (!matchPattern(filter, m_Constant(&cst_tmp))) return failure();
    if (!bias.getType().isa<NoneType>() &&
        !matchPattern(bias, m_Constant(&cst_tmp)))
      return failure();
    if (fc_op.fused_activation_function() != "NONE") return failure();

    // Only fuse multiplier if all dimensions other than the depth dimension
    // are equal to 1 since otherwise
    // `matmul(x, filter) * cst != matmul(x, filter * cst)`
    // even if `filter` and `cst` are be broadcastable.
    auto shape = cst.getType().getShape();
    if (!IsDimensionsDegenerateExceptLastOne(shape)) return failure();

    int64_t element_size = shape.empty() ? 1 : shape[shape.size() - 1];
    // Expand and transpose the multiplier since weights are using the
    // OHWI data format in TFLite.
    int64_t normalized_shape[2] = {element_size, 1};
    auto new_cst = cst.reshape(RankedTensorType::get(
        normalized_shape, cst.getType().getElementType()));
    Type new_type = new_cst.getType();
    if (!IsBroadcastableElementsAttrAndType(new_type, filter.getType())) {
      return failure();
    }

    auto new_op =
        rewriter.create<arith::ConstantOp>(mul_op.getLoc(), new_type, new_cst);
    Value new_const_val = new_op.getResult();

    // Rewrite. Since the folder of TFL::MulOp couldn't broadcast the operands,
    // TF::MulOp is used to fold the constant.
    // TODO(b/139192933): switch to the TFL constant folding
    auto new_filter =
        rewriter.create<TF::MulOp>(mul_op.getLoc(), filter, new_const_val).z();
    // If bias isn't None, it needs to be multiplied as well.
    if (!bias.getType().isa<NoneType>()) {
      bias =
          rewriter.create<TF::MulOp>(mul_op.getLoc(), bias, constant_val).z();
    }

    auto fc = rewriter.create<TFL::FullyConnectedOp>(
        FusedLoc::get(fc_op.getContext(), {fc_op.getLoc(), mul_op.getLoc()}),
        mul_op.getType(),
        /*input=*/fc_op.input(),
        /*filter=*/new_filter,
        /*bias=*/bias,
        /*fused_activation_function=*/
        rewriter.getStringAttr(mul_op.fused_activation_function()),
        /*weights_format=*/rewriter.getStringAttr(fc_op.weights_format()),
        /*keep_num_dims=*/rewriter.getBoolAttr(fc_op.keep_num_dims()),
        /*asymmetric_quantize_inputs=*/fc_op.asymmetric_quantize_inputsAttr());
    rewriter.replaceOp(mul_op, fc.output());

    return success();
  }
};

// Fuse Mul with proceeding Affine ops. This is an C++ implementation of the
// following table gen implementation, which doesn't derived the result type of
// the TFL_DequantizeOp.
// def : Pat<(TFL_MulOp (TFL_Conv2DOp:$conv_output $input,
//                          (TFL_DequantizeOp (TFL_QuantizeOp
//                              (Arith_ConstantOp F32ElementsAttr:$filter),
//                              $qtype)),
//                          (Arith_ConstantOp F32ElementsAttr:$bias),
//                          $h_factor, $w_factor, TFL_AF_None,
//                          $padding, $stride_h, $stride_w),
//                      (Arith_ConstantOp F32ElementsAttr:$value), $act_fn),
//           (TFL_Conv2DOp $input,
//                      (TFL_DequantizeOp (TFL_QuantizeOp
//                          (TFL_MulOp (Arith_ConstantOp $filter),
//                                     (Arith_ConstantOp (ExpandTo4DForConv
//                                     $value)),
//                                      TFL_AF_None),
//                          (RescaleQtype $qtype, $value))),
//                      (TFL_MulOp (Arith_ConstantOp $bias), (Arith_ConstantOp
//                      $value),
//                          TFL_AF_None),
//                      $h_factor, $w_factor, $act_fn,
//                      $padding, $stride_h, $stride_w),
//         [(CanFuseConvOrDepthwiseConv<"false"> $filter, $value),
//          (HasOneUse $conv_output),
//          (IsPerAxisQuantization $qtype), // per-axis quantization
//         ]>;
template <typename AffineOpType>
struct FuseAffinOpAndMulWithQDQs : public OpRewritePattern<TFL::MulOp> {
  using OpRewritePattern<TFL::MulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::MulOp mul_op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_37_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_37(mht_37_v, 1214, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "matchAndRewrite");

    // Mul. Required 1-D rhs for batch normalization.
    DenseElementsAttr gamma_cst;
    Value gamma = mul_op.rhs();
    if (!matchPattern(gamma, m_Constant(&gamma_cst))) return failure();
    if (gamma_cst.getType().getRank() != 1) return failure();

    // Affine op
    Operation *mul_op_lhs = mul_op.lhs().getDefiningOp();
    auto fc_op = dyn_cast_or_null<AffineOpType>(mul_op_lhs);
    if (!fc_op) return failure();
    Value filter = fc_op.filter();
    Value bias = fc_op.bias();

    // QDQs
    auto dq_op = dyn_cast_or_null<TFL::DequantizeOp>(filter.getDefiningOp());
    if (!dq_op) return failure();
    auto q_op =
        dyn_cast_or_null<TFL::QuantizeOp>(dq_op.input().getDefiningOp());
    if (!q_op) return failure();
    filter = q_op.input();

    // weight constant
    ElementsAttr cst_tmp;
    if (!matchPattern(filter, m_Constant(&cst_tmp))) return failure();
    if (!bias.getType().isa<NoneType>() &&
        !matchPattern(bias, m_Constant(&cst_tmp)))
      return failure();
    if (fc_op.fused_activation_function() != "NONE") return failure();

    // Broadcast the constant operand of Mul if it isn't compatible to the
    // filter input. We only support broadcasting the operand along the depth
    // dimension, when the operand's depth is 1.
    rewriter.setInsertionPoint(q_op);
    Location loc = fc_op.getLoc();
    Value broadcasted_gamma;
    if (isa<TFL::Conv2DOp>(mul_op_lhs)) {
      auto mul_rhs = ExpandTo4DForConv(gamma_cst);
      broadcasted_gamma = rewriter.create<ConstOp>(loc, mul_rhs);
    } else if (isa<TFL::DepthwiseConv2DOp>(mul_op_lhs)) {
      auto mul_rhs = ExpandTo4DForDepthwiseConv(gamma_cst);
      broadcasted_gamma = rewriter.create<ConstOp>(loc, mul_rhs);
    } else {
      return failure();
    }

    // Make sure that the fused bias will be a 1D tensor.
    auto gamma_shape = gamma.getType().cast<ShapedType>();
    if (!gamma_shape.hasRank() || gamma_shape.getRank() != 1) {
      return failure();
    }

    // Rewrite filter constant. Since the folder of TFL::MulOp couldn't
    // broadcast the operands, TF::MulOp is used to fold the constant.
    auto new_filter =
        rewriter.create<TF::MulOp>(loc, filter, broadcasted_gamma).z();
    // Update the scale in the quantize op.
    auto new_qtype = RescaleQtype(q_op.qtype(), gamma_cst);
    if (!new_qtype) return failure();
    rewriter.replaceOpWithNewOp<TFL::QuantizeOp>(q_op, new_qtype.getValue(),
                                                 new_filter, new_qtype);

    // If bias isn't None, it needs to be multiplied as well.
    if (!bias.getType().isa<NoneType>()) {
      rewriter.setInsertionPoint(fc_op);
      auto new_bias = rewriter.create<TF::MulOp>(loc, bias, gamma);
      fc_op.getOperation()->replaceUsesOfWith(bias, new_bias);
    }

    // Remove the tailing mul op.
    mul_op.replaceAllUsesWith(fc_op.getResult());
    return success();
  }
};

using FuseConv2DAndMulWithQDQs = FuseAffinOpAndMulWithQDQs<TFL::Conv2DOp>;
using FuseDepthwiseConv2DAndMulWithQDQs =
    FuseAffinOpAndMulWithQDQs<TFL::DepthwiseConv2DOp>;

// Fuse Binary Op with following Affine operation.
template <typename AffineOpType>
struct FuseBinaryOpToFollowingAffineOp : public OpRewritePattern<AffineOpType> {
  using OpRewritePattern<AffineOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineOpType fc_op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_38_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_38(mht_38_v, 1302, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "matchAndRewrite");

    // Binary op.
    Operation *binary_op = fc_op.input().getDefiningOp();
    if (!binary_op || binary_op->getNumOperands() != 2) return failure();
    // We only handle the cases the RHS is a scalar.
    // TODO(fengliuai): Currently the canonicalizer pass couldn't guarantee that
    // the constant operands are on the RHS, we need to consider LHS constant
    // operand if necessary.
    DenseFPElementsAttr cst;
    if (!matchPattern(binary_op->getOperand(1), m_Constant(&cst)))
      return failure();
    if (cst.getNumElements() != 1) return failure();
    APFloat cst_value = *cst.value_begin<APFloat>();

    // Affine op.
    Value filter = fc_op.filter();
    Value bias = fc_op.bias();
    DenseFPElementsAttr filter_cst, bias_cst;
    if (!matchPattern(filter, m_Constant(&filter_cst))) {
      // The filter maybe quantized, then we should set it to the real constant.
      auto dq = llvm::dyn_cast_or_null<DequantizeOp>(filter.getDefiningOp());
      if (!dq) return failure();
      auto q = llvm::dyn_cast_or_null<QuantizeOp>(dq.input().getDefiningOp());
      if (!q || !matchPattern(q.input(), m_Constant(&filter_cst))) {
        return failure();
      }
      filter = q.input();
    }
    if (!bias.getType().isa<NoneType>() &&
        !matchPattern(bias, m_Constant(&bias_cst)))
      return failure();
    auto binary_op_activation_func =
        binary_op->template getAttrOfType<StringAttr>(
            "fused_activation_function");
    if (!binary_op_activation_func ||
        binary_op_activation_func.getValue() != "NONE")
      return failure();
    ShapedType filter_type = filter_cst.getType();

    if (llvm::isa<AddOp, SubOp>(binary_op)) {
      auto padding = fc_op->template getAttrOfType<StringAttr>("padding");
      if (padding && padding.getValue() != "VALID") return failure();

      // The fusion of add/sub is actually applying the following
      // transformation:
      // w * (x + c) + b => w * x + (w * c + b)
      // so we have to update the bias.
      if (llvm::isa<SubOp>(binary_op)) cst_value.changeSign();

      auto bias_and_slice =
          GetBiasDimAndSliceSize(filter_type.getShape(), fc_op);
      int64_t bias_size = bias_and_slice.first;
      int64_t slice_size = bias_and_slice.second;
      ShapedType new_bias_type =
          RankedTensorType::get({bias_size}, filter_type.getElementType());

      // The new bias should be a 1-D tensor with length equals to the bias
      // dimension of the weight.
      SmallVector<APFloat, 4> new_bias_values;
      if (bias.getType().isa<NoneType>()) {  // none bias, a list of zeros
        new_bias_values.resize(bias_size,
                               APFloat::getZero(cst_value.getSemantics()));
      } else if (bias_cst.getNumElements() == 1) {  // scalar bias, broadcast it
        new_bias_values.resize(bias_size, *bias_cst.value_begin<APFloat>());
      } else if (bias_cst.getNumElements() == bias_size) {  // 1-d bias, copy it
        new_bias_values.insert(new_bias_values.begin(),
                               bias_cst.value_begin<APFloat>(),
                               bias_cst.value_end<APFloat>());
      } else {
        return failure();
      }

      int64_t flatten_index = 0;
      for (auto fp_it = filter_cst.value_begin<APFloat>(),
                fp_end = filter_cst.value_end<APFloat>();
           fp_it != fp_end; ++fp_it) {
        int bias_index = (flatten_index++ / slice_size) % bias_size;

        new_bias_values[bias_index] =
            new_bias_values[bias_index] + *fp_it * cst_value;
      }
      auto new_bias = DenseFPElementsAttr::get(new_bias_type, new_bias_values);
      auto new_bias_op =
          rewriter.create<ConstOp>(fc_op.getLoc(), new_bias_type, new_bias);
      fc_op.setOperand(0, binary_op->getOperand(0));
      fc_op.setOperand(2, new_bias_op);
    } else if (llvm::isa<MulOp, DivOp>(binary_op)) {
      // The fusion of mul/div is actually applying the following
      // transformation:
      // w * (x ' c) + b => (w ' c) x + b
      // so we have to update the weight.
      bool is_mul = llvm::isa<MulOp>(binary_op);
      auto new_filter =
          filter_cst.mapValues(filter_type.getElementType(), [&](APFloat it) {
            return (is_mul ? it * cst_value : it / cst_value).bitcastToAPInt();
          });
      // We recreate the constant op in case it is shared by the other ops. This
      // might increase the model size.
      auto new_filter_op = rewriter.create<ConstOp>(
          fc_op.getLoc(), filter.getType(), new_filter);
      fc_op.setOperand(0, binary_op->getOperand(0));
      if (fc_op.filter() != filter) {
        // This filter goes through quantize and dequantize ops. Then we just
        // need to update the weight to the quantize op.
        filter.replaceAllUsesWith(new_filter_op);
      } else {
        // This filter doesn't go through quantize and dequantize ops, Then
        // we update the weight of the affine op directly.
        fc_op.setOperand(1, new_filter_op);
      }
    } else {
      return failure();
    }
    return success();
  }

 private:
  // Returns the dimension length of the channel dimension and also the slide
  // size by each position in the channel dimension accordingly. tfl.conv2d and
  // tfl.fully_connected has heading channel dimension, but tfl.depthwise_conv2d
  // has tailing channel dimension. This function is to provide a utility to
  // create the above information from the op property.
  static std::pair<int64_t, int64_t> GetBiasDimAndSliceSize(
      ArrayRef<int64_t> filter_shape, AffineOpType op) {
    // Channel dimension index is specified as op property
    auto channel_index_iter = filter_shape.begin();
    std::advance(channel_index_iter, op.GetChannelDimIndex());
    // The slide size is the size of the data in higher dimensions.
    int64_t slice_size =
        std::accumulate(std::next(channel_index_iter), filter_shape.end(), 1,
                        std::multiplies<int64_t>());
    return {*channel_index_iter, slice_size};
  }
};

// If the operand to a broadcastable op is a splat constant, try to replace it
// with a 0-d constant, e.g. before this optimization,
//   %cst = arith.constant dense<1.0> : tensor<16x16x4xf32>
//   %0 = "tfl.conv_2d"...
//   %1 = "tfl.add"(%0, %cst) : (tensor<16x16x4xf32>, tensor<16x16x4xf32>)
// After this optimization:
//   %cst = arith.constant dense<1.0> : tensor<f32>
//   %0 = "tfl.conv_2d"...
//   %1 = "tfl.add"(%0, %cst) : (tensor<16x16x4xf32>, tensor<f32>)
// This pattern can enable more fusing opportunities when the binary op is
// following conv ops.
template <typename BinaryOpType>
struct ScalarizeSplatConstantForBroadcastableOps
    : public OpRewritePattern<BinaryOpType> {
  using OpRewritePattern<BinaryOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(BinaryOpType binary_op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_39_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_39(mht_39_v, 1457, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "matchAndRewrite");

    DenseElementsAttr splat_elements_attr;
    if (!IsScalarizableSplatConstant(binary_op.rhs(), &splat_elements_attr)) {
      return failure();
    }

    constexpr int kSplatOperandIndex = 1;
    auto result_type =
        binary_op.getResult().getType().template cast<ShapedType>();
    mlir::Value non_splat_operand =
        binary_op.getOperand(1 - kSplatOperandIndex);
    auto non_splat_operand_type =
        non_splat_operand.getType().cast<ShapedType>();
    // If the other operand's shape does not equal to the result shape, then we
    // cannot scalarize the splat constant because the result shape relies on
    // the splat constant op's shape for broadcasting.
    if (!non_splat_operand_type.hasStaticShape() ||
        non_splat_operand_type.getShape() != result_type.getShape() ||
        non_splat_operand_type.getRank() > 4) {
      return failure();
    }

    // If non-splat operand is not fusable affine ops, then no need to apply
    // this transformation.
    if (!CanFuseAffineOp(non_splat_operand.getDefiningOp(), binary_op)) {
      return failure();
    }

    // Creates a new scalar constant op using the splat value.
    mlir::Value splat_operand = binary_op.getOperand(kSplatOperandIndex);
    auto scalar_elements_attr = DenseElementsAttr::get(
        RankedTensorType::get({},
                              splat_elements_attr.getType().getElementType()),
        splat_elements_attr.getSplatValue<mlir::Attribute>());

    auto scalar_constant_op = rewriter.create<arith::ConstantOp>(
        splat_operand.getLoc(), scalar_elements_attr.getType(),
        scalar_elements_attr);

    binary_op.setOperand(kSplatOperandIndex, scalar_constant_op);
    return success();
  }

 private:
  // Returns true if this value is a splat constant op which can be scalarized.
  // Also returns the elements attr if this value is indeed a splat constant.
  bool IsScalarizableSplatConstant(mlir::Value value,
                                   DenseElementsAttr *elements_attr) const {
   std::vector<std::string> mht_40_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_40(mht_40_v, 1507, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "IsScalarizableSplatConstant");

    if (!matchPattern(value, m_Constant(elements_attr))) {
      return false;
    }
    auto element_type = value.getType().cast<ShapedType>().getElementType();
    // Ignore per-axis quantized constants because after converting to scalar,
    // we will lose per-axis qantization parameter.
    if (element_type.isa<quant::UniformQuantizedPerAxisType>()) {
      return false;
    }
    if (IsScalar(value)) {
      return false;
    }
    return elements_attr->isSplat();
  }

  // If this type is a scalar shaped type.
  bool IsScalar(mlir::Value value) const {
   std::vector<std::string> mht_41_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_41(mht_41_v, 1527, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "IsScalar");

    auto type = value.getType().dyn_cast<ShapedType>();
    if (!type) {
      return false;
    }
    if (!type.hasStaticShape()) {
      return false;
    }
    return type.getNumElements() == 1;
  }

  // Returns true if we can fuse an affine op with consuming binary op.
  bool CanFuseAffineOp(Operation *affine_op, Operation *binary_op) const {
   std::vector<std::string> mht_42_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_42(mht_42_v, 1542, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "CanFuseAffineOp");

    if (!isa_and_nonnull<TFL::Conv2DOp, TFL::DepthwiseConv2DOp,
                         TFL::FullyConnectedOp>(affine_op)) {
      return false;
    }
    DenseElementsAttr value;
    // Check that bias are constants if not none.
    Value bias = affine_op->getOperand(2);
    if (!bias.getType().isa<NoneType>() &&
        !matchPattern(bias, m_Constant(&value))) {
      return false;
    }
    // If the binary op is mul/div, also check that filter is constant.
    if (isa<TFL::MulOp, TFL::DivOp>(binary_op) &&
        !matchPattern(affine_op->getOperand(1), m_Constant(&value))) {
      return false;
    }

    // We can only fuse F32/BF16.
    auto is_fusable_type = [](Type t) {
   std::vector<std::string> mht_43_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_43(mht_43_v, 1564, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "lambda");

      Type element_type = t;
      if (auto shaped_type = t.dyn_cast<ShapedType>()) {
        element_type = shaped_type.getElementType();
      }
      return element_type.isBF16() || element_type.isF32();
    };
    for (Type t : binary_op->getOperandTypes()) {
      if (!is_fusable_type(t)) {
        return false;
      }
    }

    return true;
  }
};

using ScalarizeSplatConstantForSub =
    ScalarizeSplatConstantForBroadcastableOps<TFL::SubOp>;
using ScalarizeSplatConstantForAdd =
    ScalarizeSplatConstantForBroadcastableOps<TFL::AddOp>;
using ScalarizeSplatConstantForMul =
    ScalarizeSplatConstantForBroadcastableOps<TFL::MulOp>;
using ScalarizeSplatConstantForDiv =
    ScalarizeSplatConstantForBroadcastableOps<TFL::DivOp>;

struct ConvertTrivialTransposeOpToReshapeOp
    : public OpRewritePattern<TFL::TransposeOp> {
  using OpRewritePattern<TFL::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::TransposeOp transpose_op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_44_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_44(mht_44_v, 1598, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "matchAndRewrite");

    auto input_type = transpose_op.input().getType().cast<ShapedType>();
    auto output_type = transpose_op.output().getType().cast<ShapedType>();
    // It's possible to know if the transformation is safe only if the input
    // & output shapes are fully known and permutation is a constant.
    if (!input_type.hasStaticShape() || !output_type.hasStaticShape())
      return failure();
    Value perm = transpose_op.perm();
    DenseElementsAttr perm_values_attr;
    if (!matchPattern(perm, m_Constant(&perm_values_attr))) return failure();

    auto input_shape = input_type.getShape();
    SmallVector<int64_t, 8> perm_values;
    for (const auto &dim : perm_values_attr.getValues<APInt>())
      perm_values.push_back(dim.getSExtValue());

    // This should never happen unless the input graph is malformed.
    if (input_shape.size() != perm_values.size()) {
      transpose_op.emitError(
          "TransposeOP has inconsistent input and perm values.");
    }

    SmallVector<int, 8> old_major_index_ordering;
    SmallVector<int, 8> new_major_index_ordering;
    for (int i = 0, end = input_shape.size(); i < end; i++) {
      if (input_shape[i] != 1) {
        old_major_index_ordering.push_back(i);
      }

      if (input_shape[perm_values[i]] != 1) {
        new_major_index_ordering.push_back(perm_values[i]);
      }
    }
    if (old_major_index_ordering != new_major_index_ordering) {
      return failure();
    }

    // Rewrite.
    Location loc = transpose_op.getLoc();

    SmallVector<int32_t, 8> output_shape_values;
    for (auto dim : output_type.getShape()) {
      output_shape_values.push_back(dim);
    }
    auto type = mlir::RankedTensorType::get(output_shape_values.size(),
                                            rewriter.getIntegerType(32));
    auto new_shape_attr =
        mlir::DenseIntElementsAttr::get(type, output_shape_values);
    auto new_shape = rewriter.create<TF::ConstOp>(loc, new_shape_attr);

    rewriter.replaceOpWithNewOp<TFL::ReshapeOp>(
        transpose_op, transpose_op.output().getType(), transpose_op.input(),
        new_shape);

    return success();
  }
};

// Remove Reshape before FullyConnected when `keep_num_dims=false` and Reshape
// does not alter the last dimension as FullyConnected will collapse all other
// dimensions into a single dimension. For example,
//
//   %shape = arith.constant dense<[1, 128, 64]> : tensor<3xi32>
//   %reshape = tfl.reshape(%input, %shape) // %input: tensor<128x64xf32>
//   %fc = tfl.fully_connected(%reshape, %filter, %bias)
//           {keep_num_dims = false, weights_format = "DEFAULT"}
//
// can be canonicalized to
//
//   %fc = tfl.fully_connected(%input, %filter, %bias)
//           {keep_num_dims = false, weights_format = "DEFAULT"}
struct RemoveReshapeBeforeFullyConnected
    : public OpRewritePattern<TFL::FullyConnectedOp> {
  using OpRewritePattern<TFL::FullyConnectedOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::FullyConnectedOp fully_connected_op,
                                PatternRewriter &) const override {
   std::vector<std::string> mht_45_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_45(mht_45_v, 1677, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "matchAndRewrite");

    auto input = fully_connected_op.input();
    auto input_ty = input.getType().dyn_cast<ShapedType>();
    auto output_ty = fully_connected_op.output()[0]
                         .getType()
                         .template dyn_cast<ShapedType>();
    if (!input_ty.hasStaticShape() ||
        fully_connected_op.weights_format() != "DEFAULT" ||
        fully_connected_op.keep_num_dims() || !output_ty.hasStaticShape() ||
        output_ty.getRank() != 2) {
      return failure();
    }

    auto reshape_op = input.getDefiningOp<TFL::ReshapeOp>();
    if (!reshape_op) return failure();

    // Check if the last dimension does not change after reshape.
    auto reshape_input = reshape_op.input();
    auto reshape_input_ty = reshape_input.getType().dyn_cast<ShapedType>();
    if (!reshape_input_ty.hasStaticShape() || input_ty.getRank() == 0 ||
        reshape_input_ty.getRank() == 0 ||
        input_ty.getDimSize(input_ty.getRank() - 1) !=
            reshape_input_ty.getDimSize(reshape_input_ty.getRank() - 1)) {
      return failure();
    }

    // Connect the input to the one of reshape.
    fully_connected_op.setOperand(0, reshape_input);
    return success();
  }
};

// Remove Reshape after FullyConnected when `keep_num_dims=false`, the Reshape
// does not alter the last dimension and it restores the batch dimensions
// collapsed by the FullyConnected op due to `keep_num_dims=false`. For example,
//
//   // %input: tensor<4x16x32xf32>
//   %fc = tfl.fully_connected(%input, %filter, %bias)
//           {keep_num_dims = false, weights_format = "DEFAULT"}
//   %shape = arith.constant dense<[4, 16, 32]> : tensor<3xi32>
//   %rs = tfl.reshape(%fc, %shape)
//
// can be canonicalized to
//
//   %fc = tfl.fully_connected(%input, %filter, %bias)
//           {keep_num_dims = true, weights_format = "DEFAULT"}
struct RemoveReshapeAfterFullyConnected
    : public OpRewritePattern<TFL::ReshapeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::ReshapeOp reshape_op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_46_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_46(mht_46_v, 1731, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "matchAndRewrite");

    auto fully_connected_op = llvm::dyn_cast_or_null<TFL::FullyConnectedOp>(
        reshape_op.input().getDefiningOp());
    if (!fully_connected_op || fully_connected_op.getNumResults() != 1 ||
        fully_connected_op.weights_format() != "DEFAULT" ||
        fully_connected_op.keep_num_dims())
      return failure();
    if (!reshape_op.input().hasOneUse()) return failure();

    auto input_shape = fully_connected_op.input().getType().cast<ShapedType>();
    auto output_shape = fully_connected_op.getType(0).cast<ShapedType>();
    auto reshape_shape = reshape_op.getType().cast<ShapedType>();
    if (!input_shape.hasStaticShape() || !output_shape.hasStaticShape() ||
        !reshape_shape.hasStaticShape())
      return failure();

    // Check that the reshape doesn't modify the last dimension and it restores
    // the input (batch) dimension with the exception of the feature (last)
    // dimension.
    if (output_shape.getShape().empty() || reshape_shape.getShape().empty() ||
        output_shape.getShape().back() != reshape_shape.getShape().back() ||
        input_shape.getShape().drop_back() !=
            reshape_shape.getShape().drop_back())
      return failure();

    llvm::SmallVector<Type, 1> output_type{reshape_op.getType()};
    rewriter.replaceOpWithNewOp<TFL::FullyConnectedOp>(
        reshape_op, output_type, /*input=*/fully_connected_op.input(),
        /*filter=*/fully_connected_op.filter(),
        /*bias=*/fully_connected_op.bias(),
        /*fused_activation_function=*/
        fully_connected_op.fused_activation_function(),
        /*weights_format=*/fully_connected_op.weights_format(),
        /*keep_num_dims=*/true,
        /*asymmetric_quantize_inputs=*/
        fully_connected_op.asymmetric_quantize_inputsAttr());
    return success();
  }
};

// Fuses Unpack with proceeding Concatenation to Reshape if output type has
// static shape and activation function is none. For example:
//
//   // %input: tensor<1x3x2xf32>
//   %unpack:3 = "tfl.unpack"(%input) {axis = 1 : i32, num = 3 : i32}
//   %res = "tfl.concatenation"(%unpack#0, %unpack#1, %unpack#2)
//        {axis = -1 : i32, fused_activation_function = "NONE"}
//
// can be optimized to
//
//   %cst = arith.constant dense<[1, 6]> : tensor<2xi32>
//   %res = "tfl.reshape"(%input, %cst)
struct FuseUnpackAndConcatToReshape
    : public OpRewritePattern<TFL::ConcatenationOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TFL::ConcatenationOp concat_op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_47_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_47(mht_47_v, 1791, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "matchAndRewrite");

    if (concat_op.fused_activation_function() != "NONE") {
      return failure();
    }

    // Checks all operands come from the same unpack op.
    auto first_operand = concat_op.values().front();
    auto unpack_op =
        dyn_cast_or_null<TFL::UnpackOp>(first_operand.getDefiningOp());
    if (!unpack_op || unpack_op.getNumResults() != concat_op.getNumOperands()) {
      return failure();
    }
    for (auto &index_and_value : llvm::enumerate(concat_op.values())) {
      if (index_and_value.value() !=
          unpack_op.getResult(index_and_value.index())) {
        return failure();
      }
    }

    auto output_type = concat_op.getType().cast<ShapedType>();
    if (!output_type.hasStaticShape()) {
      return failure();
    }

    auto new_shape_array = output_type.getShape();
    // This is to workaround the unnecessary cast i64 -> i32.
    SmallVector<int32_t, 4> new_shape_array_i32;
    for (auto size : new_shape_array) {
      new_shape_array_i32.push_back(static_cast<int32_t>(size));
    }
    auto new_shape = rewriter.create<TFL::ConstOp>(
        concat_op.getLoc(),
        DenseIntElementsAttr::get(
            RankedTensorType::get(new_shape_array_i32.size(),
                                  rewriter.getIntegerType(32)),
            new_shape_array_i32));

    rewriter.replaceOpWithNewOp<TFL::ReshapeOp>(concat_op, output_type,
                                                unpack_op.input(), new_shape);
    return success();
  }
};

// Reduce the K of a TopKV2Op for the following case.
//
// values, indices = tfl.topkv2(%inputs, K)
// %1 = tfl.slice(values, 0, k)
// %2 = tfl.slice(indices,0, k)
// .... (values and indices only used for %1 and %2)
//
// %1 or %2 can be absent. If values and indices are only used here,
// this pattern can be replaced with (conceptually)
//
// %values, %indices = tfl.topkv2(%inputs, k)
// replace all use of %1 with values
// replace all use of %2 with indices
//
struct OptimizeTopK : public OpRewritePattern<TFL::TopKV2Op> {
  using OpRewritePattern::OpRewritePattern;

  // It computes the last dim k of slice size of value.user.
  // If value has no use then return 0.
  llvm::Optional<int32_t> ComputeSliceK(Value value) const {
    if (value.use_empty()) return 0;
    auto slice_op =
        llvm::dyn_cast_or_null<TFL::SliceOp>(value.getUses().begin().getUser());
    // We only match for the case where value is used by SliceOp.
    if (!slice_op) return llvm::None;
    DenseElementsAttr begin;
    DenseElementsAttr size;
    if (!matchPattern(slice_op->getOperand(1), m_Constant(&begin)) ||
        !matchPattern(slice_op->getOperand(2), m_Constant(&size)))
      return llvm::None;

    // Check if "begin" is a zero tensor.
    for (auto begin_idx : begin.getValues<APInt>())
      if (begin_idx != 0) return llvm::None;

    // Check if "size" is equal to slice_op.input.shape except
    // for last dimension.
    // It can be done  by verifying the number of elements:
    // i.e., num_input/input_last_dim = num_result/k
    auto input_ty = value.getType().dyn_cast_or_null<ShapedType>();
    auto result_ty = slice_op.getType().dyn_cast<ShapedType>();
    if (!input_ty || !result_ty) return llvm::None;
    if (!input_ty.hasStaticShape() || !result_ty.hasStaticShape())
      return llvm::None;
    if (!input_ty.getRank() || !result_ty.getRank()) return llvm::None;
    int num_input = input_ty.getNumElements();
    int input_last_dim = input_ty.getShape().back();
    if (input_last_dim < 1) return llvm::None;
    int num_result = result_ty.getNumElements();
    auto size_last = *(--size.value_end<APInt>());
    int32_t k = size_last.getSExtValue();
    if (num_input / input_last_dim * k != num_result) return llvm::None;
    // We don't match sliceOp with last dim size = 0.
    if (!k) return llvm::None;
    return k;
  }

  LogicalResult matchAndRewrite(TFL::TopKV2Op op,
                                PatternRewriter &rewriter) const override {
   std::vector<std::string> mht_48_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_48(mht_48_v, 1895, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "matchAndRewrite");

    auto values = op.values();
    auto indices = op.indices();
    // op.values() and op.indices() cannot be used more than once.
    if (!values.hasOneUse() && !values.use_empty()) return failure();
    if (!indices.hasOneUse() && !indices.use_empty()) return failure();

    auto k_values_or = ComputeSliceK(values);
    auto k_indices_or = ComputeSliceK(indices);
    if (!k_values_or.hasValue() || !k_indices_or.hasValue()) return failure();
    int32_t k_values = k_values_or.getValue();
    int32_t k_indices = k_indices_or.getValue();
    // We don't match two SliceOp with different sizes.
    if (k_values != k_indices && !values.use_empty() && !indices.use_empty())
      return failure();

    // Start replacing.
    auto k = !values.use_empty() ? k_values : k_indices;
    // Build scalar tensor k.
    auto k_ty = mlir::RankedTensorType::get({}, rewriter.getIntegerType(32));
    Value k_cst = rewriter.create<TFL::ConstOp>(
        op.getLoc(), DenseElementsAttr::get(k_ty, k));
    // Compute new result types.
    auto values_ty = values.getType().dyn_cast<ShapedType>();
    auto indices_ty = indices.getType().dyn_cast<ShapedType>();
    auto shape = std::vector<int64_t>();
    for (auto d : values_ty.getShape().drop_back()) {
      shape.push_back(d);
    }
    shape.push_back(static_cast<int64_t>(k));
    auto new_values_ty =
        mlir::RankedTensorType::get(shape, values_ty.getElementType());
    auto new_indices_ty =
        mlir::RankedTensorType::get(shape, indices_ty.getElementType());
    TFL::TopKV2Op top_k_op = rewriter.create<TFL::TopKV2Op>(
        op.getLoc(), new_values_ty, new_indices_ty, op->getOperand(0), k_cst);

    // Remove original ops (topk, Slice, Slice).
    if (!values.use_empty()) {
      auto values_slice_op = llvm::dyn_cast_or_null<TFL::SliceOp>(
          values.getUses().begin().getUser());
      values_slice_op.getResult().replaceAllUsesWith(top_k_op.values());
      values_slice_op.erase();
    }
    if (!indices.use_empty()) {
      auto indices_slice_op = llvm::dyn_cast_or_null<TFL::SliceOp>(
          indices.getUses().begin().getUser());
      indices_slice_op.getResult().replaceAllUsesWith(top_k_op.indices());
      indices_slice_op.erase();
    }
    op.erase();
    return success();
  }
};

using FuseBinaryOpToFollowingFullyConnected =
    FuseBinaryOpToFollowingAffineOp<FullyConnectedOp>;
using FuseBinaryOpToFollowingDepthwiseConv2D =
    FuseBinaryOpToFollowingAffineOp<DepthwiseConv2DOp>;
using FuseBinaryOpToFollowingConv2D = FuseBinaryOpToFollowingAffineOp<Conv2DOp>;

// Adds canonicalization patterns to the list of patterns.
void AddCanonicalizationPatterns(MLIRContext *context,
                                 RewritePatternSet *patterns) {
   std::vector<std::string> mht_49_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_49(mht_49_v, 1961, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "AddCanonicalizationPatterns");

  for (auto op : context->getRegisteredOperations())
    op.getCanonicalizationPatterns(*patterns, context);
}

void OptimizePass::runOnOperation() {
   std::vector<std::string> mht_50_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSlitePStransformsPSoptimizeDTcc mht_50(mht_50_v, 1969, "", "./tensorflow/compiler/mlir/lite/transforms/optimize.cc", "OptimizePass::runOnOperation");

  RewritePatternSet patterns(&getContext());
  auto *ctx = &getContext();
  auto func = getOperation();

  // Merge reshapes into fully connected ops before we start moving them past
  // binary ops.
  RewritePatternSet phase_0_patterns(&getContext());
  phase_0_patterns
      .add<RemoveReshapeAfterFullyConnected, RemoveReshapeBeforeFullyConnected>(
          ctx);
  (void)applyPatternsAndFoldGreedily(func, std::move(phase_0_patterns));

  // Potentially the binary ops might be fused together, like hard_swish, thus
  // we explore these potentially first and then fuse the binary ops with the
  // following ops in a second pattern match.
  TFL::populateWithGenerated(patterns);
  patterns.add<FuseFullyConnectedAndAdd, FuseAddAndFullyConnected,
               FuseFullyConnectedAndMul, FuseMulAndFullyConnected,
               FuseFullyConnectedAndReluX<TFL::ReluOp, kRelu>,
               FuseFullyConnectedAndReluX<TFL::Relu6Op, kRelu6>,
               FuseFullyConnectedAndReluX<TFL::Relu1Op, kRelu1>>(ctx);
  if (enable_canonicalization_) AddCanonicalizationPatterns(ctx, &patterns);
  (void)applyPatternsAndFoldGreedily(func, std::move(patterns));

  // Fuse the binary ops with the following ops.
  RewritePatternSet phase_2_patterns(&getContext());
  TFL::populateWithGenerated(phase_2_patterns);
  phase_2_patterns.add<
      ScalarizeSplatConstantForAdd, ScalarizeSplatConstantForSub,
      ScalarizeSplatConstantForMul, ScalarizeSplatConstantForDiv,
      FuseFullyConnectedAndAdd, FuseAddAndFullyConnected,
      FuseFullyConnectedAndMul, FuseMulAndFullyConnected,
      FuseFullyConnectedAndReluX<TFL::ReluOp, kRelu>,
      FuseFullyConnectedAndReluX<TFL::Relu6Op, kRelu6>,
      FuseFullyConnectedAndReluX<TFL::Relu1Op, kRelu1>,
      FuseBinaryOpToFollowingConv2D, FuseBinaryOpToFollowingDepthwiseConv2D,
      FuseBinaryOpToFollowingFullyConnected, FuseConv2DAndMulWithQDQs,
      FuseDepthwiseConv2DAndMulWithQDQs, ConvertTrivialTransposeOpToReshapeOp,
      RemoveReshapeAfterFullyConnected, RemoveReshapeBeforeFullyConnected,
      FuseUnpackAndConcatToReshape, OptimizeTopK>(ctx);
  if (enable_canonicalization_)
    AddCanonicalizationPatterns(ctx, &phase_2_patterns);
  (void)applyPatternsAndFoldGreedily(func, std::move(phase_2_patterns));
}
}  // namespace

// Creates an instance of the TensorFlow Lite dialect Optimize pass.
std::unique_ptr<OperationPass<FuncOp>> CreateOptimizePass(
    bool enable_canonicalization) {
  return std::make_unique<OptimizePass>(enable_canonicalization);
}

static PassRegistration<OptimizePass> pass;

}  // namespace TFL
}  // namespace mlir
