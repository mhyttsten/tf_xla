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
class MHTracer_DTPStensorflowPSdtensorPSmlirPSshape_utilsDTcc {
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
   MHTracer_DTPStensorflowPSdtensorPSmlirPSshape_utilsDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPSdtensorPSmlirPSshape_utilsDTcc() {
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

#include "tensorflow/dtensor/mlir/shape_utils.h"

#include "llvm/Support/FormatVariadic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OperationSupport.h"  // from @llvm-project
#include "mlir/IR/Value.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/tensorflow/utils/shape_inference_utils.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/dtensor/cc/constants.h"
#include "tensorflow/dtensor/mlir/ir/tf_dtensor.h"
#include "tensorflow/dtensor/mlir/value_utils.h"

namespace tensorflow {
namespace dtensor {

StatusOr<llvm::ArrayRef<int64_t>> ExtractGlobalInputShape(
    mlir::OpOperand& input_value) {
  const int operand_index = input_value.getOperandNumber();
  auto input_defining_op = input_value.get().getDefiningOp();

  if (input_defining_op) {
    if (auto layout_op =
            llvm::dyn_cast<mlir::TF::DTensorLayout>(input_defining_op)) {
      auto global_shape = layout_op.global_shape();
      if (!global_shape)
        return errors::Internal("global_shape does not have static rank");
      return *global_shape;
    }
    return ExtractGlobalOutputShape(input_value.get().cast<mlir::OpResult>());
  }

  // If we reach this point, we're working with a function argument.
  auto op = input_value.getOwner();
  auto enclosing_function = op->getParentOfType<mlir::func::FuncOp>();
  if (!enclosing_function)
    return errors::InvalidArgument(
        llvm::formatv("Could not find global shape of {0}-th input to op: {1}",
                      operand_index, op->getName())
            .str());

  auto block_arg = input_value.get().dyn_cast<mlir::BlockArgument>();
  auto global_shape_attr =
      enclosing_function.getArgAttrOfType<mlir::TF::ShapeAttr>(
          block_arg.getArgNumber(), kGlobalShapeDialectAttr);
  if (!global_shape_attr)
    return errors::InvalidArgument(
        "`tf._global_shape` attribute of operation not found.");

  return global_shape_attr.getShape();
}

StatusOr<llvm::ArrayRef<int64_t>> ExtractGlobalOutputShape(
    mlir::OpResult result_value) {
  auto op = result_value.getOwner();
  const int output_index = result_value.getResultNumber();

  if (op->getOpResult(output_index).hasOneUse()) {
    auto user = op->getOpResult(output_index).getUses().begin().getUser();
    if (auto layout_op = mlir::dyn_cast<mlir::TF::DTensorLayout>(user)) {
      auto global_shape = layout_op.global_shape();
      if (!global_shape)
        return errors::Internal("global_shape does not have static rank");
      return *global_shape;
    }
  }

  auto global_shape_attr = op->getAttrOfType<mlir::ArrayAttr>(kGlobalShape);
  if (!global_shape_attr)
    return errors::InvalidArgument(
        "`_global_shape` attribute of operation not found.");

  const int num_results = op->getNumResults();
  assert(global_shape_attr.size() == num_results);

  if (output_index >= op->getNumResults())
    return errors::InvalidArgument(
        llvm::formatv("Requested global shape of {0} output but op has only "
                      "{1} return values.",
                      output_index, num_results)
            .str());

  auto shape_attr = global_shape_attr[output_index];
  return shape_attr.cast<mlir::TF::ShapeAttr>().getShape();
}

namespace {

// Extracts attributes from a MLIR operation, including derived attributes, into
// one NamedAttrList.
mlir::NamedAttrList GetAllAttributesFromOperation(mlir::Operation* op) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSshape_utilsDTcc mht_0(mht_0_v, 280, "", "./tensorflow/dtensor/mlir/shape_utils.cc", "GetAllAttributesFromOperation");

  mlir::NamedAttrList attr_list;
  attr_list.append(op->getAttrDictionary().getValue());

  if (auto derived = llvm::dyn_cast<mlir::DerivedAttributeOpInterface>(op)) {
    auto materialized = derived.materializeDerivedAttributes();
    attr_list.append(materialized.getValue());
  }

  return attr_list;
}

// Infers output shape of `op` given its local operand shape. For shape
// inference function that requires input operation to be a constant, if input
// operation is `DTensorLayout` op, then we use input of DTensorLayout op
// instead for correct constant matching.
mlir::LogicalResult InferShapeOfTFOpWithCustomOperandConstantFn(
    llvm::Optional<mlir::Location> location, mlir::Operation* op,
    int64_t graph_version,
    llvm::SmallVectorImpl<mlir::ShapedTypeComponents>& inferred_return_shapes) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSshape_utilsDTcc mht_1(mht_1_v, 302, "", "./tensorflow/dtensor/mlir/shape_utils.cc", "InferShapeOfTFOpWithCustomOperandConstantFn");

  if (auto type_op = llvm::dyn_cast<mlir::InferTypeOpInterface>(op)) {
    auto attributes = GetAllAttributesFromOperation(op);
    llvm::SmallVector<mlir::Type, 4> inferred_return_types;
    auto result = type_op.inferReturnTypes(
        op->getContext(), location, op->getOperands(),
        mlir::DictionaryAttr::get(op->getContext(), attributes),
        op->getRegions(), inferred_return_types);
    if (failed(result)) return mlir::failure();

    inferred_return_shapes.resize(inferred_return_types.size());
    for (const auto& inferred_return_type :
         llvm::enumerate(inferred_return_types)) {
      if (auto shaped_type =
              inferred_return_type.value().dyn_cast<mlir::ShapedType>()) {
        if (shaped_type.hasRank()) {
          inferred_return_shapes[inferred_return_type.index()] =
              mlir::ShapedTypeComponents(shaped_type.getShape(),
                                         shaped_type.getElementType());
        } else {
          inferred_return_shapes[inferred_return_type.index()] =
              mlir::ShapedTypeComponents(shaped_type.getElementType());
        }
      }
    }

    return mlir::success();
  }

  if (auto shape_type_op =
          llvm::dyn_cast<mlir::InferShapedTypeOpInterface>(op)) {
    auto attributes = GetAllAttributesFromOperation(op);
    return shape_type_op.inferReturnTypeComponents(
        op->getContext(), location, op->getOperands(),
        mlir::DictionaryAttr::get(op->getContext(), attributes),
        op->getRegions(), inferred_return_shapes);
  }

  // If `operand` is from DTensorLayout op, use input value of DTensorLayout op
  // instead.
  auto operand_as_constant_fn = [](mlir::Value operand) -> mlir::Attribute {
    while (auto input_op = llvm::dyn_cast_or_null<mlir::TF::DTensorLayout>(
               operand.getDefiningOp())) {
      operand = input_op.input();
    }

    mlir::Attribute attr;
    if (matchPattern(operand, m_Constant(&attr))) return attr;
    return nullptr;
  };

  auto op_result_as_shape_fn =
      [](shape_inference::InferenceContext& ic,
         mlir::OpResult op_result) -> shape_inference::ShapeHandle {
    auto rt = op_result.getType().dyn_cast<mlir::RankedTensorType>();
    if (!rt || rt.getRank() != 1 || !rt.hasStaticShape()) return {};

    std::vector<shape_inference::DimensionHandle> dims(rt.getDimSize(0),
                                                       ic.UnknownDim());
    mlir::Attribute attr;
    if (matchPattern(op_result, m_Constant(&attr))) {
      auto elements = attr.dyn_cast<mlir::DenseIntElementsAttr>();
      if (elements)
        for (const auto& element :
             llvm::enumerate(elements.getValues<llvm::APInt>()))
          dims[element.index()] = ic.MakeDim(element.value().getSExtValue());
    }
    return ic.MakeShape(dims);
  };

  auto result_element_type_fn = [](int) -> mlir::Type { return nullptr; };

  return mlir::TF::InferReturnTypeComponentsForTFOp(
      location, op, graph_version, operand_as_constant_fn,
      op_result_as_shape_fn, result_element_type_fn, inferred_return_shapes);
}

}  // namespace

mlir::Operation* InferSPMDExpandedLocalShape(mlir::Operation* op) {
   std::vector<std::string> mht_2_v;
   MHTracer_DTPStensorflowPSdtensorPSmlirPSshape_utilsDTcc mht_2(mht_2_v, 384, "", "./tensorflow/dtensor/mlir/shape_utils.cc", "InferSPMDExpandedLocalShape");

  llvm::SmallVector<mlir::ShapedTypeComponents, 4> inferred_return_types;
  (void)InferShapeOfTFOpWithCustomOperandConstantFn(
      op->getLoc(), op, TF_GRAPH_DEF_VERSION, inferred_return_types);
  assert(inferred_return_types.size() == op->getNumResults());

  for (auto it : llvm::zip(inferred_return_types, op->getOpResults())) {
    const auto& return_type = std::get<0>(it);
    auto& op_result = std::get<1>(it);
    const auto element_type =
        op_result.getType().cast<mlir::TensorType>().getElementType();

    if (return_type.hasRank()) {
      op_result.setType(
          mlir::RankedTensorType::get(return_type.getDims(), element_type));
    } else {
      op_result.setType(mlir::UnrankedTensorType::get(element_type));
    }
  }

  return op;
}

StatusOr<llvm::ArrayRef<int64_t>> GetShapeOfValue(const mlir::Value& value,
                                                  bool fail_on_dynamic) {
  // Getting the subtype or self allows supporting extracting the underlying
  // shape that variant or resource tensors point to.
  mlir::Type type = GetSubtypeOrSelf(value);
  if (auto ranked_type = type.dyn_cast<mlir::RankedTensorType>()) {
    if (ranked_type.hasStaticShape() || !fail_on_dynamic)
      return ranked_type.getShape();
    else
      return errors::InvalidArgument("value shape is not static");
  }
  return errors::InvalidArgument("value type is not a RankedTensorType");
}

StatusOr<llvm::ArrayRef<int64_t>> GetGlobalShapeOfValueFromDTensorLayout(
    const mlir::Value& value) {
  if (value.isa<mlir::OpResult>() &&
      mlir::isa<mlir::TF::DTensorLayout>(value.getDefiningOp())) {
    auto layout_op = mlir::cast<mlir::TF::DTensorLayout>(value.getDefiningOp());
    if (layout_op.global_shape()) return layout_op.global_shape().getValue();
  } else if (value.hasOneUse() &&
             mlir::isa<mlir::TF::DTensorLayout>(*value.getUsers().begin())) {
    auto layout_op =
        mlir::cast<mlir::TF::DTensorLayout>(*value.getUsers().begin());
    if (layout_op.global_shape()) return layout_op.global_shape().getValue();
  }
  return errors::InvalidArgument(
      "consumer or producer of value is not a DTensorLayout");
}

}  // namespace dtensor
}  // namespace tensorflow
