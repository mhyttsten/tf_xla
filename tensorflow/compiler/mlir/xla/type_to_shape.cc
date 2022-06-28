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
class MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStype_to_shapeDTcc {
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
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStype_to_shapeDTcc(std::vector<std::string> params, int lineNumber, std::string prefix, std::string fileName, std::string functionName) {
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
   ~MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStype_to_shapeDTcc() {
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

#include "tensorflow/compiler/mlir/xla/type_to_shape.h"

#include <string>

#include "mlir/IR/AffineMap.h"  // from @llvm-project
#include "mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/IR/Diagnostics.h"  // from @llvm-project
#include "mlir/IR/Location.h"  // from @llvm-project
#include "mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "tensorflow/compiler/mlir/hlo/include/mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_tensor.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/convert_type.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

using ::int64_t;
using mlir::IntegerType;
using mlir::MemRefType;
using mlir::RankedTensorType;
using mlir::VectorType;
using xla::PrimitiveType;
using xla::ShapeUtil;

namespace xla {

PrimitiveType TypeToPrimitiveType(mlir::Type type) {
   std::vector<std::string> mht_0_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStype_to_shapeDTcc mht_0(mht_0_v, 214, "", "./tensorflow/compiler/mlir/xla/type_to_shape.cc", "TypeToPrimitiveType");

  if (type.isBF16()) {
    return PrimitiveType::BF16;
  } else if (type.isF16()) {
    return PrimitiveType::F16;
  } else if (type.isF32()) {
    return PrimitiveType::F32;
  } else if (type.isF64()) {
    return PrimitiveType::F64;
  } else if (auto complex_type = type.dyn_cast<mlir::ComplexType>()) {
    mlir::Type element_ty = complex_type.getElementType();
    if (element_ty.isF32()) {
      return PrimitiveType::C64;

    } else if (element_ty.isF64()) {
      return PrimitiveType::C128;
    }
    return PrimitiveType::PRIMITIVE_TYPE_INVALID;
  } else if (auto integer_type = type.dyn_cast<mlir::IntegerType>()) {
    bool is_unsigned = integer_type.isUnsigned();
    switch (integer_type.getWidth()) {
      case 1:
        return PrimitiveType::PRED;
      case 8:
        return is_unsigned ? PrimitiveType::U8 : PrimitiveType::S8;
      case 16:
        return is_unsigned ? PrimitiveType::U16 : PrimitiveType::S16;
      case 32:
        return is_unsigned ? PrimitiveType::U32 : PrimitiveType::S32;
      case 64:
        return is_unsigned ? PrimitiveType::U64 : PrimitiveType::S64;
      default:
        return PrimitiveType::PRIMITIVE_TYPE_INVALID;
    }
  }
  return PrimitiveType::PRIMITIVE_TYPE_INVALID;
}

StatusOr<Shape> TypeToShape(
    mlir::Type type, CustomShapeRepresentationFn shape_representation_fn) {
  tensorflow::PartialTensorShape partial_tensor_shape =
      tensorflow::ConvertTypeToTensorShape(type);

  tensorflow::TensorShape fully_defined_tensor_shape;
  if (!partial_tensor_shape.AsTensorShape(&fully_defined_tensor_shape)) {
    return tensorflow::errors::InvalidArgument(
        "XLA HLO only allows fully-defined shape");
  }

  tensorflow::DataType dtype;
  TF_RETURN_IF_ERROR(tensorflow::ConvertToDataType(type, &dtype));

  return shape_representation_fn(fully_defined_tensor_shape, dtype);
}

Shape TypeToShape(mlir::Type type) {
   std::vector<std::string> mht_1_v;
   MHTracer_DTPStensorflowPScompilerPSmlirPSxlaPStype_to_shapeDTcc mht_1(mht_1_v, 272, "", "./tensorflow/compiler/mlir/xla/type_to_shape.cc", "TypeToShape");

  PrimitiveType ptype = TypeToPrimitiveType(type);
  if (ptype != PrimitiveType::PRIMITIVE_TYPE_INVALID)
    return ShapeUtil::MakeShape(ptype, {});

  if (type.isIntOrFloat()) {
    auto* context = type.getContext();
    mlir::emitError(mlir::UnknownLoc::get(context))
        << "lowering should have been handled by primitive type lowering for "
        << debugString(type);
  } else if (auto v = type.dyn_cast<mlir::VectorType>()) {
    llvm::SmallVector<int64_t, 4> span(v.getShape().begin(),
                                       v.getShape().end());
    mlir::Type element_type = v.getElementType();
    PrimitiveType primitive_type = TypeToPrimitiveType(element_type);
    if (primitive_type != PrimitiveType::PRIMITIVE_TYPE_INVALID)
      return ShapeUtil::MakeShape(primitive_type, span);
  } else if (auto m = type.dyn_cast<mlir::MemRefType>()) {
    llvm::SmallVector<int64_t, 6> span(m.getShape().begin(),
                                       m.getShape().end());
    mlir::Type element_type = m.getElementType();
    // Treat a memref of a vector as if it was a memref of primitive type with
    // the vector dimensions at the end.
    if (auto v = element_type.dyn_cast<mlir::VectorType>()) {
      element_type = v.getElementType();
      span.insert(span.end(), v.getShape().begin(), v.getShape().end());
    }
    PrimitiveType primitive_type = TypeToPrimitiveType(element_type);
    if (primitive_type == PrimitiveType::PRIMITIVE_TYPE_INVALID) return {};
    // For the primitive type case, the shape of the memref is similar to the
    // vector type case (i.e., it is, modulo the layout, the same dimensions
    // and primitive type).
    if (m.getLayout().isIdentity())
      return ShapeUtil::MakeShape(primitive_type, span);

    llvm::SmallVector<int64_t, 4> strides;
    int64_t offset;
    if (failed(mlir::getStridesAndOffset(m, strides, offset))) return {};

    llvm::SmallVector<std::pair<int64_t, int>, 4> strides_with_indices;
    for (const auto& e : llvm::enumerate(strides)) {
      strides_with_indices.push_back({e.value(), e.index()});
    }
    std::stable_sort(strides_with_indices.begin(), strides_with_indices.end());

    llvm::SmallVector<int64_t, 4> minor_to_major;
    int64_t stride = 1;
    for (const auto& pr : strides_with_indices) {
      minor_to_major.push_back(pr.second);

      // Either the affine map is not perfectly strided, or the dimensions
      // recovered from strides don't match the actual dimensions in shapes.
      if (stride != pr.first && m.getShape()[pr.second] != 1) return {};

      stride *= m.getShape()[pr.second];
    }

    llvm::SmallVector<int64_t, 4> dimensions(m.getShape().begin(),
                                             m.getShape().end());
    return ::xla::ShapeUtil::MakeShapeWithLayout(primitive_type, dimensions,
                                                 minor_to_major);
  } else if (auto t = type.dyn_cast<mlir::RankedTensorType>()) {
    // TODO(jpienaar): This is only handling the base case with primitive
    // element type.
    llvm::SmallVector<int64_t, 4> span(t.getShape().begin(),
                                       t.getShape().end());
    // Only fully static shapes are supported.
    // TODO(b/115638799): Update once xla::Shape can support dynamic shapes.
    if (std::find(t.getShape().begin(), t.getShape().end(), -1) !=
        t.getShape().end())
      return {};
    mlir::Type element_type = t.getElementType();
    PrimitiveType primitive_type = TypeToPrimitiveType(element_type);
    // Only primitive element type supported.
    if (primitive_type != PrimitiveType::PRIMITIVE_TYPE_INVALID)
      return ShapeUtil::MakeShape(primitive_type, span);
  } else if (auto tuple_type = type.dyn_cast<mlir::TupleType>()) {
    llvm::SmallVector<Shape, 4> shapes;
    shapes.reserve(tuple_type.size());
    for (mlir::Type sub_type : tuple_type.getTypes()) {
      shapes.push_back(TypeToShape(sub_type));
    }
    return ShapeUtil::MakeTupleShape(shapes);

  } else if (type.isa<mlir::mhlo::TokenType>()) {
    return ShapeUtil::MakeTokenShape();
  }

  // Return empty XLA shape to signify error. No MLIR Type maps to a empty
  // Shape.
  return {};
}

}  // namespace xla
